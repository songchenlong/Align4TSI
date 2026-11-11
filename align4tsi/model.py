"""Complete Align4TSI over padded, variable-specific native interval sequences."""
import torch
from torch import nn

from .alignment import CrossVariableAlignment
from .config import Align4TSIConfig
from .initialization import reserve_encoder_initialization
from .temporal import TemporalEncoder
from .training import binary_mask


class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(5, config.hidden_dim)
        self.variable_embedding = nn.Embedding(config.num_variables, config.hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(config.hidden_dim))
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, values, visible, query, valid, scales, lengths):
        batch, variables, steps = values.shape
        condition = visible.to(values.dtype)
        target = query.to(values.dtype)
        valid_float = valid.to(values.dtype)
        safe = torch.where(visible, values, torch.zeros_like(values))
        position = torch.arange(
            steps, device=values.device, dtype=values.dtype).view(1, 1, steps)
        position = position / (lengths.to(values.dtype) - 1).clamp_min(1).unsqueeze(-1)
        position = position * valid_float
        log_scale = torch.log1p(scales).unsqueeze(-1).expand(-1, -1, steps)
        features = torch.stack([safe, condition, target, position, log_scale], dim=-1)
        hidden = self.projection(features)
        variable_ids = torch.arange(variables, device=values.device)
        hidden = hidden + self.variable_embedding(variable_ids).view(1, variables, 1, -1)
        hidden = hidden + target.unsqueeze(-1) * self.mask_token.view(1, 1, 1, -1)
        return hidden * valid_float.unsqueeze(-1)


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, hidden):
        return self.network(hidden).squeeze(-1)


class Align4TSI(nn.Module):
    """Predict native slots from visible values, query flags and known geometry.

    Values and masks have shape [B,D,K]. Scales and native lengths have shape
    [D] or [B,D]. The query mask marks requested positions, never their values.
    """

    def __init__(self, config: Align4TSIConfig):
        super().__init__()
        self.config = config
        self.embedding = InputEmbedding(config)
        reserve_encoder_initialization(config)
        # Preserve both the initialization stream and optimizer parameter order.
        self.add_module("temporal", None)
        self.prediction_head = PredictionHead(config.hidden_dim)
        self.alignment = CrossVariableAlignment(config)
        self.temporal = TemporalEncoder(config)

    def prepare_inputs(self, values, conditioning_mask, query_mask, valid_mask, scales, lengths):
        if (
            values.ndim != 3 or min(values.shape) < 1
            or values.shape[1] != self.config.num_variables
        ):
            raise ValueError("values must have shape [B,num_variables,K] with positive dimensions")
        if (
            not values.is_floating_point()
            or values.dtype != self.embedding.projection.weight.dtype
        ):
            raise TypeError("values must be floating point and match the model parameter dtype")
        visible = binary_mask(
            conditioning_mask, values.shape, "conditioning_mask").to(values.device)
        query = binary_mask(query_mask, values.shape, "query_mask").to(values.device)
        valid = binary_mask(valid_mask, values.shape, "valid_mask").to(values.device)
        if bool(((visible | query) & ~valid).any()) or bool((visible & query).any()):
            raise ValueError("Conditioning and query masks must be disjoint subsets of valid positions")
        if bool((~valid.any(dim=-1)).any()):
            raise ValueError("Every variable sequence must have a valid native slot")
        if not bool(torch.isfinite(values[visible]).all()):
            raise ValueError("Conditioning-visible values must be finite")
        metadata = []
        for value, name in [(scales, "scale_hours"), (lengths, "variable_lengths")]:
            value = torch.as_tensor(value, device=values.device)
            if value.ndim == 1:
                value = value.unsqueeze(0).expand(values.shape[0], -1)
            if (
                value.shape != values.shape[:2]
                or not bool(torch.isfinite(value).all()) or bool((value <= 0).any())
            ):
                raise ValueError(
                    f"{name} must be finite, positive, and have shape [D] or [B,D]")
            metadata.append(value)
        scales, lengths = metadata
        if (
            bool((lengths != lengths.floor()).any())
            or bool((lengths > values.shape[-1]).any())
        ):
            raise ValueError("Native lengths must be integers between 1 and padded K")
        slots = torch.arange(values.shape[-1], device=values.device)[None, None]
        if not torch.equal(valid, slots < lengths[..., None]):
            raise ValueError(
                "valid_mask must be the contiguous native-slot prefix given by variable_lengths; "
                "represent missing observations in conditioning_mask, not valid_mask")
        return visible, query, valid, scales.to(values.dtype), lengths

    def forward(
        self,
        values: torch.Tensor,
        conditioning_mask: torch.Tensor,
        query_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        scale_hours: torch.Tensor,
        variable_lengths: torch.Tensor,
    ) -> torch.Tensor:
        visible, query, valid, scales, lengths = self.prepare_inputs(
            values, conditioning_mask, query_mask, valid_mask, scale_hours, variable_lengths)
        hidden = self.embedding(values, visible, query, valid, scales, lengths)
        temporal = self.temporal(hidden, valid)
        native = temporal * valid[..., None].to(temporal.dtype)
        context = self.alignment(native, visible, valid, scales, lengths)
        fused = (self.config.fusion_weight * native + context) * valid[..., None].to(native.dtype)
        return self.prediction_head(fused) * valid.to(values.dtype)

    @torch.no_grad()
    def impute(
        self,
        values: torch.Tensor,
        conditioning_mask: torch.Tensor,
        query_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        scale_hours: torch.Tensor,
        variable_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Fill unavailable slots while retaining visible values; call eval() first."""
        prediction = self(
            values, conditioning_mask, query_mask, valid_mask, scale_hours, variable_lengths)
        visible = conditioning_mask.bool().to(values.device)
        valid = valid_mask.bool().to(values.device)
        return torch.where(visible, values, prediction) * valid.to(values.dtype)
