"""Shared pre-normalized temporal context with local and global attention heads."""
import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_windows = config.head_windows
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        self.attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.hidden_dim))
        self.ffn_dropout = nn.Dropout(config.dropout)

    def attention_mask(self, valid):
        steps = valid.shape[-1]
        position = torch.arange(steps, device=valid.device)
        distance = (position[:, None] - position[None, :]).abs()
        masks = [
            torch.zeros_like(distance, dtype=torch.bool) if w is None else distance > w
            for w in self.head_windows
        ]
        blocked = torch.stack(masks)[None].expand(valid.shape[0], -1, -1, -1)
        # Padding queries get one fallback key; their outputs are discarded below.
        fallback = torch.ones_like(blocked)
        fallback[..., 0] = False
        return torch.where((~valid)[:, None, :, None], fallback, blocked)

    def forward(self, hidden, valid):
        mask = self.attention_mask(valid)
        mask = mask.reshape(
            hidden.shape[0] * self.num_heads, hidden.shape[1], hidden.shape[1])
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask,
            key_padding_mask=~valid, need_weights=False)
        attended = torch.where(valid[..., None], attended, torch.zeros_like(attended))
        hidden = hidden + self.attention_dropout(attended)
        hidden = hidden * valid[..., None].to(hidden.dtype)
        hidden = hidden + self.ffn_dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden * valid[..., None].to(hidden.dtype)


class TemporalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TemporalBlock(config) for _ in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, hidden, valid):
        batch, variables, steps, channels = hidden.shape
        output = hidden.reshape(batch * variables, steps, channels)
        mask = valid.reshape(batch * variables, steps).bool()
        for layer in self.layers:
            output = layer(output, mask)
        output = self.output_norm(output) * mask[..., None].to(output.dtype)
        return output.reshape(batch, variables, steps, channels)
