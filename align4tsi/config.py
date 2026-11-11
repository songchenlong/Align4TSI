from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Align4TSIConfig:
    num_variables: int
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 128
    num_anchors: int = 96
    head_windows: tuple[int | None, ...] = (1, 2, 4, None)
    fusion_weight: float = 1.5
    dropout: float = 0.0

    def __post_init__(self):
        dimensions = (
            "num_variables", "hidden_dim", "num_heads", "num_layers", "ff_dim", "num_anchors"
        )
        for name in dimensions:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        windows = tuple(self.head_windows)
        if len(windows) != self.num_heads:
            raise ValueError("head_windows must provide one window per attention head")
        if any(
            w is not None and (isinstance(w, bool) or not isinstance(w, int) or w < 0)
            for w in windows
        ):
            raise ValueError("Each window must be a nonnegative integer or None for a global head")
        if not math.isfinite(self.fusion_weight) or self.fusion_weight <= 0:
            raise ValueError("fusion_weight must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        object.__setattr__(self, "head_windows", windows)
