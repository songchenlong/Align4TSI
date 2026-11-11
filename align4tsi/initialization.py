"""Preserve the random initialization stream used by the reference implementation."""
from torch import nn


def reserve_encoder_initialization(config):
    """Consume the reference constructor's preliminary encoder draws.

    That encoder was replaced before training. Retaining its RNG consumption,
    without retaining the module, pairs fresh seeds with the reported model.
    """
    nn.TransformerEncoderLayer(
        d_model=config.hidden_dim,
        nhead=config.num_heads,
        dim_feedforward=config.ff_dim,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
