"""Anchor Write, cross-variable attention, and native-position Read."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(logits, mask):
    """A completely masked row has zero attention weights."""
    mask = mask.bool()
    masked = logits.masked_fill(~mask, float("-inf"))
    maximum = masked.amax(dim=-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    numerator = torch.exp(masked - maximum) * mask.to(logits.dtype)
    denominator = numerator.sum(dim=-1, keepdim=True)
    return numerator / denominator.clamp_min(torch.finfo(logits.dtype).eps)


class CrossVariableAlignment(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.num_anchors = config.num_anchors
        self.anchor_embeddings = nn.Parameter(
            torch.empty(config.num_anchors, config.hidden_dim))
        self.variable_embedding = nn.Parameter(
            torch.empty(config.num_variables, config.hidden_dim))
        nn.init.normal_(self.anchor_embeddings, std=.02)
        nn.init.normal_(self.variable_embedding, std=.02)
        # Frozen uniform gaps retain the reference state layout and center arithmetic.
        self.raw_reference_gaps = nn.Parameter(
            torch.zeros(config.num_anchors), requires_grad=False)
        self.writer_query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.writer_key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.writer_value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.writer_output = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.variable_query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.variable_key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.variable_value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.variable_output = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.reader_query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.reader_key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.reader_value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.reader_output = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.raw_writer_bandwidth = nn.Parameter(torch.full((config.num_heads,), -1.5))
        self.raw_reader_bandwidth = nn.Parameter(torch.full((config.num_heads,), -1.5))
        self.variable_norm = nn.LayerNorm(config.hidden_dim)
        self.context_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def time_references(self):
        gaps = F.softplus(self.raw_reference_gaps) + 1e-4
        cumulative = torch.cumsum(gaps, dim=0)
        return (cumulative - .5 * gaps) / gaps.sum()

    @staticmethod
    def native_coordinates(scales, lengths, steps):
        index = torch.arange(
            steps, dtype=scales.dtype, device=scales.device).view(1, 1, steps)
        centers = (index + .5) * scales.unsqueeze(-1)
        horizon = (scales * lengths.to(scales.dtype)).amax(dim=1, keepdim=True).clamp_min(1e-6)
        return centers / horizon.view(scales.shape[0], 1, 1)

    def split_heads(self, tensor):
        return tensor.view(*tensor.shape[:-1], self.num_heads, self.head_dim)

    def write(self, hidden, visible, coordinates, references):
        batch, variables, _, channels = hidden.shape
        value = self.split_heads(self.writer_value(hidden)).permute(0, 1, 3, 2, 4)
        query = self.split_heads(self.writer_query(self.anchor_embeddings)).permute(1, 0, 2)
        key = self.split_heads(self.writer_key(hidden)).permute(0, 1, 3, 2, 4)
        scores = torch.einsum("hle,bdhke->bdhlk", query, key) / math.sqrt(self.head_dim)
        distance = coordinates.unsqueeze(-2) - references.view(1, 1, -1, 1)
        bandwidth = F.softplus(self.raw_writer_bandwidth) + 1e-4
        scores = scores - .5 * (distance.unsqueeze(2) / bandwidth.view(1, 1, -1, 1, 1)).square()
        weights = masked_softmax(scores, visible[:, :, None, None, :].expand_as(scores))
        memory = torch.einsum("bdhlk,bdhke->bdhle", weights, value)
        memory = memory.permute(0, 1, 3, 2, 4).reshape(
            batch, variables, self.num_anchors, channels)
        memory = self.writer_output(memory)
        support = visible.any(dim=-1, keepdim=True).expand(batch, variables, self.num_anchors)
        return memory * support[..., None].to(memory.dtype), support

    def mix_variables(self, memory, support):
        batch, variables, anchors, channels = memory.shape
        queries = memory + self.anchor_embeddings.view(1, 1, anchors, channels)
        queries = queries + self.variable_embedding.view(1, variables, 1, channels)
        query = self.split_heads(self.variable_query(queries)).permute(0, 2, 3, 1, 4)
        key = self.split_heads(self.variable_key(memory)).permute(0, 2, 3, 1, 4)
        value = self.split_heads(self.variable_value(memory)).permute(0, 2, 3, 1, 4)
        scores = torch.einsum("blhqd,blhsd->blhqs", query, key) / math.sqrt(self.head_dim)
        mask = support.permute(0, 2, 1)[:, :, None, None, :].expand_as(scores)
        weights = masked_softmax(scores, mask)
        context = torch.einsum("blhqs,blhse->blhqe", weights, value)
        context = context.permute(0, 3, 1, 2, 4).reshape(batch, variables, anchors, channels)
        context = self.variable_output(context)
        any_source = support.any(dim=1)
        mixed = self.variable_norm(memory + self.dropout(context))
        return mixed * any_source[:, None, :, None].to(mixed.dtype), any_source

    def read(self, hidden, memory, support, coordinates, references):
        batch, variables, steps, channels = hidden.shape
        keys = memory + self.anchor_embeddings.view(1, 1, self.num_anchors, channels)
        keys = keys + self.variable_embedding.view(1, variables, 1, channels)
        value = self.split_heads(self.reader_value(memory)).permute(0, 1, 3, 2, 4)
        query = self.split_heads(self.reader_query(hidden)).permute(0, 1, 3, 2, 4)
        key = self.split_heads(self.reader_key(keys)).permute(0, 1, 3, 2, 4)
        scores = torch.einsum("bdhke,bdhle->bdhkl", query, key) / math.sqrt(self.head_dim)
        distance = coordinates.unsqueeze(-1) - references.view(1, 1, 1, -1)
        bandwidth = F.softplus(self.raw_reader_bandwidth) + 1e-4
        scores = scores - .5 * (distance.unsqueeze(2) / bandwidth.view(1, 1, -1, 1, 1)).square()
        weights = masked_softmax(scores, support[:, None, None, None, :].expand_as(scores))
        context = torch.einsum("bdhkl,bdhle->bdhke", weights, value)
        context = context.permute(0, 1, 3, 2, 4).reshape(batch, variables, steps, channels)
        return self.context_norm(self.reader_output(context))

    def forward(self, hidden, visible, valid, scales, lengths):
        coordinates = self.native_coordinates(scales, lengths, hidden.shape[-2])
        references = self.time_references().to(hidden.device, hidden.dtype)
        memory, support = self.write(hidden, visible & valid, coordinates, references)
        memory, support = self.mix_variables(memory, support)
        context = self.read(hidden, memory, support, coordinates, references)
        return self.dropout(context)
