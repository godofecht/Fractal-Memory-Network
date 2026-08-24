"""Attention mechanisms for the Fractal Memory Network."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAttention(nn.Module):
    """Causal self-attention with an optional recurrent memory prefix."""

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.dropout_prob)

        self.use_relative_positions = config.use_relative_positions
        if self.use_relative_positions:
            self.relative_positions_encoding = RelativePositionEncoding(
                config.max_relative_position,
                self.attention_head_size,
            )

    def transpose_for_scores(self, x):
        shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, memory_states=None):
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, sequence, hidden)")

        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_size}")

        memory_length = 0
        key_value_states = hidden_states
        if memory_states is not None:
            if memory_states.ndim != 3:
                raise ValueError("memory_states must have shape (batch, sequence, hidden)")
            if memory_states.size(0) != batch_size or memory_states.size(2) != hidden_size:
                raise ValueError("memory_states batch/hidden dimensions must match hidden_states")
            memory_length = memory_states.size(1)
            key_value_states = torch.cat((memory_states, hidden_states), dim=1)

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(key_value_states))
        value_layer = self.transpose_for_scores(self.value(key_value_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.use_relative_positions:
            attention_scores = attention_scores + self.relative_positions_encoding(
                query_layer,
                memory_length,
            )
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        current_future = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            diagonal=1,
        )
        if memory_length:
            memory_visible = torch.zeros(
                sequence_length,
                memory_length,
                dtype=torch.bool,
                device=hidden_states.device,
            )
            causal_mask = torch.cat((memory_visible, current_future), dim=-1)
        else:
            causal_mask = current_future
        attention_scores = attention_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0),
            torch.finfo(attention_scores.dtype).min,
        )

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("attention_mask must have shape (batch, current_sequence_length)")
            current_valid = attention_mask.to(dtype=torch.bool)
            if memory_length:
                memory_valid = torch.ones(
                    batch_size,
                    memory_length,
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                key_valid = torch.cat((memory_valid, current_valid), dim=-1)
            else:
                key_valid = current_valid
            attention_scores = attention_scores.masked_fill(
                ~key_valid[:, None, None, :],
                torch.finfo(attention_scores.dtype).min,
            )

        attention_probs = self.dropout(F.softmax(attention_scores, dim=-1))
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, sequence_length, self.all_head_size)
        return context_layer, attention_probs


class RelativePositionEncoding(nn.Module):
    def __init__(self, max_relative_position, hidden_size):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.rel_embeddings = nn.Parameter(
            torch.empty(2 * max_relative_position + 1, hidden_size)
        )
        nn.init.xavier_uniform_(self.rel_embeddings)

    def forward(self, query_layer, memory_length=0):
        query_length = query_layer.size(2)
        device = query_layer.device

        query_positions = torch.arange(query_length, device=device)
        key_positions = torch.arange(-memory_length, query_length, device=device)
        distances = key_positions.unsqueeze(0) - query_positions.unsqueeze(1)
        distances = distances.clamp(
            -self.max_relative_position,
            self.max_relative_position,
        ) + self.max_relative_position

        relative_embeddings = F.embedding(distances.long(), self.rel_embeddings)
        return torch.einsum("bhqd,qkd->bhqk", query_layer, relative_embeddings)
