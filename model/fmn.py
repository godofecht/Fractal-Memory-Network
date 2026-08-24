"""Implementation of the Fractal Memory Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiScaleAttention


class FractalMemoryNetwork(nn.Module):
    """Decoder-only language model with hierarchical recurrent memory levels."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._validate_config()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)

        self.memory_levels = nn.ModuleList(
            FractalMemoryLevel(config, level_idx)
            for level_idx in range(config.num_memory_levels)
        )
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        self.init_weights()

    def _validate_config(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if len(self.config.memory_update_rate) != self.config.num_memory_levels:
            raise ValueError("memory_update_rate must contain one value per memory level")
        if any(rate < 1 for rate in self.config.memory_update_rate):
            raise ValueError("memory_update_rate values must be >= 1")

    def init_weights(self):
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_layer.bias)

    def get_position_embeddings(self, position_ids):
        return self.position_embeddings(position_ids)

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        labels=None,
        memory_states=None,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.size(1) > self.config.max_seq_length:
            raise ValueError(
                f"sequence length {input_ids.size(1)} exceeds max_seq_length "
                f"{self.config.max_seq_length}"
            )

        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1),
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand_as(input_ids)

        if memory_states is None:
            memory_states = [None] * self.config.num_memory_levels
        elif len(memory_states) != self.config.num_memory_levels:
            raise ValueError("memory_states must contain one tensor per memory level")

        embeddings = self.word_embeddings(input_ids) + self.get_position_embeddings(position_ids)
        hidden_states = self.dropout(self.layer_norm(embeddings))

        new_memory_states = []
        for level_idx, memory_level in enumerate(self.memory_levels):
            hidden_states, new_memory = memory_level(
                hidden_states,
                attention_mask=attention_mask,
                memory_state=memory_states[level_idx],
            )
            new_memory_states.append(new_memory)

        logits = self.output_layer(hidden_states)
        outputs = (logits, new_memory_states)

        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            outputs = (loss,) + outputs

        return outputs


class FractalMemoryLevel(nn.Module):
    def __init__(self, config, level_idx):
        super().__init__()
        self.update_rate = config.memory_update_rate[level_idx]

        self.attention = MultiScaleAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.memory_projection = nn.Linear(config.hidden_size, config.hidden_size)

        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states, attention_mask=None, memory_state=None):
        residual = hidden_states
        attention_output, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            memory_states=memory_state,
        )
        hidden_states = self.layer_norm1(residual + self.dropout(attention_output))

        residual = hidden_states
        hidden_states = F.gelu(self.intermediate(hidden_states))
        hidden_states = self.dropout(self.output(hidden_states))
        hidden_states = self.layer_norm2(residual + hidden_states)

        candidate_memory = self.memory_projection(hidden_states)
        if memory_state is not None and memory_state.shape == candidate_memory.shape:
            alpha = 1.0 / self.update_rate
            new_memory = memory_state + alpha * (candidate_memory - memory_state)
        else:
            new_memory = candidate_memory

        return hidden_states, new_memory
