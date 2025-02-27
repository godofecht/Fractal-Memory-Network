"""Implementation of the Fractal Memory Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiScaleAttention

class FractalMemoryNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Memory levels
        self.memory_levels = nn.ModuleList([
            FractalMemoryLevel(config, level_idx)
            for level_idx in range(config.num_memory_levels)
        ])
        
        # Output head
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        self.word_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.normal_(mean=0.0, std=0.02)
    
    def get_position_embeddings(self, position_ids):
        return self.position_embeddings(position_ids)
    
    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        labels=None,
        memory_states=None
    ):
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1), dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand_as(input_ids)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.get_position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        
        # Apply layer norm and dropout
        hidden_states = self.layer_norm(embeddings)
        hidden_states = self.dropout(hidden_states)
        
        # Initialize or get memory states
        if memory_states is None:
            memory_states = [None] * self.config.num_memory_levels
        
        # Process through memory levels
        new_memory_states = []
        for level_idx, memory_level in enumerate(self.memory_levels):
            hidden_states, new_memory = memory_level(
                hidden_states,
                attention_mask,
                memory_states[level_idx]
            )
            new_memory_states.append(new_memory)
        
        # Get logits
        logits = self.output_layer(hidden_states)
        
        outputs = (logits, new_memory_states)
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs = (loss,) + outputs
        
        return outputs

class FractalMemoryLevel(nn.Module):
    def __init__(self, config, level_idx):
        super().__init__()
        self.level_idx = level_idx
        self.update_rate = config.memory_update_rate[level_idx]
        
        self.attention = MultiScaleAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Memory state
        self.memory_size = config.memory_size[level_idx]
        self.memory_key = nn.Linear(config.hidden_size, self.memory_size)
        self.memory_value = nn.Linear(config.hidden_size, self.memory_size)
    
    def forward(self, hidden_states, attention_mask=None, memory_state=None):
        residual = hidden_states
        
        # Self attention
        attention_output, _ = self.attention(
            hidden_states,
            attention_mask,
            memory_state
        )
        hidden_states = self.layer_norm1(residual + attention_output)
        
        # Feed forward
        residual = hidden_states
        hidden_states = self.intermediate(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.output(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm2(residual + hidden_states)
        
        # Update memory state
        if self.training:
            # Only update memory every update_rate steps
            if (self.level_idx + 1) % self.update_rate == 0:
                memory_key = self.memory_key(hidden_states)
                memory_value = self.memory_value(hidden_states)
                new_memory = torch.cat([memory_key, memory_value], dim=-1)
            else:
                new_memory = memory_state if memory_state is not None else None
        else:
            memory_key = self.memory_key(hidden_states)
            memory_value = self.memory_value(hidden_states)
            new_memory = torch.cat([memory_key, memory_value], dim=-1)
        
        return hidden_states, new_memory 