"""Attention mechanisms for the Fractal Memory Network."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class MultiScaleAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
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
            self.max_relative_position = config.max_relative_position
            self.relative_positions_encoding = RelativePositionEncoding(
                self.max_relative_position,
                self.attention_head_size
            )
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None, memory_states=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        if memory_states is not None:
            key_layer = self.transpose_for_scores(self.key(memory_states))
            value_layer = self.transpose_for_scores(self.value(memory_states))
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        if self.use_relative_positions:
            relative_position_scores = self.relative_positions_encoding(query_layer, key_layer)
            attention_scores = attention_scores + relative_position_scores
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs

class RelativePositionEncoding(nn.Module):
    def __init__(self, max_relative_position, hidden_size):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.hidden_size = hidden_size
        
        self.rel_embeddings = nn.Parameter(torch.Tensor(2 * max_relative_position + 1, hidden_size))
        nn.init.xavier_uniform_(self.rel_embeddings)
    
    def forward(self, query_layer, key_layer):
        seq_length = query_layer.size(2)
        
        # Generate position indices
        range_vec = torch.arange(seq_length, device=query_layer.device)
        range_mat = range_vec.unsqueeze(0).expand(seq_length, -1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip and shift distances to be in the range [0, 2*max_relative_position]
        distance_mat_clipped = torch.clamp(
            distance_mat + self.max_relative_position,
            0,
            2 * self.max_relative_position
        )
        
        # Map distances to embeddings
        final_mat = F.embedding(distance_mat_clipped.long(), self.rel_embeddings)
        
        # Compute attention scores
        rel_logits = torch.einsum('bhld,lrd->bhlr', query_layer, final_mat)
        
        return rel_logits 