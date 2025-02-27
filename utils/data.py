"""Data processing utilities for the Fractal Memory Network."""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional

class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        return_tensors: str = "pt"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.examples = self.process_texts(texts)
    
    def process_texts(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        examples = []
        
        for text in texts:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors=self.return_tensors
            )
            
            examples.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
            })
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

def create_sliding_window_inputs(
    input_ids: torch.Tensor,
    window_size: int,
    stride: Optional[int] = None
) -> torch.Tensor:
    """Create sliding window inputs for long sequences.
    
    Args:
        input_ids: Input tensor of shape (batch_size, sequence_length)
        window_size: Size of each window
        stride: Number of tokens to slide the window by (default: window_size // 2)
    
    Returns:
        Tensor of shape (num_windows, window_size) containing the windowed sequences
    """
    if stride is None:
        stride = window_size // 2
    
    # Get shapes
    batch_size, seq_length = input_ids.shape
    num_windows = ((seq_length - window_size) // stride) + 1
    
    # Create windows
    windows = []
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        window = input_ids[:, start_idx:end_idx]
        windows.append(window)
    
    return torch.stack(windows, dim=1)

def create_attention_mask(
    input_ids: torch.Tensor,
    padding_token_id: int
) -> torch.Tensor:
    """Create attention mask for padded sequences.
    
    Args:
        input_ids: Input tensor of shape (batch_size, sequence_length)
        padding_token_id: ID of the padding token
    
    Returns:
        Attention mask tensor of shape (batch_size, sequence_length)
    """
    return (input_ids != padding_token_id).float()

def shift_tokens_right(
    input_ids: torch.Tensor,
    pad_token_id: int,
    decoder_start_token_id: int
) -> torch.Tensor:
    """Shift input ids one token to the right for decoder input.
    
    Args:
        input_ids: Input tensor of shape (batch_size, sequence_length)
        pad_token_id: ID of the padding token
        decoder_start_token_id: ID of the decoder start token
    
    Returns:
        Shifted input tensor
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    
    # Replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    
    return shifted_input_ids 