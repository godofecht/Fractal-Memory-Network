"""Data processing utilities for the Fractal Memory Network."""

from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int,
        return_tensors: str = "pt",
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
                return_tensors=self.return_tensors,
            )
            examples.append(
                {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                }
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def create_sliding_window_inputs(
    input_ids: torch.Tensor,
    window_size: int,
    stride: Optional[int] = None,
) -> torch.Tensor:
    """Create overlapping windows from a batch of token sequences."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, sequence)")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size > input_ids.size(1):
        raise ValueError("window_size cannot exceed sequence length")

    if stride is None:
        stride = max(1, window_size // 2)
    if stride <= 0:
        raise ValueError("stride must be positive")

    windows = []
    for start in range(0, input_ids.size(1) - window_size + 1, stride):
        windows.append(input_ids[:, start : start + window_size])
    return torch.stack(windows, dim=1)


def create_attention_mask(input_ids: torch.Tensor, padding_token_id: int) -> torch.Tensor:
    return (input_ids != padding_token_id).long()


def shift_tokens_right(
    input_ids: torch.Tensor,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> torch.Tensor:
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids
