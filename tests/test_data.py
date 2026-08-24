import pytest
import torch

from utils.data import create_attention_mask, create_sliding_window_inputs, shift_tokens_right


def test_sliding_windows():
    inputs = torch.arange(12).view(1, 12)
    windows = create_sliding_window_inputs(inputs, window_size=4, stride=2)
    assert windows.shape == (1, 5, 4)
    torch.testing.assert_close(windows[0, 1], torch.tensor([2, 3, 4, 5]))


def test_sliding_windows_validate_shape():
    with pytest.raises(ValueError):
        create_sliding_window_inputs(torch.arange(4), window_size=2)
    with pytest.raises(ValueError):
        create_sliding_window_inputs(torch.arange(4).view(1, 4), window_size=5)


def test_attention_mask_and_shift():
    inputs = torch.tensor([[4, 5, 0, 0]])
    mask = create_attention_mask(inputs, padding_token_id=0)
    shifted = shift_tokens_right(inputs, pad_token_id=0, decoder_start_token_id=9)

    torch.testing.assert_close(mask, torch.tensor([[1, 1, 0, 0]]))
    torch.testing.assert_close(shifted, torch.tensor([[9, 4, 5, 0]]))
