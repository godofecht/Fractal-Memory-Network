import torch

from model.fmn import FractalMemoryNetwork


class TinyConfig:
    vocab_size = 64
    max_seq_length = 32
    hidden_size = 32
    num_memory_levels = 3
    num_attention_heads = 4
    intermediate_size = 64
    dropout_prob = 0.0
    memory_update_rate = [1, 2, 4]
    use_relative_positions = True
    max_relative_position = 8


def test_forward_loss_and_memory_shapes():
    torch.manual_seed(0)
    model = FractalMemoryNetwork(TinyConfig)
    input_ids = torch.randint(0, TinyConfig.vocab_size, (2, 8))

    loss, logits, memory_states = model(input_ids, labels=input_ids)

    assert torch.isfinite(loss)
    assert logits.shape == (2, 8, TinyConfig.vocab_size)
    assert len(memory_states) == TinyConfig.num_memory_levels
    assert all(state.shape == (2, 8, TinyConfig.hidden_size) for state in memory_states)


def test_memory_can_be_reused_across_segments():
    torch.manual_seed(1)
    model = FractalMemoryNetwork(TinyConfig).eval()
    first = torch.randint(0, TinyConfig.vocab_size, (2, 5))
    second = torch.randint(0, TinyConfig.vocab_size, (2, 7))

    _, memory_states = model(first)
    logits, next_memory_states = model(second, memory_states=memory_states)

    assert logits.shape == (2, 7, TinyConfig.vocab_size)
    assert all(state.shape == (2, 7, TinyConfig.hidden_size) for state in next_memory_states)


def test_causal_attention_prevents_future_token_leakage():
    torch.manual_seed(2)
    model = FractalMemoryNetwork(TinyConfig).eval()

    left = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    right = torch.tensor([[1, 2, 3, 4, 21, 22, 23, 24]])

    left_logits, _ = model(left)
    right_logits, _ = model(right)

    torch.testing.assert_close(left_logits[:, :4], right_logits[:, :4])
