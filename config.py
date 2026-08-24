"""Configuration parameters for the Fractal Memory Network."""


class ModelConfig:
    vocab_size = 50257
    max_seq_length = 1024
    hidden_size = 768
    num_memory_levels = 4
    num_attention_heads = 12
    intermediate_size = 3072
    dropout_prob = 0.1

    # Higher values produce slower exponential memory updates.
    memory_update_rate = [1, 2, 4, 8]
    use_relative_positions = True
    max_relative_position = 64


class TrainingConfig:
    batch_size = 16
    learning_rate = 5e-5
    num_epochs = 100
    warmup_steps = 1000
    max_steps = 100000
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0

    weight_decay = 0.01
    adam_epsilon = 1e-8

    cuda_device = "cuda"
    num_workers = 4
    pin_memory = True
    mixed_precision = True
    cudnn_benchmark = True

    logging_steps = 10
    save_steps = 1000
    eval_steps = 1000

    patience = 5
    min_delta = 0.001

    train_file = "manual_data/train.json"
    val_file = "manual_data/test.json"
    tokenizer_name = "gpt2"


class Config:
    model = ModelConfig
    training = TrainingConfig
