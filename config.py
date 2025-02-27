"""Configuration parameters for the Fractal Memory Network."""

class ModelConfig:
    # Model architecture
    vocab_size = 50257  # GPT-2 vocabulary size
    max_seq_length = 1024
    hidden_size = 768
    num_memory_levels = 4
    num_attention_heads = 12
    intermediate_size = 3072
    dropout_prob = 0.1
    
    # Memory specific parameters
    memory_update_rate = [1, 2, 4, 8]  # Update frequencies for each level
    memory_size = [hidden_size] * num_memory_levels
    use_relative_positions = True
    max_relative_position = 64

class TrainingConfig:
    # Training parameters
    batch_size = 16  # Reduced for GPU memory
    learning_rate = 5e-5
    num_epochs = 100
    warmup_steps = 1000
    max_steps = 100000
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0
    
    # Optimizer parameters
    weight_decay = 0.01
    adam_epsilon = 1e-8
    
    # CUDA specific
    cuda_device = "cuda"
    num_workers = 4
    pin_memory = True
    mixed_precision = True
    cudnn_benchmark = True
    
    # Logging and saving
    logging_steps = 10
    save_steps = 1000
    eval_steps = 1000
    
    # Early stopping
    patience = 5
    min_delta = 0.001
    
    # Dataset parameters
    train_file = "manual_data/train.json"
    val_file = "manual_data/test.json"
    
class Config:
    model = ModelConfig
    training = TrainingConfig 