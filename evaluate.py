"""Evaluation and benchmarking script for the Fractal Memory Network."""

import torch
import numpy as np
from datasets import load_dataset, load_metric
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import json
import os

from model.fmn import FractalMemoryNetwork
from config import Config
from utils.data import TextDataset, create_attention_mask

def create_synthetic_data(num_samples=1000, max_length=128):
    """Create synthetic data for testing the model."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Generate random sequences of tokens
    data = []
    for _ in range(num_samples):
        # Create a sequence with some pattern (e.g., repeated subsequences)
        length = np.random.randint(32, max_length)
        # Create patterns that test memory capabilities
        pattern_length = np.random.randint(4, 16)
        pattern = np.random.randint(0, 1000, pattern_length)
        # Repeat the pattern with some noise
        num_repeats = length // pattern_length
        sequence = np.tile(pattern, num_repeats)
        # Add some noise
        noise = np.random.randint(0, 100, sequence.shape) * (np.random.random(sequence.shape) < 0.1)
        sequence = sequence + noise
        sequence = sequence[:length]
        
        # Convert to tokens
        tokens = ' '.join(map(str, sequence))
        data.append(tokens)
    
    return data

def evaluate_model(model, eval_dataloader, device):
    """Evaluate the model on the given dataloader."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    perplexity_metric = evaluate.load("perplexity")
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs[0]
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity
    }

def main():
    # Create synthetic datasets
    print("Generating synthetic data...")
    train_texts = create_synthetic_data(num_samples=1000)
    eval_texts = create_synthetic_data(num_samples=200)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create datasets
    train_dataset = TextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=Config.model.max_seq_length
    )
    eval_dataset = TextDataset(
        texts=eval_texts,
        tokenizer=tokenizer,
        max_length=Config.model.max_seq_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.training.batch_size,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=Config.training.batch_size,
        shuffle=False
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FractalMemoryNetwork(Config.model)
    model.to(device)
    
    # Load latest checkpoint if exists
    checkpoints_dir = "checkpoints"
    if os.path.exists(checkpoints_dir):
        checkpoints = sorted([d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint-")])
        if checkpoints:
            latest_checkpoint = os.path.join(checkpoints_dir, checkpoints[-1])
            print(f"Loading checkpoint from {latest_checkpoint}")
            model.load_state_dict(torch.load(os.path.join(latest_checkpoint, "pytorch_model.bin")))
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, eval_dataloader, device)
    
    # Save results
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "metrics.json")
    
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nEvaluation Results:")
    print(f"Average Loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    
    # Additional benchmarking using HuggingFace's evaluate
    print("\nRunning additional benchmarks...")
    
    # Test on specific capabilities
    memory_tests = create_synthetic_data(
        num_samples=50,
        max_length=Config.model.max_seq_length
    )
    
    memory_dataset = TextDataset(
        texts=memory_tests,
        tokenizer=tokenizer,
        max_length=Config.model.max_seq_length
    )
    memory_dataloader = DataLoader(
        memory_dataset,
        batch_size=Config.training.batch_size,
        shuffle=False
    )
    
    memory_metrics = evaluate_model(model, memory_dataloader, device)
    
    print("\nLong-term Memory Test Results:")
    print(f"Memory Test Perplexity: {memory_metrics['perplexity']:.4f}")

if __name__ == "__main__":
    main() 