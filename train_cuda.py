"""CUDA-optimized training script for the Fractal Memory Network."""

import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import json
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
from transformers import GPT2Tokenizer
import psutil
import gc
import csv
from pathlib import Path

from model.fmn import FractalMemoryNetwork
from config import Config

class TextDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Flatten and process all data types
        for data_type in self.data.keys():
            for item in self.data[data_type]:
                # Process both input and target sequences
                if "input" in item and "target" in item:
                    self.samples.append({
                        "input": item["input"],
                        "target": item["target"]
                    })
                elif "sequence" in item:  # For long-term dependency type
                    self.samples.append({
                        "input": item["sequence"],
                        "target": item["sequence"]
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            sample["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            sample["target"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze()
        }

def print_gpu_utilization():
    """Print current GPU utilization."""
    print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")

class TrainingLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.csv"
        self.log_file = self.log_dir / f"training_{timestamp}.log"
        
        # Initialize CSV writer
        self.metrics_file.write_text("epoch,train_loss,val_loss,gpu_memory_used,gpu_memory_cached\n")
        
    def log_metrics(self, metrics: dict):
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get('epoch', ''),
                metrics.get('train_loss', ''),
                metrics.get('val_loss', ''),
                metrics.get('gpu_memory_used', ''),
                metrics.get('gpu_memory_cached', '')
            ])
    
    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)

def train():
    # Initialize logger
    logger = TrainingLogger()
    logger.log_message("Starting training...")
    
    # Set up CUDA device
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
    
    logger.log_message("\nInitial GPU state:")
    print_gpu_utilization()
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset = TextDataset("manual_data/train.json", tokenizer, Config.model.max_seq_length)
    val_dataset = TextDataset("manual_data/test.json", tokenizer, Config.model.max_seq_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = FractalMemoryNetwork(Config.model).to(device)
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        logger.log_message(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Initialize optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=Config.training.learning_rate,
        weight_decay=Config.training.weight_decay,
        eps=Config.training.adam_epsilon
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    patience = 0
    max_patience = Config.training.patience
    
    logger.log_message("\nStarting training:")
    print_gpu_utilization()
    
    for epoch in range(Config.training.num_epochs):
        model.train()
        total_train_loss = 0
        
        # Training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in train_pbar:
            # Move batch to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs[0]
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs[0]
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Log metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "gpu_memory_used": torch.cuda.memory_allocated() / 1024**2,
            "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2
        }
        logger.log_metrics(metrics)
        
        # Print epoch summary
        logger.log_message(f"\nEpoch {epoch+1}:")
        logger.log_message(f"Average Training Loss: {avg_train_loss:.4f}")
        logger.log_message(f"Average Validation Loss: {avg_val_loss:.4f}")
        print_gpu_utilization()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            
            # Save model
            save_dir = os.path.join("checkpoints", f"epoch_{epoch+1}_val_loss_{avg_val_loss:.4f}")
            os.makedirs(save_dir, exist_ok=True)
            
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(save_dir, "model.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
            
            logger.log_message(f"Saved best model to {save_dir}")
        else:
            patience += 1
        
        # Early stopping
        if patience >= max_patience:
            logger.log_message(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Clear GPU cache
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.log_message("\nTraining completed!")
    print_gpu_utilization()

if __name__ == "__main__":
    train() 