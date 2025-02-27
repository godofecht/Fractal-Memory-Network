"""Kaggle-optimized training script for Fractal Memory Network."""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from transformers import GPT2Tokenizer
import gc
import kaggle

from model.fmn import FractalMemoryNetwork
from config import Config
from utils.data import TextDataset

class KaggleLogger:
    """Kaggle-specific logging with metadata and artifact management."""
    
    def __init__(self, experiment_name: str):
        self.base_path = Path("/kaggle/working")
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.output_dir = self.base_path / f"{experiment_name}_{self.timestamp}"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.plot_dir = self.output_dir / "plots"
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir, self.plot_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
        # Create log files
        self.metrics_file = self.log_dir / "metrics.json"
        self.metrics_file.write_text("{}")
        
        # Initialize plotting
        plt.style.use('seaborn')
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
        
    def log_metrics(self, metrics: dict, epoch: int):
        """Log metrics and update visualizations."""
        self.epochs.append(epoch)
        self.train_losses.append(metrics['train_loss'])
        self.val_losses.append(metrics['val_loss'])
        
        # Save metrics to file
        if self.metrics_file.exists():
            current_metrics = json.loads(self.metrics_file.read_text())
        else:
            current_metrics = {}
        
        current_metrics[f"epoch_{epoch}"] = metrics
        self.metrics_file.write_text(json.dumps(current_metrics, indent=2))
        
        # Update plots
        self._update_plots()
        
    def _update_plots(self):
        """Update and save training visualization plots."""
        # Clear previous plots
        for ax in self.axes:
            ax.clear()
        
        # Loss plot
        self.axes[0].plot(self.epochs, self.train_losses, label='Train Loss')
        self.axes[0].plot(self.epochs, self.val_losses, label='Validation Loss')
        self.axes[0].set_title('Training and Validation Loss')
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].legend()
        self.axes[0].grid(True)
        
        # Loss ratio plot
        if len(self.train_losses) > 1:
            loss_ratios = [val/train for train, val in zip(self.train_losses, self.val_losses)]
            self.axes[1].plot(self.epochs, loss_ratios, label='Val/Train Ratio')
            self.axes[1].set_title('Validation/Training Loss Ratio')
            self.axes[1].set_xlabel('Epoch')
            self.axes[1].set_ylabel('Ratio')
            self.axes[1].legend()
            self.axes[1].grid(True)
        
        plt.tight_layout()
        self.fig.savefig(self.plot_dir / 'training_progress.png')
        
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                       epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint with Kaggle metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
        # Log checkpoint metadata
        metadata = {
            'epoch': epoch,
            'loss': loss,
            'path': str(checkpoint_path),
            'is_best': is_best
        }
        
        with open(self.log_dir / 'checkpoints.json', 'a') as f:
            f.write(json.dumps(metadata) + '\n')

def train():
    """Main training function optimized for Kaggle environment."""
    
    # Initialize Kaggle-specific logger
    logger = KaggleLogger("FMN_experiment")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Generate data if not exists
    if not (Path("/kaggle/working/manual_data").exists()):
        print("Generating training data...")
        from create_manual_data import generate_manual_dataset
        generate_manual_dataset()
    
    # Load datasets
    train_dataset = TextDataset("manual_data/train.json", tokenizer, Config.model.max_seq_length)
    val_dataset = TextDataset("manual_data/test.json", tokenizer, Config.model.max_seq_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.training.batch_size,
        shuffle=True,
        num_workers=2,  # Kaggle-optimized
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = FractalMemoryNetwork(Config.model).to(device)
    
    # Initialize optimizer and scaler
    optimizer = AdamW(
        model.parameters(),
        lr=Config.training.learning_rate,
        weight_decay=Config.training.weight_decay,
        eps=Config.training.adam_epsilon
    )
    
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience = 0
    
    print("Starting training...")
    for epoch in range(Config.training.num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
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
        
        # Validation phase
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
        
        # Log metrics and save checkpoint
        metrics = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'gpu_memory': torch.cuda.memory_allocated() / 1024**2
        }
        logger.log_metrics(metrics, epoch + 1)
        
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience = 0
        else:
            patience += 1
        
        logger.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=avg_val_loss,
            is_best=is_best
        )
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Average Training Loss: {avg_val_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if patience >= Config.training.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Clear GPU cache
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\nTraining completed!")
    return logger.output_dir

if __name__ == "__main__":
    output_dir = train()
    print(f"\nAll outputs saved to: {output_dir}") 