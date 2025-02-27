"""Training script for the Fractal Memory Network."""

import os
import torch
import logging
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from model.fmn import FractalMemoryNetwork
from config import Config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def train():
    # Initialize wandb
    wandb.init(project="fractal-memory-network")
    
    # Load dataset
    dataset = load_dataset(
        Config.training.dataset_name,
        Config.training.dataset_config_name
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FractalMemoryNetwork(Config.model)
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=Config.training.learning_rate,
        eps=Config.training.adam_epsilon,
        weight_decay=Config.training.weight_decay
    )
    
    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.training.warmup_steps,
        num_training_steps=Config.training.max_steps
    )
    
    # Training loop
    global_step = 0
    model.train()
    
    while global_step < Config.training.max_steps:
        epoch_iterator = tqdm(
            DataLoader(
                dataset["train"],
                batch_size=Config.training.batch_size,
                shuffle=True
            ),
            desc="Training"
        )
        
        for step, batch in enumerate(epoch_iterator):
            # Prepare input
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) if "labels" in batch else input_ids
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs[0]
            
            # Backward pass
            loss.backward()
            
            if (step + 1) % Config.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    Config.training.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log metrics
                if global_step % Config.training.logging_steps == 0:
                    wandb.log({
                        "loss": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "step": global_step,
                    })
                
                # Save model
                if global_step % Config.training.save_steps == 0:
                    output_dir = os.path.join("checkpoints", f"checkpoint-{global_step}")
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    logger.info(f"Saved model checkpoint to {output_dir}")
                
                if global_step >= Config.training.max_steps:
                    break
            
            if global_step >= Config.training.max_steps:
                break
        
        if global_step >= Config.training.max_steps:
            break
    
    wandb.finish()

if __name__ == "__main__":
    train() 