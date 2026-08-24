"""Train the Fractal Memory Network on the checked-in manual dataset."""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import Config
from model import FractalMemoryNetwork
from utils.data import TextDataset

logger = logging.getLogger(__name__)


def load_manual_texts(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        grouped_samples = json.load(handle)

    texts = []
    for samples in grouped_samples.values():
        for sample in samples:
            source = sample.get("input", "").strip()
            target = sample.get("target", "").strip()
            text = "\n".join(part for part in (source, target) if part)
            if text:
                texts.append(text)
    if not texts:
        raise ValueError(f"no training samples found in {path}")
    return texts


def save_checkpoint(model, optimizer, scheduler, step: int, output_dir: str):
    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        checkpoint_dir / "checkpoint.pt",
    )
    logger.info("Saved checkpoint to %s", checkpoint_dir)


def train(output_dir: str = "checkpoints", use_wandb: bool = False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    wandb = None
    if use_wandb:
        import wandb as wandb_module

        wandb = wandb_module
        wandb.init(project="fractal-memory-network")

    tokenizer = AutoTokenizer.from_pretrained(Config.training.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_manual_texts(Config.training.train_file)
    dataset = TextDataset(texts, tokenizer, Config.model.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.training.batch_size,
        shuffle=True,
        num_workers=Config.training.num_workers,
        pin_memory=Config.training.pin_memory and torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FractalMemoryNetwork(Config.model).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=Config.training.learning_rate,
        eps=Config.training.adam_epsilon,
        weight_decay=Config.training.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.training.warmup_steps,
        num_training_steps=Config.training.max_steps,
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for _ in range(Config.training.num_epochs):
        for step, batch in enumerate(tqdm(dataloader, desc="Training")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.masked_fill(attention_mask == 0, -100)

            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
            loss = loss / Config.training.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % Config.training.gradient_accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.training.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % Config.training.logging_steps == 0:
                scalar_loss = loss.item() * Config.training.gradient_accumulation_steps
                logger.info("step=%d loss=%.6f", global_step, scalar_loss)
                if wandb is not None:
                    wandb.log(
                        {
                            "loss": scalar_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "step": global_step,
                        }
                    )

            if global_step % Config.training.save_steps == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, output_dir)

            if global_step >= Config.training.max_steps:
                break

        if global_step >= Config.training.max_steps:
            break

    if global_step and global_step % Config.training.save_steps:
        save_checkpoint(model, optimizer, scheduler, global_step, output_dir)
    if wandb is not None:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args.output_dir, args.wandb)


if __name__ == "__main__":
    main()
