"""Comprehensive benchmarking script for the Fractal Memory Network."""

import torch
import json
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
import evaluate
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from typing import Dict, List

from model.fmn import FractalMemoryNetwork
from config import Config
from utils.data import TextDataset

class ModelBenchmark:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FractalMemoryNetwork(Config.model).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval()
        
        # Load metrics
        self.perplexity = evaluate.load("perplexity")
        self.metrics = {}
    
    def evaluate_synthetic_data(self, data_path: str) -> Dict:
        """Evaluate model on synthetic test data."""
        with open(data_path, "r") as f:
            test_data = json.load(f)
        
        results = {}
        
        # Evaluate repeated patterns
        repeated_mse = []
        for sample in tqdm(test_data["repeated_patterns"], desc="Evaluating repeated patterns"):
            input_seq = torch.tensor(sample["input"]).unsqueeze(0).to(self.device)
            target_seq = torch.tensor(sample["target"]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_seq)[0]
                mse = mean_squared_error(target_seq.cpu().numpy().flatten(),
                                       output.argmax(-1).cpu().numpy().flatten())
                repeated_mse.append(mse)
        
        results["repeated_pattern_mse"] = np.mean(repeated_mse)
        
        # Evaluate hierarchical patterns
        hierarchical_mse = []
        for sample in tqdm(test_data["hierarchical_patterns"], desc="Evaluating hierarchical patterns"):
            input_seq = torch.tensor(sample["input"]).unsqueeze(0).to(self.device)
            target_seq = torch.tensor(sample["target"]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_seq)[0]
                mse = mean_squared_error(target_seq.cpu().numpy().flatten(),
                                       output.argmax(-1).cpu().numpy().flatten())
                hierarchical_mse.append(mse)
        
        results["hierarchical_pattern_mse"] = np.mean(hierarchical_mse)
        
        # Evaluate long-term dependencies
        longterm_acc = []
        for sample in tqdm(test_data["long_term_dependencies"], desc="Evaluating long-term dependencies"):
            sequence = torch.tensor(sample["sequence"]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(sequence)[0]
                pred = output.argmax(-1).cpu().numpy().flatten()
                # Check if the model correctly predicts the marker pairs
                markers = [int(x) for x in sample["description"].split("positions: ")[1].strip("[]").split(",")]
                correct = 0
                total = len(markers)
                for pos in markers:
                    if abs(pred[pos + 10] - (sequence[0][pos].item() + 1)) <= 5:  # Allow small error margin
                        correct += 1
                longterm_acc.append(correct / total)
        
        results["long_term_dependency_accuracy"] = np.mean(longterm_acc)
        
        return results
    
    def evaluate_wikitext(self) -> Dict:
        """Evaluate model on WikiText-103 test set."""
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        test_data = dataset["test"]
        
        # Prepare data
        encoded_data = self.tokenizer(
            test_data["text"],
            max_length=Config.model.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        test_dataset = TextDataset(
            texts=test_data["text"],
            tokenizer=self.tokenizer,
            max_length=Config.model.max_seq_length
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.training.batch_size,
            shuffle=False
        )
        
        # Calculate perplexity
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating on WikiText"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs[0]
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        
        return {"wikitext_perplexity": perplexity}
    
    def plot_results(self, results: Dict, output_dir: str = "benchmark_results"):
        """Plot benchmark results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot synthetic data results
        synthetic_metrics = [
            results["repeated_pattern_mse"],
            results["hierarchical_pattern_mse"],
            1 - results["long_term_dependency_accuracy"]  # Convert accuracy to error
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(
            ["Repeated", "Hierarchical", "Long-term"],
            synthetic_metrics
        )
        plt.title("Error Metrics on Synthetic Data")
        plt.ylabel("Error Rate")
        plt.savefig(os.path.join(output_dir, "synthetic_results.png"))
        plt.close()
        
        # Save numerical results
        with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
            json.dump(results, f, indent=2)

def main():
    # Initialize benchmark
    benchmark = ModelBenchmark()
    
    # Run synthetic data evaluation
    print("Evaluating on synthetic data...")
    synthetic_results = benchmark.evaluate_synthetic_data("data/test.json")
    
    # Run WikiText evaluation
    print("\nEvaluating on WikiText-103...")
    wikitext_results = benchmark.evaluate_wikitext()
    
    # Combine results
    all_results = {**synthetic_results, **wikitext_results}
    
    # Plot and save results
    benchmark.plot_results(all_results)
    
    # Print summary
    print("\nBenchmark Results:")
    print(f"Repeated Pattern MSE: {all_results['repeated_pattern_mse']:.4f}")
    print(f"Hierarchical Pattern MSE: {all_results['hierarchical_pattern_mse']:.4f}")
    print(f"Long-term Dependency Accuracy: {all_results['long_term_dependency_accuracy']:.4f}")
    print(f"WikiText Perplexity: {all_results['wikitext_perplexity']:.4f}")

if __name__ == "__main__":
    main() 