"""Generate structured training data for the Fractal Memory Network."""

import numpy as np
import json
import os
from typing import List, Dict, Tuple

def generate_repeated_pattern(min_len: int = 4, max_len: int = 16, num_repeats: int = 4) -> Tuple[List[int], List[int]]:
    """Generate a sequence with repeated patterns."""
    pattern_length = np.random.randint(min_len, max_len)
    pattern = np.random.randint(0, 1000, pattern_length)
    sequence = np.tile(pattern, num_repeats)
    # Add noise to some positions
    noise_mask = np.random.random(sequence.shape) < 0.1
    noise = np.random.randint(0, 100, sequence.shape) * noise_mask
    noisy_sequence = sequence + noise
    return sequence.tolist(), noisy_sequence.tolist()

def generate_hierarchical_pattern(levels: int = 3, base_length: int = 4) -> Tuple[List[int], List[int]]:
    """Generate hierarchical patterns with different frequencies."""
    sequence = []
    pattern_lengths = [base_length * (2 ** i) for i in range(levels)]
    
    for length in pattern_lengths:
        pattern = np.random.randint(0, 1000, length)
        repeats = 128 // length  # Ensure total length is consistent
        sequence.extend(np.tile(pattern, repeats))
    
    sequence = sequence[:128]  # Truncate to fixed length
    # Add noise
    noise_mask = np.random.random(len(sequence)) < 0.1
    noise = np.random.randint(0, 100, len(sequence)) * noise_mask
    noisy_sequence = np.array(sequence) + noise
    return sequence, noisy_sequence.tolist()

def generate_long_term_dependency(length: int = 128) -> Tuple[List[int], str]:
    """Generate sequences with long-term dependencies."""
    sequence = np.random.randint(0, 1000, length)
    # Insert markers at the start and corresponding positions later
    num_markers = 3
    marker_positions = np.random.choice(length - 20, num_markers, replace=False)
    marker_values = np.random.randint(1000, 2000, num_markers)
    
    for i, (pos, val) in enumerate(zip(marker_positions, marker_values)):
        sequence[pos] = val
        # Place corresponding marker later in sequence
        sequence[pos + 10 + i] = val + 1  # Predictable relationship
    
    description = f"Sequence contains {num_markers} marker pairs with positions: {marker_positions}"
    return sequence.tolist(), description

def generate_dataset(num_samples: int = 1000, output_dir: str = "data") -> None:
    """Generate a complete dataset with different types of patterns."""
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = {
        "repeated_patterns": [],
        "hierarchical_patterns": [],
        "long_term_dependencies": []
    }
    
    # Generate repeated patterns
    for _ in range(num_samples // 3):
        clean, noisy = generate_repeated_pattern()
        dataset["repeated_patterns"].append({
            "input": noisy,
            "target": clean,
            "type": "repeated"
        })
    
    # Generate hierarchical patterns
    for _ in range(num_samples // 3):
        clean, noisy = generate_hierarchical_pattern()
        dataset["hierarchical_patterns"].append({
            "input": noisy,
            "target": clean,
            "type": "hierarchical"
        })
    
    # Generate long-term dependencies
    for _ in range(num_samples // 3):
        sequence, description = generate_long_term_dependency()
        dataset["long_term_dependencies"].append({
            "sequence": sequence,
            "description": description,
            "type": "long_term"
        })
    
    # Save training data
    train_path = os.path.join(output_dir, "train.json")
    with open(train_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    # Generate a smaller test set
    test_dataset = {
        "repeated_patterns": dataset["repeated_patterns"][:50],
        "hierarchical_patterns": dataset["hierarchical_patterns"][:50],
        "long_term_dependencies": dataset["long_term_dependencies"][:50]
    }
    
    # Save test data
    test_path = os.path.join(output_dir, "test.json")
    with open(test_path, "w") as f:
        json.dump(test_dataset, f, indent=2)
    
    print(f"Generated {num_samples} training samples and {150} test samples")
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    generate_dataset() 