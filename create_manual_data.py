"""Manual text data creation for testing specific capabilities of the Fractal Memory Network."""

import json
import os
from typing import List, Dict, Any

def create_repeating_text() -> Dict[str, Any]:
    """Create text with repeating patterns."""
    base_pattern = "the cat and the dog"
    sequence = (base_pattern + " ") * 5
    
    # Add variations to test memory
    noisy_sequence = sequence.replace("the cat and the dog the cat", "the cat but the dog the cat")
    
    return {
        "input": noisy_sequence,
        "target": sequence,
        "type": "repeated",
        "description": "Simple repeating phrase with a variation"
    }

def create_nested_text() -> Dict[str, Any]:
    """Create text with nested clauses and dependencies."""
    text = """
    The scientist, who had studied the ancient manuscripts that were discovered 
    in the temple which was built by the civilization that vanished centuries ago, 
    finally understood the message that was hidden within the text.
    """
    
    # Version with some clauses reordered
    noisy_text = """
    The scientist, who had studied the ancient manuscripts that were discovered
    in the temple that vanished centuries ago which was built by the civilization,
    finally understood the message that was hidden within the text.
    """
    
    return {
        "input": noisy_text.strip(),
        "target": text.strip(),
        "type": "hierarchical",
        "description": "Nested clauses with long-range grammatical dependencies"
    }

def create_long_term_text() -> Dict[str, Any]:
    """Create text with long-term dependencies between concepts."""
    text = """
    Alice had a red book and a blue pen. Bob preferred green colors.
    Later that day, Alice used her blue pen to write notes in her red book.
    Meanwhile, Bob was painting with his favorite green watercolors.
    """
    
    # Version with inconsistent references
    noisy_text = """
    Alice had a red book and a blue pen. Bob preferred green colors.
    Later that day, Alice used her green pen to write notes in her blue book.
    Meanwhile, Bob was painting with his favorite red watercolors.
    """
    
    return {
        "input": noisy_text.strip(),
        "target": text.strip(),
        "type": "long_term",
        "description": "Long-term consistency in object properties and references"
    }

def create_mathematical_text() -> Dict[str, Any]:
    """Create text with mathematical relationships expressed in words."""
    text = """
    On day one, there was one bird. On day two, there were two birds.
    On day three, there were four birds. On day four, there were eight birds.
    On day five, there were sixteen birds. Each day, the number of birds doubled.
    """
    
    # Version with some incorrect progressions
    noisy_text = """
    On day one, there was one bird. On day two, there were two birds.
    On day three, there were four birds. On day four, there were seven birds.
    On day five, there were twelve birds. Each day, the number of birds doubled.
    """
    
    return {
        "input": noisy_text.strip(),
        "target": text.strip(),
        "type": "mathematical",
        "description": "Geometric progression described in text"
    }

def create_contextual_text() -> Dict[str, Any]:
    """Create text where meaning depends on context."""
    text = """
    In summer: The trees were full of green leaves, and people wore light clothes.
    SEASON_CHANGE
    In winter: The trees were bare, and people wore heavy coats.
    SEASON_CHANGE
    Back in summer: The trees were full of green leaves again, and light clothes returned.
    """
    
    # Version with context violations
    noisy_text = """
    In summer: The trees were full of green leaves, and people wore light clothes.
    SEASON_CHANGE
    In winter: The trees were full of flowers, and people wore swimsuits.
    SEASON_CHANGE
    Back in summer: The trees were covered in snow, and heavy coats were common.
    """
    
    return {
        "input": noisy_text.strip(),
        "target": text.strip(),
        "type": "contextual",
        "description": "Context-dependent descriptions with seasonal changes"
    }

def create_logical_sequence() -> Dict[str, Any]:
    """Create text with logical cause and effect relationships."""
    text = """
    The sky darkened with heavy clouds. Soon after, rain began to fall.
    People opened their umbrellas and sought shelter. The streets became wet.
    Later, the sun emerged, and the puddles began to dry.
    """
    
    # Version with logical inconsistencies
    noisy_text = """
    The sky darkened with heavy clouds. Soon after, the sun was blazing.
    People opened their umbrellas and went sunbathing. The streets became wet.
    Later, it started raining, and the puddles began to appear.
    """
    
    return {
        "input": noisy_text.strip(),
        "target": text.strip(),
        "type": "logical",
        "description": "Logical sequence of weather-related events"
    }

def generate_manual_dataset() -> None:
    """Generate and save the manual text dataset."""
    dataset = {
        "training": {
            "repeated": [],
            "hierarchical": [],
            "long_term": [],
            "mathematical": [],
            "contextual": [],
            "logical": []
        },
        "testing": {
            "repeated": [],
            "hierarchical": [],
            "long_term": [],
            "mathematical": [],
            "contextual": [],
            "logical": []
        }
    }
    
    # Generate variations for training data
    for _ in range(10):
        dataset["training"]["repeated"].append(create_repeating_text())
        dataset["training"]["hierarchical"].append(create_nested_text())
        dataset["training"]["long_term"].append(create_long_term_text())
        dataset["training"]["mathematical"].append(create_mathematical_text())
        dataset["training"]["contextual"].append(create_contextual_text())
        dataset["training"]["logical"].append(create_logical_sequence())
    
    # Generate variations for testing data
    for _ in range(5):
        dataset["testing"]["repeated"].append(create_repeating_text())
        dataset["testing"]["hierarchical"].append(create_nested_text())
        dataset["testing"]["long_term"].append(create_long_term_text())
        dataset["testing"]["mathematical"].append(create_mathematical_text())
        dataset["testing"]["contextual"].append(create_contextual_text())
        dataset["testing"]["logical"].append(create_logical_sequence())
    
    # Save the datasets
    os.makedirs("manual_data", exist_ok=True)
    
    # Save training data
    with open("manual_data/train.json", "w") as f:
        json.dump(dataset["training"], f, indent=2)
    
    # Save testing data
    with open("manual_data/test.json", "w") as f:
        json.dump(dataset["testing"], f, indent=2)
    
    # Create a metadata file with descriptions
    metadata = {
        "dataset_description": "Manually crafted text datasets for testing FMN capabilities",
        "pattern_types": {
            "repeated": "Simple repeating phrases with variations",
            "hierarchical": "Nested clauses and complex grammatical structures",
            "long_term": "Long-range dependencies between concepts and properties",
            "mathematical": "Number sequences and patterns described in text",
            "contextual": "Context-dependent descriptions and state changes",
            "logical": "Cause and effect relationships in sequences"
        },
        "statistics": {
            "training_samples_per_type": 10,
            "testing_samples_per_type": 5,
            "total_training_samples": 60,  # 6 types * 10 samples
            "total_testing_samples": 30    # 6 types * 5 samples
        }
    }
    
    with open("manual_data/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Manual text dataset generation complete!")
    print(f"Created {metadata['statistics']['total_training_samples']} training samples")
    print(f"Created {metadata['statistics']['total_testing_samples']} test samples")
    print("Data saved in manual_data/")

if __name__ == "__main__":
    generate_manual_dataset()
