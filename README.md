# Fractal Memory Network Language Model

This repository implements a Fractal Memory Network (FMN) for language modeling tasks. FMNs are designed to handle long-range dependencies in sequential data through a hierarchical memory structure that operates at different temporal scales.

## Features

- Hierarchical memory structure with multiple time scales
- Self-attention mechanisms at each level
- Fractal connectivity pattern for efficient information flow
- Customizable number of memory levels and attention heads
- Support for both autoregressive and masked language modeling

## Project Structure

```
.
├── requirements.txt
├── config.py           # Configuration parameters
├── model/
│   ├── __init__.py
│   ├── fmn.py         # Core FMN implementation
│   └── attention.py   # Custom attention mechanisms
├── train.py           # Training script
└── utils/
    ├── __init__.py
    └── data.py        # Data loading and preprocessing
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Configure your model parameters in `config.py`
2. Run training:
   ```bash
   python train.py
   ```

## Architecture

The Fractal Memory Network consists of multiple memory levels, each operating at different temporal scales. The key components are:

1. Input Embedding Layer
2. Multiple Memory Levels with:
   - Self-attention mechanisms
   - Fractal update rules
   - Skip connections
3. Output Layer

The fractal structure allows the model to capture both short-term and long-term dependencies efficiently. 