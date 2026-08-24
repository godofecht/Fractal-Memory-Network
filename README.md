# Fractal Memory Network

Fractal Memory Network (FMN) is a research implementation of a decoder-only language model with multiple recurrent memory levels operating at different time scales. Each level combines causal self-attention with a persistent memory prefix, then updates that memory with a different exponential rate.

The core model is independently installable and testable. Training and benchmark dependencies are optional rather than being required to import the model.

## Architecture

For a token segment, FMN first builds token and position embeddings. The representation then passes through a stack of memory levels. At each level, the current segment attends causally to both the previous memory state and the visible prefix of the current segment. The resulting hidden state is projected into a new memory candidate. Fast levels replace their state aggressively; slower levels blend toward the candidate over a longer time scale.

Memory tensors always use the model hidden width, so they can be passed directly from one segment to the next. The returned `memory_states` list contains one tensor per level with shape `(batch, sequence, hidden_size)`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

For the training and benchmark scripts:

```bash
python -m pip install -e ".[train,benchmark]"
```

## Minimal usage

```python
import torch

from config import Config
from model import FractalMemoryNetwork

model = FractalMemoryNetwork(Config.model).eval()
input_ids = torch.randint(0, Config.model.vocab_size, (1, 32))

logits, memory = model(input_ids)
next_ids = torch.randint(0, Config.model.vocab_size, (1, 32))
next_logits, next_memory = model(next_ids, memory_states=memory)
```

For language-model training, pass `labels=input_ids`; the model shifts logits and labels internally and uses causal attention so future tokens are not visible to earlier positions.

## Configuration

The default architecture lives in `config.py`. `memory_update_rate = [1, 2, 4, 8]` controls the hierarchy: a rate of `1` fully adopts the newest memory candidate, while larger values update progressively more slowly.

## Training

The default training path consumes `manual_data/train.json` and uses the configured GPT-2 tokenizer. W&B logging is optional.

```bash
python -m pip install -e ".[train]"
python train.py
python train.py --wandb
```

Checkpoints are written as standard PyTorch state dictionaries under `checkpoints/`.

## CI

GitHub Actions installs the core package on Python 3.11, compiles the importable modules, and runs the unit suite. Tests cover loss/logit shapes, hierarchical memory reuse, causal masking, and sequence utility behavior. The workflow is intentionally CPU-only so routine validation stays lightweight.

## Repository status

FMN is a research prototype, not a pretrained model release. The checked-in manual datasets are useful for controlled experiments, while benchmark and training scripts may require network access for external tokenizers or datasets. The core model itself has no Hugging Face runtime dependency.
