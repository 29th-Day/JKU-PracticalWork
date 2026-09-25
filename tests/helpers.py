"""Random labeled batches for tests, built from the features a config reads."""

import torch

from project.features import Batch
from project.inspection import random_inputs
from project.model import ExoMinerConfig


def random_batch(config: ExoMinerConfig, batch_size: int) -> Batch:
    """A `Batch` of random inputs with alternating labels 0, 1, 0, ..."""
    labels = torch.arange(batch_size) % 2
    return {"features": random_inputs(config, batch_size), "label": labels.float()}
