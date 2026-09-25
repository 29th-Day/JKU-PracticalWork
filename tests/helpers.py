"""Random inputs for tests, built from the features a config reads."""

import torch
from torch import Tensor

from project.features import FEATURE_SHAPES, Batch
from project.model import ExoMinerConfig, required_features


def random_inputs(config: ExoMinerConfig, batch_size: int) -> dict[str, Tensor]:
    """Standard-normal tensors for every feature the model built from `config` reads."""
    return {
        name: torch.randn(batch_size, *FEATURE_SHAPES[name])
        for name in sorted(required_features(config))
    }


def random_batch(config: ExoMinerConfig, batch_size: int) -> Batch:
    """A `Batch` of random inputs with alternating labels 0, 1, 0, ..."""
    labels = torch.arange(batch_size) % 2
    return {"features": random_inputs(config, batch_size), "label": labels.float()}
