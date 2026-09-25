"""The ExoMiner++ model, its configuration and presets."""

from project.model.config import (
    ConfigValue,
    ExoMinerConfig,
    OptimizerConfig,
    from_flat,
    required_features,
    to_flat,
    with_overrides,
)
from project.model.exominer import ExoMinerPlusPlus
from project.model.presets import (
    DIAGRAM,
    PAPER,
    PAPER_OPTIMIZER,
    REPO_V1,
    REPO_V1_OPTIMIZER,
)

__all__ = [
    "DIAGRAM",
    "PAPER",
    "PAPER_OPTIMIZER",
    "REPO_V1",
    "REPO_V1_OPTIMIZER",
    "ConfigValue",
    "ExoMinerConfig",
    "ExoMinerPlusPlus",
    "OptimizerConfig",
    "from_flat",
    "required_features",
    "to_flat",
    "with_overrides",
]
