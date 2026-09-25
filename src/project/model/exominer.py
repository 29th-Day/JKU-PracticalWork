"""The ExoMiner++ network (Valizadegan et al. 2025)."""

__all__ = ["ExoMinerPlusPlus"]

from collections.abc import Mapping
from functools import partial
from typing import override

import torch
from torch import Tensor, nn

from project.features import (
    CENTROID_VIEWS,
    FEATURE_SHAPES,
    FLUX_TREND_VIEWS,
    FULL_ORBIT_BINS,
    GLOBAL_FLUX_VIEWS,
    MOMENTUM_DUMP_VIEWS,
    PERIODOGRAM_BINS,
    PERIODOGRAM_VIEWS,
    TRANSIT_BINS,
)
from project.model.branches import (
    Branch,
    DifferenceImageBranch,
    ScalarBranch,
    TimeSeriesBranch,
    TransitViewBranches,
    UnfoldedFluxBranch,
)
from project.model.config import ExoMinerConfig, required_features
from project.model.layers import init_weights, make_activation


class ExoMinerPlusPlus(nn.Module):
    """Branches per diagnostic, concatenated and classified by an FC head.

    `forward` takes normalized features keyed by name (see `project.features`) and
    returns one logit per example, shape `(B,)`. Missing features raise `KeyError`,
    features whose per-example shape differs from `FEATURE_SHAPES` raise `ValueError`.
    """

    def __init__(self, config: ExoMinerConfig) -> None:
        super().__init__()
        self.config: ExoMinerConfig = config
        self.required_features: frozenset[str] = required_features(config)
        branches = _build_branches(config)
        self.branches: nn.ModuleDict = nn.ModuleDict(branches)
        width = sum(branch.out_features for branch in branches.values())
        layers: list[nn.Module] = []
        for _ in range(config.head_layers):
            layers.append(nn.Linear(width, config.head_units))
            layers.append(make_activation(config.activation))
            layers.append(nn.Dropout(config.head_dropout))
            width = config.head_units
        layers.append(nn.Linear(width, 1))
        self.head: nn.Sequential = nn.Sequential(*layers)
        self.apply(partial(init_weights, conv_init=config.conv_init))

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        self._check_inputs(inputs)
        outputs = [branch(inputs) for branch in self.branches.values()]
        return self.head(torch.cat(outputs, dim=1)).squeeze(1)

    def _check_inputs(self, inputs: Mapping[str, Tensor]) -> None:
        missing = sorted(self.required_features - inputs.keys())
        if missing:
            raise KeyError(f"missing input features: {missing}")
        for name in sorted(self.required_features):
            shape = tuple(inputs[name].shape[1:])
            expected = FEATURE_SHAPES[name]
            if shape != expected:
                message = f"{name}: per-example shape {shape}, expected {expected}"
                raise ValueError(message)


def _build_branches(config: ExoMinerConfig) -> dict[str, Branch]:
    """Builds every enabled branch, keyed by its config field name."""
    units = config.branch_units
    activation = config.activation
    dropout = config.branch_dropout
    scalar_dropout = dropout if config.scalar_branch_dropout else 0.0
    time_series = [
        ("global_flux", config.global_flux, GLOBAL_FLUX_VIEWS, FULL_ORBIT_BINS),
        ("flux_trend", config.flux_trend, FLUX_TREND_VIEWS, FULL_ORBIT_BINS),
        (
            "flux_periodogram",
            config.flux_periodogram,
            PERIODOGRAM_VIEWS,
            PERIODOGRAM_BINS,
        ),
        ("local_centroid", config.local_centroid, CENTROID_VIEWS, TRANSIT_BINS),
        ("momentum_dump", config.momentum_dump, MOMENTUM_DUMP_VIEWS, TRANSIT_BINS),
    ]
    branches: dict[str, Branch] = {}
    for name, branch_config, views, bins in time_series:
        if branch_config.enabled:
            branches[name] = TimeSeriesBranch(
                branch_config, views, bins, units, activation, dropout
            )
    if config.transit_views.enabled:
        branches["transit_views"] = TransitViewBranches(
            config.transit_views, units, activation, dropout
        )
    if config.unfolded_flux.enabled:
        branches["unfolded_flux"] = UnfoldedFluxBranch(
            config.unfolded_flux, units, activation, dropout
        )
    if config.diff_img.enabled:
        branches["diff_img"] = DifferenceImageBranch(
            config.diff_img, units, activation, dropout
        )
    for name, scalar_config in (
        ("stellar", config.stellar),
        ("dv_tce_fit", config.dv_tce_fit),
    ):
        if scalar_config.enabled:
            branches[name] = ScalarBranch(
                scalar_config, units, activation, scalar_dropout
            )
    return branches
