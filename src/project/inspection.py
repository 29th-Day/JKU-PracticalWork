"""Per-operation activation statistics of an exported model, for debugging.

The statistics are keyed by the names of the exported graph's nodes, which are
also the node ids Model Explorer shows for a `torch.export` program.
"""

__all__ = ["STATISTICS", "activation_statistics", "random_inputs"]

import math
from collections.abc import Mapping
from typing import override

import torch
from torch import Tensor
from torch.export import ExportedProgram
from torch.fx import Interpreter, Node

from project.features import FEATURE_SHAPES
from project.model import ExoMinerConfig, required_features

STATISTICS = ("mean", "std", "max_abs", "zero_fraction", "non_finite")


def activation_statistics(
        program: ExportedProgram,
        inputs: Mapping[str, Tensor],
) -> dict[str, dict[str, float]]:
    """Runs `program` on `inputs` and summarizes every floating-point output.

    Returns, per graph node, the `STATISTICS`: mean, standard deviation and
    largest absolute value of the finite values, the fraction of exact zeros
    (e.g. dead ReLUs) and the number of NaN or infinite values.
    """
    recorder = _Recorder(program.module())
    with torch.no_grad():
        recorder.run(inputs)
    return recorder.statistics


def random_inputs(config: ExoMinerConfig, batch_size: int) -> dict[str, Tensor]:
    """Standard-normal tensors for every feature the model built from `config` reads."""
    return {
        name: torch.randn(batch_size, *FEATURE_SHAPES[name])
        for name in sorted(required_features(config))
    }


class _Recorder(Interpreter):
    """Interpreter that summarizes each operation's output while running."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__(module)
        self.statistics: dict[str, dict[str, float]] = {}

    @override
    def run_node(self, n: Node) -> object:
        output = super().run_node(n)
        is_float = isinstance(output, Tensor) and output.is_floating_point()
        if n.op == "call_function" and is_float:
            self.statistics[n.name] = _summarize(output)
        return output


def _summarize(tensor: Tensor) -> dict[str, float]:
    values = tensor.detach().float().flatten()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        mean = std = max_abs = math.nan
    else:
        mean = finite.mean().item()
        std = finite.std(correction=0).item()
        max_abs = finite.abs().max().item()
    return {
        "mean": mean,
        "std": std,
        "max_abs": max_abs,
        "zero_fraction": (values == 0).float().mean().item(),
        "non_finite": float(values.numel() - finite.numel()),
    }
