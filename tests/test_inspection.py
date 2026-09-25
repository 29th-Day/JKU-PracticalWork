import math
from collections.abc import Mapping
from typing import override

import pytest
import torch
from torch import Tensor, nn

from project.inspection import activation_statistics, random_inputs
from project.model import PAPER, ExoMinerPlusPlus


class _ReluTimesTwo(nn.Module):
    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        return torch.relu(inputs["x"]) * 2


def _statistics(x: Tensor) -> dict[str, dict[str, float]]:
    inputs = {"x": x}
    program = torch.export.export(_ReluTimesTwo(), (inputs,))
    return activation_statistics(program, inputs)


def test_statistics_of_a_relu() -> None:
    relu = _statistics(torch.tensor([-1.0, 0.0, 2.0, 3.0]))["relu"]

    assert relu["mean"] == pytest.approx(1.25)
    population_std = torch.tensor([0.0, 0.0, 2.0, 3.0]).std(correction=0).item()
    assert relu["std"] == pytest.approx(population_std)
    assert relu["max_abs"] == pytest.approx(3.0)
    assert relu["zero_fraction"] == pytest.approx(0.5)
    assert relu["non_finite"] == 0


def test_non_finite_values_are_counted_and_skipped() -> None:
    stats = _statistics(torch.tensor([1.0, math.nan, math.inf, 3.0]))["mul"]

    assert stats["non_finite"] == 2
    assert stats["mean"] == pytest.approx(4.0)
    assert stats["max_abs"] == pytest.approx(6.0)


def test_all_non_finite_output_gives_nan_statistics() -> None:
    stats = _statistics(torch.tensor([math.nan, math.nan]))["relu"]

    assert stats["non_finite"] == 2
    assert math.isnan(stats["mean"])
    assert math.isnan(stats["max_abs"])


def test_statistics_cover_every_exominer_operation() -> None:
    model = ExoMinerPlusPlus(PAPER).eval()
    inputs = random_inputs(PAPER, 4)
    program = torch.export.export(model, (inputs,))

    stats = activation_statistics(program, inputs)

    operations = {
        node.name for node in program.graph.nodes if node.op == "call_function"
    }
    assert set(stats) <= operations
    assert {"conv2d", "relu", "linear"} <= set(stats)
    assert len(stats) > 200
