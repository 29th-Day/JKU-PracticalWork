"""Visualizes, inspects and debugs ExoMiner++ in Google's Model Explorer.

Each requested model is exported with `torch.export` and loaded into one Model
Explorer session; every branch is a collapsible layer group. One forward pass on
a random batch adds per-operation activation statistics as node data overlays
(mean, std, max_abs, zero_fraction for dead ReLUs, non_finite for NaN/inf).

Examples:
    uv run scripts/explore_model.py
    uv run scripts/explore_model.py --preset paper --preset repo_v1
    uv run scripts/explore_model.py --override global_flux.pool_size=4
    uv run scripts/explore_model.py --checkpoint run/model.ckpt
"""

__all__ = ["main"]

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import model_explorer  # pyright: ignore[reportMissingTypeStubs]
import torch
from model_explorer.node_data_builder import (  # pyright: ignore[reportMissingTypeStubs]
    GradientItem,
    GraphNodeData,
    NodeDataResult,
)

from project.classifier import ExoMinerClassifier
from project.inspection import STATISTICS, activation_statistics, random_inputs
from project.model import (
    DIAGRAM,
    PAPER,
    REPO_V1,
    ConfigValue,
    ExoMinerPlusPlus,
    with_overrides,
)

PRESETS = {"paper": PAPER, "diagram": DIAGRAM, "repo_v1": REPO_V1}
# Low values white, high values red.
GRADIENT = [
    GradientItem(stop=0, bgColor="#ffffff"),
    GradientItem(stop=1, bgColor="#d62728"),
]


@dataclass(frozen=True)
class _Arguments:
    presets: list[str]
    overrides: dict[str, ConfigValue]
    checkpoint: Path | None
    batch_size: int
    seed: int
    port: int
    no_browser: bool


def main() -> None:
    arguments = _parse_arguments()
    torch.manual_seed(arguments.seed)
    explorer_config = model_explorer.config()
    for name, model in _models(arguments):
        inputs = random_inputs(model.config, arguments.batch_size)
        program = torch.export.export(model, (inputs,))
        explorer_config.add_model_from_pytorch(name, exported_program=program)
        statistics = activation_statistics(program, inputs)
        for statistic in STATISTICS:
            node_data = _node_data(statistics, statistic)
            explorer_config.add_node_data(
                f"{name}: {statistic}", node_data, model_name=name
            )
    model_explorer.visualize_from_config(
        explorer_config,
        port=arguments.port,
        no_open_in_browser=arguments.no_browser,
    )


def _models(arguments: _Arguments) -> list[tuple[str, ExoMinerPlusPlus]]:
    """Builds the models to show, named as they appear in Model Explorer."""
    if arguments.checkpoint is not None:
        classifier = ExoMinerClassifier.load_from_checkpoint(
            arguments.checkpoint, map_location="cpu"
        )
        return [(arguments.checkpoint.stem, classifier.model.eval())]
    suffix = f" ({len(arguments.overrides)} overrides)" if arguments.overrides else ""
    models: list[tuple[str, ExoMinerPlusPlus]] = []
    for preset in arguments.presets:
        config = with_overrides(PRESETS[preset], arguments.overrides)
        models.append((f"{preset}{suffix}", ExoMinerPlusPlus(config).eval()))
    return models


def _node_data(
        statistics: dict[str, dict[str, float]],
        statistic: str,
) -> GraphNodeData:
    """One statistic of every node; NaNs are left out because JSON has none."""
    results = {
        node: NodeDataResult(value=values[statistic])
        for node, values in statistics.items()
        if not math.isnan(values[statistic])
    }
    return GraphNodeData(results=results, gradient=GRADIENT)


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser(
        description="Visualize ExoMiner++ in Google's Model Explorer."
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESETS),
        help="preset to show; repeat to compare several (default: paper)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="flat config override applied to every preset, e.g. head_units=256",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Lightning checkpoint to show instead of fresh presets",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="only start the server, e.g. over SSH; open the printed URL yourself",
    )
    namespace = parser.parse_args()
    presets: list[str] | None = namespace.preset
    overrides: list[str] = namespace.override
    checkpoint: Path | None = namespace.checkpoint
    if checkpoint is not None and (presets or overrides):
        parser.error("--checkpoint cannot be combined with --preset or --override")
    return _Arguments(
        presets=presets or ["paper"],
        overrides=dict(_parse_override(parser, text) for text in overrides),
        checkpoint=checkpoint,
        batch_size=namespace.batch_size,
        seed=namespace.seed,
        port=namespace.port,
        no_browser=namespace.no_browser,
    )


def _parse_override(
        parser: argparse.ArgumentParser,
        text: str,
) -> tuple[str, ConfigValue]:
    """Parses `KEY=VALUE`; VALUE is read as JSON when possible, else as text."""
    key, separator, raw = text.partition("=")
    if not separator:
        parser.error(f"override {text!r} is not of the form KEY=VALUE")
    try:
        value: ConfigValue = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


if __name__ == "__main__":
    main()
