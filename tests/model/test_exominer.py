import pytest
import torch

from project.inspection import random_inputs
from project.model import (
    DIAGRAM,
    PAPER,
    REPO_V1,
    ConfigValue,
    ExoMinerConfig,
    ExoMinerPlusPlus,
    with_overrides,
)

PRESETS = [PAPER, DIAGRAM, REPO_V1]


@pytest.mark.parametrize("config", PRESETS)
@pytest.mark.parametrize("batch_size", [1, 8])
def test_forward_returns_one_logit_per_example(
        config: ExoMinerConfig,
        batch_size: int,
) -> None:
    model = ExoMinerPlusPlus(config)

    assert model(random_inputs(config, batch_size)).shape == (batch_size,)


@pytest.mark.parametrize("config", PRESETS)
def test_every_parameter_gets_a_gradient(config: ExoMinerConfig) -> None:
    model = ExoMinerPlusPlus(config)

    model(random_inputs(config, 4)).sum().backward()

    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing


@pytest.mark.parametrize("config", PRESETS)
def test_eval_mode_is_deterministic(config: ExoMinerConfig) -> None:
    model = ExoMinerPlusPlus(config).eval()
    inputs = random_inputs(config, 4)

    assert torch.equal(model(inputs), model(inputs))


def test_missing_feature_raises_key_error() -> None:
    model = ExoMinerPlusPlus(PAPER)
    inputs = random_inputs(PAPER, 2)
    del inputs["tce_steff_norm"]

    with pytest.raises(KeyError, match="tce_steff_norm"):
        _ = model(inputs)


@pytest.mark.parametrize(
    ("name", "shape"),
    [("tce_steff_norm", (2, 1)), ("global_flux_view_fluxnorm", (2, 301, 1))],
)
def test_wrong_feature_shape_raises_value_error(
        name: str,
        shape: tuple[int, ...],
) -> None:
    model = ExoMinerPlusPlus(PAPER)
    inputs = random_inputs(PAPER, 2)
    inputs[name] = torch.randn(shape)

    with pytest.raises(ValueError, match=name):
        _ = model(inputs)


def test_unbatched_scalar_raises_value_error() -> None:
    model = ExoMinerPlusPlus(PAPER)
    inputs = random_inputs(PAPER, 2)
    inputs["tce_steff_norm"] = torch.tensor(0.5)

    with pytest.raises(ValueError, match="tce_steff_norm"):
        _ = model(inputs)


def test_mismatched_batch_sizes_raise_value_error() -> None:
    model = ExoMinerPlusPlus(PAPER)
    inputs = random_inputs(PAPER, 2)
    inputs["tce_steff_norm"] = torch.randn(3)

    with pytest.raises(ValueError, match="tce_steff_norm"):
        _ = model(inputs)


def test_extra_features_are_ignored() -> None:
    model = ExoMinerPlusPlus(PAPER)
    inputs = random_inputs(PAPER, 2)
    inputs["unused_feature"] = torch.randn(2)

    assert model(inputs).shape == (2,)


def test_disabled_branch_is_not_built_and_shrinks_head() -> None:
    config = with_overrides(
        PAPER,
        {"momentum_dump.enabled": False, "transit_views.odd_even_enabled": False},
    )
    model = ExoMinerPlusPlus(config)
    first_layer = model.head[0]

    assert "momentum_dump" not in model.branches
    assert isinstance(first_layer, torch.nn.Linear)
    assert first_layer.in_features == 48 - 2 * 4
    assert model(random_inputs(config, 2)).shape == (2,)


@pytest.mark.parametrize("kept", ["flux", "weak_secondary", "odd_even"])
def test_single_transit_diagnostic(kept: str) -> None:
    diagnostics = ("flux", "weak_secondary", "odd_even")
    overrides: dict[str, ConfigValue] = {
        f"transit_views.{name}_enabled": name == kept for name in diagnostics
    }
    config = with_overrides(PAPER, overrides)
    model = ExoMinerPlusPlus(config)

    assert model(random_inputs(config, 2)).shape == (2,)


def test_single_hidden_layer_head() -> None:
    config = with_overrides(PAPER, {"head_layers": 1, "head_units": 16})
    model = ExoMinerPlusPlus(config)

    assert len(model.head) == 4
    assert model(random_inputs(config, 2)).shape == (2,)


def test_paper_parameter_count() -> None:
    model = ExoMinerPlusPlus(PAPER)

    assert sum(p.numel() for p in model.parameters()) == 1_130_217
