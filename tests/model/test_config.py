import json
from dataclasses import replace

import pytest

from project.model import (
    DIAGRAM,
    PAPER,
    PAPER_OPTIMIZER,
    REPO_V1,
    ExoMinerConfig,
    required_features,
    to_flat,
    with_overrides,
)

PRESETS = [PAPER, DIAGRAM, REPO_V1]


@pytest.mark.parametrize("config", PRESETS)
def test_flat_round_trip_reproduces_preset(config: ExoMinerConfig) -> None:
    assert with_overrides(PAPER, to_flat(config)) == config


@pytest.mark.parametrize("config", PRESETS)
def test_flat_round_trip_through_json(config: ExoMinerConfig) -> None:
    flat = json.loads(json.dumps(to_flat(config)))

    assert with_overrides(PAPER, flat) == config


def test_to_flat_uses_dotted_keys() -> None:
    flat = to_flat(PAPER)

    assert flat["activation"] == "relu"
    assert flat["global_flux.pool_size"] == 8
    assert flat["transit_views.flux_scalars"] == ("flux_local_stat_abs_min",)


def test_with_overrides_changes_only_given_keys() -> None:
    config = with_overrides(PAPER, {"global_flux.blocks": 3, "branch_dropout": 0.2})

    assert config.global_flux.blocks == 3
    assert config.branch_dropout == 0.2
    assert config.flux_trend == PAPER.flux_trend


def test_with_overrides_accepts_int_for_float() -> None:
    config = with_overrides(PAPER, {"branch_dropout": 0})

    assert config.branch_dropout == 0.0
    assert isinstance(config.branch_dropout, float)


def test_with_overrides_rejects_unknown_key() -> None:
    with pytest.raises(KeyError, match="global_flux.poolsize"):
        _ = with_overrides(PAPER, {"global_flux.poolsize": 4})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("global_flux.blocks", 2.5),
        ("global_flux.enabled", 1),
        ("head_layers", True),
        ("activation", 1),
        ("stellar.scalars", "tce_steff_norm"),
    ],
)
def test_with_overrides_rejects_wrong_type(key: str, value: float | str) -> None:
    with pytest.raises(TypeError, match=key):
        _ = with_overrides(PAPER, {key: value})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"transit_views.blocks": 5, "transit_views.pool_size": 16}, "transit_views"),
        ({"diff_img.pool_size": 5}, "diff_img"),
        ({"head_units": 0}, "head_units"),
        ({"branch_dropout": 1.0}, "branch_dropout"),
        ({"activation": "gelu"}, "activation"),
        ({"stellar.scalars": ("not_a_feature",)}, "not_a_feature"),
        ({"stellar.scalars": ()}, "stellar"),
    ],
)
def test_invalid_config_raises_value_error(
        overrides: dict[str, int | float | str | tuple[str, ...]],
        message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = with_overrides(PAPER, overrides)


def test_disabled_branch_skips_geometry_check() -> None:
    config = with_overrides(
        PAPER,
        {"diff_img.enabled": False, "diff_img.pool_size": 5},
    )

    assert not config.diff_img.enabled


def test_all_branches_disabled_raises() -> None:
    flat = to_flat(PAPER)
    disabled = {key: False for key in flat if key.endswith("enabled")}

    with pytest.raises(ValueError, match="at least one branch"):
        _ = with_overrides(PAPER, disabled)


def test_presets_differ_only_in_documented_fields() -> None:
    paper = to_flat(PAPER)
    diagram = {key for key, value in to_flat(DIAGRAM).items() if paper[key] != value}
    repo = {key for key, value in to_flat(REPO_V1).items() if paper[key] != value}

    assert diagram == {
        "activation",
        "branch_units",
        "scalar_branch_dropout",
        "flux_trend.pool_size",
        "flux_periodogram.pool_size",
        "unfolded_flux.stat_activation",
    }
    assert repo == {
        "activation",
        "conv_init",
        "branch_units",
        "branch_dropout",
        "scalar_branch_dropout",
        "head_dropout",
        "global_flux.layers_per_block",
        "flux_trend.pool_size",
        "flux_periodogram.pool_size",
        "local_centroid.kernel_size",
        "local_centroid.scalars",
        "momentum_dump.kernel_size",
        "transit_views.layers_per_block",
        "transit_views.init_filters",
        "unfolded_flux.kernel_size",
        "unfolded_flux.transit_kernel_size",
        "unfolded_flux.stat_activation",
        "diff_img.scalars",
        "stellar.scalars",
    }


def test_required_features_follow_enabled_branches() -> None:
    config = with_overrides(PAPER, {"diff_img.enabled": False})
    removed = required_features(PAPER) - required_features(config)

    assert removed == {
        "diff_imgs_std_trainset",
        "oot_imgs_std_trainset",
        "target_imgs",
        "quality",
    }
    assert "mag_shift_norm" in required_features(config)


def test_optimizer_config_validates() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        _ = replace(PAPER_OPTIMIZER, learning_rate=0.0)
