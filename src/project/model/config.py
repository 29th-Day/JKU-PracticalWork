"""Configuration of the ExoMiner++ architecture and its optimizer.

Every field is a plain `bool`, `int`, `float`, `str` or tuple of feature names, so a
config can be flattened into dotted keys (`to_flat`) and rebuilt from them
(`with_overrides`), e.g. from Optuna suggestions. Invalid configs raise `ValueError`
on construction.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Literal, get_args

from project.features import (
    CENTROID_VIEWS,
    DIFF_IMG_VIEWS,
    EVEN_VIEWS,
    FLUX_TREND_VIEWS,
    FULL_ORBIT_BINS,
    GLOBAL_FLUX_VIEWS,
    IMAGE_SIZE,
    LOCAL_FLUX_VIEWS,
    MOMENTUM_DUMP_VIEWS,
    ODD_VIEWS,
    PERIODOGRAM_BINS,
    PERIODOGRAM_VIEWS,
    QUALITY,
    SCALAR_FEATURES,
    TRANSIT_BINS,
    UNFOLDED_FLUX_VIEW,
    WEAK_SECONDARY_VIEWS,
)

__all__ = [
    "Activation",
    "ConfigValue",
    "ConvBranchConfig",
    "ConvInit",
    "ExoMinerConfig",
    "OptimizerConfig",
    "ScalarBranchConfig",
    "TransitViewsConfig",
    "UnfoldedFluxConfig",
    "pooled_length",
    "required_features",
    "to_flat",
    "with_overrides",
]

Activation = Literal["relu", "prelu"]
ConvInit = Literal["he", "glorot"]
ConfigValue = bool | int | float | str | tuple[str, ...]

_POSITIVE_FIELDS = (
    "blocks",
    "layers_per_block",
    "init_filters",
    "kernel_size",
    "transit_kernel_size",
    "pool_size",
    "stat_filters",
    "branch_units",
    "head_layers",
    "head_units",
)


@dataclass(frozen=True, slots=True)
class ScalarBranchConfig:
    """Branch that only reads scalar features (stellar, DV fit)."""

    enabled: bool
    scalars: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConvBranchConfig:
    """Branch with a conv stack; block `i` has `init_filters * 2**i` filters."""

    enabled: bool
    scalars: tuple[str, ...]
    blocks: int
    layers_per_block: int
    init_filters: int
    kernel_size: int
    pool_size: int


@dataclass(frozen=True, slots=True)
class TransitViewsConfig:
    """Shared extractor for the transit views of flux, weak secondary and odd/even."""

    blocks: int
    layers_per_block: int
    init_filters: int
    kernel_size: int
    pool_size: int
    flux_enabled: bool
    flux_scalars: tuple[str, ...]
    weak_secondary_enabled: bool
    weak_secondary_scalars: tuple[str, ...]
    odd_even_enabled: bool
    odd_even_scalars: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return self.flux_enabled or self.weak_secondary_enabled or self.odd_even_enabled


@dataclass(frozen=True, slots=True)
class UnfoldedFluxConfig:
    """Unfolded flux branch; the conv kernel spans `(transit_kernel_size, kernel_size)`.

    `transit_kernel_size = 1` convolves each transit on its own (paper), larger values
    mix neighbouring transits (NASA v1.0.0). Pooling only runs along the bins.
    """

    enabled: bool
    scalars: tuple[str, ...]
    blocks: int
    layers_per_block: int
    init_filters: int
    kernel_size: int
    transit_kernel_size: int
    pool_size: int
    stat_filters: int
    stat_activation: bool


BranchConfig = (
    ScalarBranchConfig | ConvBranchConfig | TransitViewsConfig | UnfoldedFluxConfig
)


@dataclass(frozen=True, slots=True)
class ExoMinerConfig:
    """Complete ExoMiner++ architecture; see `project.model.presets` for instances."""

    activation: Activation
    conv_init: ConvInit
    branch_units: int
    branch_dropout: float
    scalar_branch_dropout: bool
    head_layers: int
    head_units: int
    head_dropout: float
    global_flux: ConvBranchConfig
    flux_trend: ConvBranchConfig
    flux_periodogram: ConvBranchConfig
    local_centroid: ConvBranchConfig
    momentum_dump: ConvBranchConfig
    transit_views: TransitViewsConfig
    unfolded_flux: UnfoldedFluxConfig
    diff_img: ConvBranchConfig
    stellar: ScalarBranchConfig
    dv_tce_fit: ScalarBranchConfig

    def __post_init__(self) -> None:
        _validate_values(self)
        _validate_branches(self)


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Adam settings."""

    learning_rate: float
    beta1: float
    beta2: float
    eps: float

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.eps <= 0:
            raise ValueError("learning_rate and eps must be > 0")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("beta1 and beta2 must be in [0, 1)")


def pooled_length(length: int, pools: int, pool_size: int) -> int:
    """Length left after `pools` max-pools with stride 1 and 'valid' padding."""
    return length - pools * (pool_size - 1)


def to_flat(config: ExoMinerConfig) -> dict[str, ConfigValue]:
    """Flattens `config` into dotted keys such as `"global_flux.pool_size"`."""
    flat: dict[str, ConfigValue] = {}
    for field in fields(config):
        value: ConfigValue | BranchConfig = getattr(config, field.name)
        if isinstance(value, _BRANCH_TYPES):
            for branch_field in fields(value):
                key = f"{field.name}.{branch_field.name}"
                flat[key] = getattr(value, branch_field.name)
        else:
            flat[field.name] = value
    return flat


def with_overrides(
        config: ExoMinerConfig,
        overrides: Mapping[str, ConfigValue],
) -> ExoMinerConfig:
    """Returns a copy of `config` with the dotted keys in `overrides` replaced.

    Raises `KeyError` for unknown keys, `TypeError` for values of the wrong type and
    `ValueError` if the resulting config is invalid.
    """
    current = to_flat(config)
    unknown = sorted(overrides.keys() - current.keys())
    if unknown:
        raise KeyError(f"unknown config keys: {unknown}")
    top_level: dict[str, ConfigValue] = {}
    per_branch: dict[str, dict[str, ConfigValue]] = {}
    for key, value in overrides.items():
        checked = _checked_value(key, current[key], value)
        branch, _, name = key.rpartition(".")
        if branch:
            per_branch.setdefault(branch, {})[name] = checked
        else:
            top_level[name] = checked
    branches: dict[str, BranchConfig] = {
        branch: replace(getattr(config, branch), **values)
        for branch, values in per_branch.items()
    }
    return replace(config, **top_level, **branches)


def required_features(config: ExoMinerConfig) -> frozenset[str]:
    """Names of all input features the model built from `config` reads."""
    names: set[str] = set()
    for enabled, views, scalars in _branch_inputs(config):
        if enabled:
            names.update(views, scalars)
    return frozenset(names)


_BRANCH_TYPES = (
    ScalarBranchConfig,
    ConvBranchConfig,
    TransitViewsConfig,
    UnfoldedFluxConfig,
)


def _branch_inputs(
        config: ExoMinerConfig,
) -> list[tuple[bool, tuple[str, ...], tuple[str, ...]]]:
    """(enabled, views, scalars) of every branch."""
    transit = config.transit_views
    return [
        (config.global_flux.enabled, GLOBAL_FLUX_VIEWS, config.global_flux.scalars),
        (config.flux_trend.enabled, FLUX_TREND_VIEWS, config.flux_trend.scalars),
        (
            config.flux_periodogram.enabled,
            PERIODOGRAM_VIEWS,
            config.flux_periodogram.scalars,
        ),
        (
            config.local_centroid.enabled,
            CENTROID_VIEWS,
            config.local_centroid.scalars,
        ),
        (
            config.momentum_dump.enabled,
            MOMENTUM_DUMP_VIEWS,
            config.momentum_dump.scalars,
        ),
        (transit.flux_enabled, LOCAL_FLUX_VIEWS, transit.flux_scalars),
        (
            transit.weak_secondary_enabled,
            WEAK_SECONDARY_VIEWS,
            transit.weak_secondary_scalars,
        ),
        (transit.odd_even_enabled, ODD_VIEWS + EVEN_VIEWS, transit.odd_even_scalars),
        (
            config.unfolded_flux.enabled,
            (UNFOLDED_FLUX_VIEW,),
            config.unfolded_flux.scalars,
        ),
        (config.diff_img.enabled, (*DIFF_IMG_VIEWS, QUALITY), config.diff_img.scalars),
        (config.stellar.enabled, (), config.stellar.scalars),
        (config.dv_tce_fit.enabled, (), config.dv_tce_fit.scalars),
    ]


def _checked_value(key: str, current: ConfigValue, value: object) -> ConfigValue:
    """Returns `value` if it has the type of `current`.

    Ints are accepted for floats, and lists of names for tuples, so flat configs
    survive a round trip through JSON.
    """
    if isinstance(current, float):
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    elif isinstance(current, tuple):
        if isinstance(value, tuple | list):
            items = tuple(value)
            if all(isinstance(item, str) for item in items):
                return items
    else:
        same_kind = isinstance(value, bool) == isinstance(current, bool)
        if same_kind and isinstance(value, type(current)):
            return value
    raise TypeError(
        f"{key} expects {type(current).__name__}, got {type(value).__name__}"
    )


def _validate_values(config: ExoMinerConfig) -> None:
    """Checks the value range of every field."""
    if config.activation not in get_args(Activation):
        raise ValueError(f"activation must be one of {get_args(Activation)}")
    if config.conv_init not in get_args(ConvInit):
        raise ValueError(f"conv_init must be one of {get_args(ConvInit)}")
    for key, value in to_flat(config).items():
        name = key.rpartition(".")[2]
        if name in _POSITIVE_FIELDS and isinstance(value, int) and value < 1:
            raise ValueError(f"{key} must be >= 1, got {value}")
        if name.endswith("dropout") and isinstance(value, float):
            if not 0 <= value < 1:
                raise ValueError(f"{key} must be in [0, 1), got {value}")
        if name.endswith("scalars") and isinstance(value, tuple):
            unknown = sorted(set(value) - set(SCALAR_FEATURES))
            if unknown:
                raise ValueError(f"{key} contains unknown scalars: {unknown}")


def _validate_branches(config: ExoMinerConfig) -> None:
    """Checks that some branch is enabled and every enabled branch keeps its shape."""
    if not any(enabled for enabled, _, _ in _branch_inputs(config)):
        raise ValueError("at least one branch must be enabled")
    for name, scalar_branch in (
        ("stellar", config.stellar),
        ("dv_tce_fit", config.dv_tce_fit),
    ):
        if scalar_branch.enabled and not scalar_branch.scalars:
            raise ValueError(f"{name} is enabled but has no scalars")
    image = config.diff_img
    geometries = [
        ("global_flux", config.global_flux, FULL_ORBIT_BINS, config.global_flux.blocks),
        ("flux_trend", config.flux_trend, FULL_ORBIT_BINS, config.flux_trend.blocks),
        (
            "flux_periodogram",
            config.flux_periodogram,
            PERIODOGRAM_BINS,
            config.flux_periodogram.blocks,
        ),
        (
            "local_centroid",
            config.local_centroid,
            TRANSIT_BINS,
            config.local_centroid.blocks,
        ),
        (
            "momentum_dump",
            config.momentum_dump,
            TRANSIT_BINS,
            config.momentum_dump.blocks,
        ),
        (
            "transit_views",
            config.transit_views,
            TRANSIT_BINS,
            config.transit_views.blocks,
        ),
        (
            "unfolded_flux",
            config.unfolded_flux,
            TRANSIT_BINS,
            config.unfolded_flux.blocks,
        ),
        ("diff_img", image, IMAGE_SIZE, image.blocks * image.layers_per_block),
    ]
    for name, branch, length, pools in geometries:
        remaining = pooled_length(length, pools, branch.pool_size)
        if branch.enabled and remaining < 1:
            size = branch.pool_size
            raise ValueError(f"{name}: {pools} pools of size {size} leave {remaining}")
