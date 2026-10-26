"""NASA's v1.0.0 normalization of the ExoMiner++ features.

Maps each model feature to its raw TFRecord column, computes the training-set
statistics (cached per training set and recipe) and normalizes a batch.
"""

__all__ = [
    "CENTROID_EPS",
    "CENTROID_OUT_OF_TRANSIT",
    "CLIP_FACTOR",
    "FEATURE_SOURCES",
    "IMAGE_EPS",
    "MAD_SCALE",
    "SCALAR_EPS",
    "SCALAR_RECIPES",
    "SOURCE_SHAPES",
    "STATS_SOURCES",
    "UNNORMALIZED_SCALARS",
    "FeatureStats",
    "NormStats",
    "ScalarRecipe",
    "compute_stats",
    "load_or_compute_stats",
    "mad_std",
    "normalize_batch",
    "normalize_scalar",
    "stats_key",
]

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from project.features import FEATURE_SHAPES, SCALAR_FEATURES, TRANSIT_BINS

# Scalars the model reads as stored in the TFRecords, without a `_norm` column
UNNORMALIZED_SCALARS: tuple[str, ...] = ("flux_local_stat_abs_min",)

_DERIVED_SOURCES = {
    "local_centr_view_std_noclip": "local_centr_view",
    "local_centr_view_std_noclip_var": "local_centr_view_var",
    "diff_imgs_std_trainset": "diff_imgs",
    "oot_imgs_std_trainset": "oot_imgs",
}


def _source(feature: str) -> str:
    if feature in _DERIVED_SOURCES:
        return _DERIVED_SOURCES[feature]
    if feature in SCALAR_FEATURES and feature not in UNNORMALIZED_SCALARS:
        return feature.removesuffix("_norm")
    return feature


FEATURE_SOURCES: dict[str, str] = {name: _source(name) for name in FEATURE_SHAPES}

SOURCE_SHAPES: dict[str, tuple[int, ...]] = {
    FEATURE_SOURCES[name]: shape for name, shape in FEATURE_SHAPES.items()
}


CLIP_FACTOR = 20.0
SCALAR_EPS = 1e-32
IMAGE_EPS = 1e-10
CENTROID_EPS = 1e-12
MAD_SCALE = 1.482602218505602
_TRANSIT_DURATIONS = 5
_IMAGES = ("diff_imgs", "oot_imgs")


def _out_of_transit_bins(
        num_bins: int,
        num_transit_durations: int,
) -> NDArray[np.int64]:
    """NASA's `get_out_of_transit_idxs_loc`."""
    bins_per_duration = num_bins / num_transit_durations
    out_of_transit = np.ones(num_bins, dtype=bool)
    in_transit = np.arange(
        int(np.floor((num_bins - bins_per_duration) / 2)),
        int(np.ceil((num_bins + bins_per_duration) / 2)),
    )
    out_of_transit[in_transit] = False
    return np.where(out_of_transit)[0].astype(np.int64)


CENTROID_OUT_OF_TRANSIT = _out_of_transit_bins(TRANSIT_BINS, _TRANSIT_DURATIONS)


@dataclass(frozen=True, slots=True)
class ScalarRecipe:
    """How NASA treats one scalar: placeholder for missing, replacement, log shift."""

    missing_value: float | None = None
    replace_value: float | None = None
    log_eps: float | None = None


_SCALAR_RECIPES: dict[str, ScalarRecipe] = {
    "boot_fap": ScalarRecipe(missing_value=-1, log_eps=SCALAR_EPS),
    "ruwe": ScalarRecipe(missing_value=0, replace_value=0),
    **{
        name: ScalarRecipe(missing_value=0)
        for name in (
            "tce_albedo_stat",
            "tce_cap_stat",
            "tce_hap_stat",
            "tce_dikco_msky",
            "tce_ptemp_stat",
        )
    },
    **{
        name: ScalarRecipe(missing_value=-1)
        for name in ("tce_dikco_msky_err", "tce_model_chisq")
    },
}

SCALAR_RECIPES: dict[str, ScalarRecipe] = {
    FEATURE_SOURCES[name]: _SCALAR_RECIPES.get(FEATURE_SOURCES[name], ScalarRecipe())
    for name in SCALAR_FEATURES
    if name not in UNNORMALIZED_SCALARS
}


STATS_SOURCES: tuple[str, ...] = (*SCALAR_RECIPES, "local_centr_view", *_IMAGES)


class FeatureStats(TypedDict):
    median: float
    mad_std: float


class NormStats(TypedDict):
    scalars: dict[str, FeatureStats]
    centroid: FeatureStats
    images: dict[str, FeatureStats]


def mad_std(values: NDArray[np.floating], *, ignore_nan: bool = False) -> float:
    """Scaled median absolute deviation, as `astropy.stats.mad_std` (no fallback)."""
    median = np.nanmedian if ignore_nan else np.median
    center = median(values)
    return float(MAD_SCALE * median(np.abs(values - center)))


def _scalar_values(name: str, values: NDArray[np.floating]) -> NDArray[np.float64]:
    recipe = SCALAR_RECIPES[name]
    kept = np.asarray(values, dtype=np.float64).ravel()
    if recipe.missing_value is not None:
        kept = kept[kept != recipe.missing_value]
    kept = kept[np.isfinite(kept)]
    if recipe.log_eps is not None:
        kept = np.log10(kept + recipe.log_eps)
    return kept


def compute_stats(features: Mapping[str, NDArray[np.float32]]) -> NormStats:
    """Statistics of the training set; `features` holds the `STATS_SOURCES` columns."""
    scalars: dict[str, FeatureStats] = {}
    for name in SCALAR_RECIPES:
        kept = _scalar_values(name, features[name])
        scalars[name] = {"median": float(np.median(kept)), "mad_std": mad_std(kept)}
    view = features["local_centr_view"][:, CENTROID_OUT_OF_TRANSIT]
    view = view.astype(np.float64).ravel()
    centroid: FeatureStats = {
        "median": float(np.median(view)),
        "mad_std": mad_std(view),
    }
    if not centroid["mad_std"] > 0:
        raise ValueError("centroid mad_std is zero over the out-of-transit bins")
    images: dict[str, FeatureStats] = {}
    for name in _IMAGES:
        pixels = features[name]
        images[name] = {
            "median": float(np.nanmedian(pixels)),
            "mad_std": mad_std(pixels, ignore_nan=True),
        }
    return {"scalars": scalars, "centroid": centroid, "images": images}


def _recipe() -> str:
    return repr((
        SCALAR_RECIPES,
        CLIP_FACTOR,
        SCALAR_EPS,
        IMAGE_EPS,
        CENTROID_EPS,
        MAD_SCALE,
        CENTROID_OUT_OF_TRANSIT.tolist(),
    ))


def stats_key(uids: NDArray[np.str_]) -> str:
    """Identify a training set by its sorted uids and the normalization recipe."""
    joined = "\n".join([_recipe(), *sorted(uids.tolist())])
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def load_or_compute_stats(
        cache_dir: Path,
        uids: NDArray[np.str_],
        load_features: Callable[[], Mapping[str, NDArray[np.float32]]],
) -> NormStats:
    """Statistics of the training set `uids`, cached under `cache_dir`.

    `load_features` is only called on a cache miss.
    """
    path = cache_dir / f"{stats_key(uids)}.json"
    if path.exists():
        return cast(NormStats, json.loads(path.read_text()))
    stats = compute_stats(load_features())
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(stats))
    os.replace(tmp, path)
    return stats


def normalize_scalar(
        name: str,
        values: NDArray[np.floating],
        stats: FeatureStats,
) -> NDArray[np.float32]:
    """Replace missing values, log-scale, clip to median +- 20 mad_std, standardize."""
    recipe = SCALAR_RECIPES[name]
    median, spread = stats["median"], stats["mad_std"]
    x = np.asarray(values, dtype=np.float64)
    missing = ~np.isfinite(x)
    if recipe.missing_value is not None:
        missing |= x == recipe.missing_value
    # the median is already in log space, an explicit replace value is not
    medianed = missing if recipe.replace_value is None else np.zeros_like(missing)
    fill = median if recipe.replace_value is None else recipe.replace_value
    x = np.where(missing, fill, x)
    if recipe.log_eps is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.where(medianed, x, np.log10(x + recipe.log_eps))
    x = np.clip(x, median - CLIP_FACTOR * spread, median + CLIP_FACTOR * spread)
    return ((x - median) / (spread + SCALAR_EPS)).astype(np.float32)


def normalize_batch(
        raw: Mapping[str, NDArray[np.float32]],
        stats: NormStats,
) -> dict[str, NDArray[np.float32]]:
    """Model features of a batch from its raw columns; arrays lead with the batch."""
    out: dict[str, NDArray[np.float32]] = {}
    for feature, source in FEATURE_SOURCES.items():
        value = raw[source]
        if source in SCALAR_RECIPES:
            out[feature] = normalize_scalar(source, value, stats["scalars"][source])
        elif source == "local_centr_view":
            centered = value - np.median(value, axis=-1, keepdims=True)
            scale = stats["centroid"]["mad_std"] + CENTROID_EPS
            out[feature] = (centered / scale).astype(np.float32)
        elif source == "local_centr_view_var":
            out[feature] = (value / stats["centroid"]["mad_std"]).astype(np.float32)
        elif source in _IMAGES:
            image = stats["images"][source]
            scale = image["mad_std"] + IMAGE_EPS
            out[feature] = ((value - image["median"]) / scale).astype(np.float32)
        else:
            out[feature] = value
    return out
