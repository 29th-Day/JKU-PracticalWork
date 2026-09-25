# ExoMiner++ Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configurable PyTorch Lightning reimplementation of the ExoMiner++ transit classifier, with presets for the paper, the Zenodo diagram and NASA's v1.0.0 code, in `src/project/`.

**Architecture:**
- `project.features` is the data↔model contract: feature names, shapes and the `Batch` types.
- `project.model` holds the frozen-dataclass config (flattenable for Optuna), the presets, the layer building blocks, the branches and the `ExoMinerPlusPlus` network.
- `project.metrics` holds the paper's metrics.
- `project.classifier` joins model and metrics in a `LightningModule`.
- The data code (later) will depend only on `project.features`.

**Tech Stack:** Python 3.12, uv, torch 2.14 (ROCm 7.2 or CUDA 13.0 via extras), lightning 2.6, torchmetrics 1.9, pytest 9, basedpyright 1.40, ruff 0.16.

**Spec:** `docs/superpowers/specs/2026-09-24-exominer-plusplus-model-design.md`

## Global Constraints

- Python `>=3.12`; `torch>=2.14`, `lightning>=2.6`, `torchmetrics>=1.9`; dev: `basedpyright>=1.40`, `pytest>=9.1`, `ruff>=0.16`.
- Torch comes from the `rocm` extra (index `https://download.pytorch.org/whl/rocm7.2`, plus `triton-rocm` from the same index) or the `cuda` extra (index `https://download.pytorch.org/whl/cu130`). Resolution is restricted to `sys_platform == 'linux'`.
- Follow `style.md`:
  - 4-space indent;
  - in a multi-line `def`, the arguments get **8** spaces and the closing `) -> T:` aligns with `def`;
  - double quotes;
  - `__all__` in every `src/` module;
  - comments are single lines, docstrings may span lines;
  - every overriding method carries `@override`.
- **Never run `ruff format`**: it undoes the `def` argument indentation `style.md` requires. `uv run ruff check --fix` is allowed; it only sorts imports.
- After every task, `uv run basedpyright` prints `0 errors, 0 warnings, 0 notes` and `uv run ruff check` prints `All checks passed!`.
- Import rules:

  | Module | May import |
  |---|---|
  | `project.features` | torch only |
  | `project.model` | `project.features` |
  | `project.metrics` | torchmetrics only |
  | `project.classifier` | all three |

- Do not modify `src/project/__init__.py`, `src/project/abc.py` or `scripts/test.py`.
- Tests run on CPU with random tensors; no dataset is needed.
- The code in this plan was run in a scratch project with torch 2.14.0, lightning 2.6.6, torchmetrics 1.9.0, pytest 9.1.1, basedpyright 1.40.1 and ruff 0.16.9: 80 tests pass and both checkers report nothing. Each task was also replayed on its own: every task's tests pass with only the files created so far. **Copy the code verbatim.**

## Review Focus

Five failure modes the spec implies but a plain reading of it wouldn't test, most likely first. Each has a test in the task that owns the code:

1. **NASA-style inputs with a trailing dimension** (scalars `(B, 1)`, views `(B, L, 1)`): expect a `ValueError` naming the feature, not a cryptic `torch.cat` error.
   Test: Task 4, `test_wrong_feature_shape_raises_value_error`.
2. **A flat config saved as JSON and loaded back** (tuples become lists): expect `with_overrides` to rebuild the identical config.
   Test: Task 2, `test_flat_round_trip_through_json`.
3. **Scoring unlabeled TCEs**, where a batch has no `label`: expect `predict_step` to work.
   Test: Task 6, `test_predict_without_labels`.
4. **A validation set with only one class**: expect warnings from torchmetrics but no crash, and `val/loss` still logged.
   Test: Task 6, `test_validation_with_a_single_class`.
5. **Only one transit-view diagnostic enabled** (e.g. only odd/even): expect the shared extractor and the head width to adapt.
   Test: Task 4, `test_single_transit_diagnostic`.

---

### Task 1: Project setup and the feature contract

**Files:**
- Modify: `pyproject.toml` (replace the whole file)
- Modify: `README.md` (append a section)
- Create: `src/project/features.py`
- Create: `tests/conftest.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing.
- Produces, in `project.features`:
  - int constants `TRANSIT_BINS = 31`, `FULL_ORBIT_BINS = 301`, `PERIODOGRAM_BINS = 674`, `UNFOLDED_TRANSITS = 20`, `SECTORS = 5`, `IMAGE_SIZE = 33`;
  - view-name tuples `GLOBAL_FLUX_VIEWS`, `FLUX_TREND_VIEWS`, `PERIODOGRAM_VIEWS`, `CENTROID_VIEWS`, `MOMENTUM_DUMP_VIEWS`, `LOCAL_FLUX_VIEWS`, `WEAK_SECONDARY_VIEWS`, `ODD_VIEWS`, `EVEN_VIEWS`, `DIFF_IMG_VIEWS` (each `tuple[str, ...]`);
  - `UNFOLDED_FLUX_VIEW: str`, `QUALITY: str`, `SCALAR_FEATURES: tuple[str, ...]` (33 names);
  - `FEATURE_SHAPES: dict[str, tuple[int, ...]]` (per-example shapes);
  - `class UnlabeledBatch(TypedDict)` with `features: dict[str, Tensor]`;
  - `class Batch(UnlabeledBatch)` with `label: Tensor`.

- [ ] **Step 1: Create a feature branch**

```bash
git switch -c feat/exominer-model
```

Expected: `Switched to a new branch 'feat/exominer-model'`. The user's uncommitted `pyproject.toml` edit comes along; Step 2 replaces that file anyway.

- [ ] **Step 2: Replace `pyproject.toml`**

Write exactly this content:

```toml
[project]
name = "project"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "29th-Day", email = "43274844+29th-Day@users.noreply.github.com" }
]
requires-python = ">=3.12"
dependencies = [
    "lightning>=2.6",
    "torchmetrics>=1.9",
]

[project.optional-dependencies]
rocm = ["torch>=2.14", "triton-rocm"]
cuda = ["torch>=2.14"]

[project.scripts]
# project = "project:main"

[dependency-groups]
dev = [
    "basedpyright>=1.40",
    "pytest>=9.1",
    "ruff>=0.16",
]

[build-system]
requires = ["uv_build>=0.12.17,<0.13.0"]
build-backend = "uv_build"

[tool.uv]
environments = ["sys_platform == 'linux'"]
conflicts = [[{ extra = "rocm" }, { extra = "cuda" }]]

[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm7.2"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-rocm", extra = "rocm" },
    { index = "pytorch-cu130", extra = "cuda" },
]
triton-rocm = [{ index = "pytorch-rocm", extra = "rocm" }]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
filterwarnings = [
    "ignore:Using padding='same' with even kernel lengths:UserWarning",
    "ignore:.*does not have many workers:UserWarning",
    "ignore:`isinstance\\(treespec, LeafSpec\\)` is deprecated",
]

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "Q"]

[tool.basedpyright]
pythonVersion = "3.12"
typeCheckingMode = "recommended"
include = ["src", "tests"]
# nn.Module.__call__ and parts of torch/Lightning are typed as Any or left untyped.
reportAny = false
reportUnknownMemberType = false
reportUnknownVariableType = false
reportUnknownArgumentType = false
# In-place torch APIs (nn.init.*_, backward, optimizer.step) return values nobody uses.
reportUnusedCallResult = false
```

- [ ] **Step 3: Append the setup instructions to `README.md`**

Keep the existing text and append:

````markdown

## Setup

Install the dependencies with the torch build that matches your GPU:

```bash
uv sync --extra rocm   # AMD, ROCm 7.2
uv sync --extra cuda   # NVIDIA, CUDA 13.0 (driver >= 580)
```

Always pass one of the extras. Without one, uv installs PyPI's default torch build.

## Checks

```bash
uv run pytest
uv run basedpyright
uv run ruff check
```
````

- [ ] **Step 4: Lock and install**

Run: `uv sync --extra rocm` (on an NVIDIA machine, `uv sync --extra cuda`).

Expected:
- `uv.lock` is rewritten;
- the install includes `torch==2.14.0+rocm7.2`, `triton-rocm==3.8.0`, `lightning==2.6.x`, `torchmetrics==1.9.x`, `pytest`, `basedpyright` and `ruff`.

The first run downloads several GB of torch wheels.

Then run: `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
Expected: `2.14.0+rocm7.2 True` (ROCm exposes the GPU through the `torch.cuda` API).

- [ ] **Step 5: Write the tests**

Create `tests/conftest.py`:

```python
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
```

Create `tests/test_features.py`:

```python
from project.features import (
    CENTROID_VIEWS,
    DIFF_IMG_VIEWS,
    EVEN_VIEWS,
    FEATURE_SHAPES,
    FLUX_TREND_VIEWS,
    GLOBAL_FLUX_VIEWS,
    LOCAL_FLUX_VIEWS,
    MOMENTUM_DUMP_VIEWS,
    ODD_VIEWS,
    PERIODOGRAM_VIEWS,
    QUALITY,
    SCALAR_FEATURES,
    UNFOLDED_FLUX_VIEW,
    WEAK_SECONDARY_VIEWS,
)


def test_view_shapes() -> None:
    full_orbit = GLOBAL_FLUX_VIEWS + FLUX_TREND_VIEWS
    transit = (
        CENTROID_VIEWS
        + MOMENTUM_DUMP_VIEWS
        + LOCAL_FLUX_VIEWS
        + WEAK_SECONDARY_VIEWS
        + ODD_VIEWS
        + EVEN_VIEWS
    )

    assert all(FEATURE_SHAPES[name] == (301,) for name in full_orbit)
    assert all(FEATURE_SHAPES[name] == (674,) for name in PERIODOGRAM_VIEWS)
    assert all(FEATURE_SHAPES[name] == (31,) for name in transit)
    assert FEATURE_SHAPES[UNFOLDED_FLUX_VIEW] == (20, 31)
    assert all(FEATURE_SHAPES[name] == (5, 33, 33) for name in DIFF_IMG_VIEWS)
    assert FEATURE_SHAPES[QUALITY] == (5,)


def test_scalars_are_scalar_shaped() -> None:
    assert len(SCALAR_FEATURES) == len(set(SCALAR_FEATURES)) == 33
    assert all(FEATURE_SHAPES[name] == () for name in SCALAR_FEATURES)


def test_feature_shapes_has_no_unlisted_features() -> None:
    views = (
        GLOBAL_FLUX_VIEWS
        + FLUX_TREND_VIEWS
        + PERIODOGRAM_VIEWS
        + CENTROID_VIEWS
        + MOMENTUM_DUMP_VIEWS
        + LOCAL_FLUX_VIEWS
        + WEAK_SECONDARY_VIEWS
        + ODD_VIEWS
        + EVEN_VIEWS
        + DIFF_IMG_VIEWS
    )
    listed = {*views, UNFOLDED_FLUX_VIEW, QUALITY, *SCALAR_FEATURES}

    assert set(FEATURE_SHAPES) == listed
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: a collection error, `ModuleNotFoundError: No module named 'project.features'`.

- [ ] **Step 7: Write `src/project/features.py`**

```python
"""Feature names, shapes and the batch layout shared by the data code and the model.

Names follow NASA's ExoMiner++ TFRecord features. Shapes are per example; batching
adds a leading dimension. Scalars have shape `()`.
"""

from typing import TypedDict

from torch import Tensor

__all__ = [
    "CENTROID_VIEWS",
    "DIFF_IMG_VIEWS",
    "EVEN_VIEWS",
    "FEATURE_SHAPES",
    "FLUX_TREND_VIEWS",
    "FULL_ORBIT_BINS",
    "GLOBAL_FLUX_VIEWS",
    "IMAGE_SIZE",
    "LOCAL_FLUX_VIEWS",
    "MOMENTUM_DUMP_VIEWS",
    "ODD_VIEWS",
    "PERIODOGRAM_BINS",
    "PERIODOGRAM_VIEWS",
    "QUALITY",
    "SCALAR_FEATURES",
    "SECTORS",
    "TRANSIT_BINS",
    "UNFOLDED_FLUX_VIEW",
    "UNFOLDED_TRANSITS",
    "WEAK_SECONDARY_VIEWS",
    "Batch",
    "UnlabeledBatch",
]

TRANSIT_BINS = 31
FULL_ORBIT_BINS = 301
PERIODOGRAM_BINS = 674
UNFOLDED_TRANSITS = 20
SECTORS = 5
IMAGE_SIZE = 33

GLOBAL_FLUX_VIEWS = ("global_flux_view_fluxnorm", "global_flux_view_fluxnorm_var")
FLUX_TREND_VIEWS = ("flux_trend_global_norm", "flux_trend_global_norm_var")
PERIODOGRAM_VIEWS = ("pgram_smooth_norm", "pgram_tpm_smooth_norm")
CENTROID_VIEWS = ("local_centr_view_std_noclip", "local_centr_view_std_noclip_var")
MOMENTUM_DUMP_VIEWS = ("local_momentum_dump_view", "local_momentum_dump_view_var")
LOCAL_FLUX_VIEWS = ("local_flux_view_fluxnorm", "local_flux_view_fluxnorm_var")
WEAK_SECONDARY_VIEWS = (
    "local_weak_secondary_view_selfnorm",
    "local_weak_secondary_view_selfnorm_var",
)
ODD_VIEWS = ("local_flux_odd_view_fluxnorm", "local_flux_odd_view_fluxnorm_var")
EVEN_VIEWS = ("local_flux_even_view_fluxnorm", "local_flux_even_view_fluxnorm_var")
UNFOLDED_FLUX_VIEW = "unfolded_local_flux_view_fluxnorm"
DIFF_IMG_VIEWS = ("diff_imgs_std_trainset", "oot_imgs_std_trainset", "target_imgs")
QUALITY = "quality"

SCALAR_FEATURES = (
    "boot_fap_norm",
    "flux_even_local_stat_abs_min_norm",
    "flux_global_stat_abs_min_norm",
    "flux_local_stat_abs_min",
    "flux_odd_local_stat_abs_min_norm",
    "flux_trend_global_stat_max_norm",
    "flux_trend_global_stat_min_norm",
    "flux_weak_secondary_local_stat_abs_min_norm",
    "mag_shift_norm",
    "pgram_smooth_max_power_norm",
    "pgram_tpm_smooth_max_power_norm",
    "ruwe_norm",
    "tce_albedo_stat_norm",
    "tce_cap_stat_norm",
    "tce_dikco_msky_err_norm",
    "tce_dikco_msky_norm",
    "tce_hap_stat_norm",
    "tce_max_mult_ev_norm",
    "tce_max_sngle_ev_norm",
    "tce_maxmes_norm",
    "tce_model_chisq_norm",
    "tce_num_transits_norm",
    "tce_num_transits_obs_norm",
    "tce_period_norm",
    "tce_prad_norm",
    "tce_ptemp_stat_norm",
    "tce_robstat_norm",
    "tce_sdens_norm",
    "tce_slogg_norm",
    "tce_smass_norm",
    "tce_smet_norm",
    "tce_sradius_norm",
    "tce_steff_norm",
)

_TRANSIT_VIEWS = (
    CENTROID_VIEWS
    + MOMENTUM_DUMP_VIEWS
    + LOCAL_FLUX_VIEWS
    + WEAK_SECONDARY_VIEWS
    + ODD_VIEWS
    + EVEN_VIEWS
)

FEATURE_SHAPES: dict[str, tuple[int, ...]] = {
    **{name: (FULL_ORBIT_BINS,) for name in GLOBAL_FLUX_VIEWS + FLUX_TREND_VIEWS},
    **{name: (PERIODOGRAM_BINS,) for name in PERIODOGRAM_VIEWS},
    **{name: (TRANSIT_BINS,) for name in _TRANSIT_VIEWS},
    UNFOLDED_FLUX_VIEW: (UNFOLDED_TRANSITS, TRANSIT_BINS),
    **{name: (SECTORS, IMAGE_SIZE, IMAGE_SIZE) for name in DIFF_IMG_VIEWS},
    QUALITY: (SECTORS,),
    **{name: () for name in SCALAR_FEATURES},
}


class UnlabeledBatch(TypedDict):
    """Batch without labels, e.g. unlabeled TCEs to score.

    `features` maps feature names to tensors of shape `(B, *FEATURE_SHAPES[name])`.
    """

    features: dict[str, Tensor]


class Batch(UnlabeledBatch):
    """Labeled batch; `label` has shape `(B,)`, 1.0 for planets (KP/CP), else 0.0."""

    label: Tensor
```

- [ ] **Step 8: Run tests and checks**

Run: `uv run pytest tests/test_features.py -v`
Expected: `3 passed`.

Run: `uv run basedpyright && uv run ruff check`
Expected: `0 errors, 0 warnings, 0 notes` and `All checks passed!`.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock README.md src/project/features.py tests/conftest.py tests/test_features.py docs/superpowers
git commit -m "$(cat <<'EOF'
Add project setup and ExoMiner++ feature contract

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Configuration and presets

**Files:**
- Create: `src/project/model/config.py`
- Create: `src/project/model/presets.py`
- Create: `src/project/model/__init__.py` (interim version; Task 4 replaces it)
- Test: `tests/model/test_config.py`

**Interfaces:**
- Consumes: `project.features` (Task 1). The feature constants are used by `required_features` and by the geometry checks.
- Produces, in `project.model.config`:
  - `Activation = Literal["relu", "prelu"]`, `ConvInit = Literal["he", "glorot"]`, `ConfigValue = bool | int | float | str | tuple[str, ...]`.
  - Frozen, slotted dataclasses:
    - `ScalarBranchConfig(enabled, scalars)`;
    - `ConvBranchConfig(enabled, scalars, blocks, layers_per_block, init_filters, kernel_size, pool_size)`;
    - `TransitViewsConfig(blocks, layers_per_block, init_filters, kernel_size, pool_size, flux_enabled, flux_scalars, weak_secondary_enabled, weak_secondary_scalars, odd_even_enabled, odd_even_scalars)`, with an `enabled` property;
    - `UnfoldedFluxConfig(enabled, scalars, blocks, layers_per_block, init_filters, kernel_size, transit_kernel_size, pool_size, stat_filters, stat_activation)`;
    - `ExoMinerConfig(activation, conv_init, branch_units, branch_dropout, scalar_branch_dropout, head_layers, head_units, head_dropout, global_flux, flux_trend, flux_periodogram, local_centroid, momentum_dump, transit_views, unfolded_flux, diff_img, stellar, dv_tce_fit)`;
    - `OptimizerConfig(learning_rate, beta1, beta2, eps)`.
  - Functions:
    - `pooled_length(length: int, pools: int, pool_size: int) -> int`;
    - `to_flat(config: ExoMinerConfig) -> dict[str, ConfigValue]`;
    - `with_overrides(config: ExoMinerConfig, overrides: Mapping[str, ConfigValue]) -> ExoMinerConfig`;
    - `required_features(config: ExoMinerConfig) -> frozenset[str]`.
- Produces, in `project.model.presets`: `PAPER`, `DIAGRAM`, `REPO_V1` (`ExoMinerConfig`) and `PAPER_OPTIMIZER`, `REPO_V1_OPTIMIZER` (`OptimizerConfig`).
- `project.model` re-exports all of the above that tests import.

- [ ] **Step 1: Write the tests**

Create `tests/model/test_config.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/model/test_config.py -v`
Expected: a collection error, `ModuleNotFoundError: No module named 'project.model'`.

- [ ] **Step 3: Write `src/project/model/config.py`**

```python
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
```

- [ ] **Step 4: Write `src/project/model/presets.py`**

```python
"""Preset configs for the three reference descriptions of ExoMiner++.

- `PAPER`: the paper text (App. A, Table 18, Sec. 3.1); gaps are filled from the
  Zenodo diagram, then from NASA's v1.0.0 code.
- `DIAGRAM`: the Zenodo architecture diagram of the model trained for the paper.
- `REPO_V1`: NASA ExoMiner v1.0.0 (`models_keras.py`, `exominer_plus_plus.yaml`).
"""

from dataclasses import replace

from project.model.config import (
    ConvBranchConfig,
    ExoMinerConfig,
    OptimizerConfig,
    ScalarBranchConfig,
    TransitViewsConfig,
    UnfoldedFluxConfig,
)

__all__ = [
    "DIAGRAM",
    "PAPER",
    "PAPER_OPTIMIZER",
    "REPO_V1",
    "REPO_V1_OPTIMIZER",
]

_STELLAR_SCALARS = (
    "tce_sdens_norm",
    "tce_steff_norm",
    "tce_smet_norm",
    "tce_slogg_norm",
    "tce_smass_norm",
    "tce_sradius_norm",
)
_DV_TCE_FIT_SCALARS = (
    "boot_fap_norm",
    "tce_cap_stat_norm",
    "tce_hap_stat_norm",
    "tce_period_norm",
    "tce_max_mult_ev_norm",
    "tce_max_sngle_ev_norm",
    "tce_robstat_norm",
    "tce_model_chisq_norm",
    "tce_prad_norm",
)
_DIKCO_SCALARS = ("tce_dikco_msky_norm", "tce_dikco_msky_err_norm")

PAPER = ExoMinerConfig(
    activation="relu",
    conv_init="he",
    branch_units=4,
    branch_dropout=0.1211,
    scalar_branch_dropout=True,
    head_layers=4,
    head_units=512,
    head_dropout=0.0215,
    global_flux=ConvBranchConfig(
        enabled=True,
        scalars=("flux_global_stat_abs_min_norm",),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=5,
        pool_size=8,
    ),
    flux_trend=ConvBranchConfig(
        enabled=True,
        scalars=("flux_trend_global_stat_max_norm", "flux_trend_global_stat_min_norm"),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=5,
        pool_size=8,
    ),
    flux_periodogram=ConvBranchConfig(
        enabled=True,
        scalars=("pgram_smooth_max_power_norm", "pgram_tpm_smooth_max_power_norm"),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=5,
        pool_size=8,
    ),
    local_centroid=ConvBranchConfig(
        enabled=True,
        scalars=(*_DIKCO_SCALARS, "mag_shift_norm", "ruwe_norm"),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=6,
        pool_size=4,
    ),
    momentum_dump=ConvBranchConfig(
        enabled=True,
        scalars=(),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=6,
        pool_size=4,
    ),
    transit_views=TransitViewsConfig(
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=6,
        pool_size=4,
        flux_enabled=True,
        flux_scalars=("flux_local_stat_abs_min",),
        weak_secondary_enabled=True,
        weak_secondary_scalars=(
            "tce_albedo_stat_norm",
            "tce_ptemp_stat_norm",
            "tce_maxmes_norm",
            "flux_weak_secondary_local_stat_abs_min_norm",
        ),
        odd_even_enabled=True,
        odd_even_scalars=(
            "flux_odd_local_stat_abs_min_norm",
            "flux_even_local_stat_abs_min_norm",
        ),
    ),
    unfolded_flux=UnfoldedFluxConfig(
        enabled=True,
        scalars=("tce_num_transits_obs_norm", "tce_num_transits_norm"),
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=6,
        transit_kernel_size=1,
        pool_size=4,
        stat_filters=4,
        stat_activation=True,
    ),
    diff_img=ConvBranchConfig(
        enabled=True,
        scalars=("mag_shift_norm",),
        blocks=3,
        layers_per_block=3,
        init_filters=4,
        kernel_size=3,
        pool_size=2,
    ),
    stellar=ScalarBranchConfig(enabled=True, scalars=_STELLAR_SCALARS),
    dv_tce_fit=ScalarBranchConfig(enabled=True, scalars=_DV_TCE_FIT_SCALARS),
)

DIAGRAM = replace(
    PAPER,
    activation="prelu",
    branch_units=3,
    scalar_branch_dropout=False,
    flux_trend=replace(PAPER.flux_trend, pool_size=4),
    flux_periodogram=replace(PAPER.flux_periodogram, pool_size=4),
    unfolded_flux=replace(PAPER.unfolded_flux, stat_activation=False),
)

REPO_V1 = replace(
    PAPER,
    activation="prelu",
    conv_init="glorot",
    branch_units=3,
    branch_dropout=0.1211390996398814,
    scalar_branch_dropout=False,
    head_dropout=0.021491238286347598,
    global_flux=replace(PAPER.global_flux, layers_per_block=5),
    flux_trend=replace(PAPER.flux_trend, pool_size=4),
    flux_periodogram=replace(PAPER.flux_periodogram, pool_size=4),
    local_centroid=replace(
        PAPER.local_centroid,
        scalars=("mag_shift_norm",),
        kernel_size=5,
    ),
    momentum_dump=replace(PAPER.momentum_dump, kernel_size=5),
    transit_views=replace(PAPER.transit_views, layers_per_block=1, init_filters=4),
    unfolded_flux=replace(
        PAPER.unfolded_flux,
        kernel_size=1,
        transit_kernel_size=6,
        stat_activation=False,
    ),
    diff_img=replace(PAPER.diff_img, scalars=("mag_shift_norm", *_DIKCO_SCALARS)),
    stellar=replace(PAPER.stellar, scalars=(*_STELLAR_SCALARS, "ruwe_norm")),
)

PAPER_OPTIMIZER = OptimizerConfig(
    learning_rate=4.18e-5,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
)

REPO_V1_OPTIMIZER = replace(PAPER_OPTIMIZER, learning_rate=4.176171931455995e-05)
```

- [ ] **Step 5: Write the interim `src/project/model/__init__.py`**

```python
"""The ExoMiner++ model, its configuration and presets."""

from project.model.config import (
    ConfigValue,
    ExoMinerConfig,
    OptimizerConfig,
    required_features,
    to_flat,
    with_overrides,
)
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
    "OptimizerConfig",
    "required_features",
    "to_flat",
    "with_overrides",
]
```

- [ ] **Step 6: Run tests and checks**

Run: `uv run pytest tests/model/test_config.py -v`
Expected: `27 passed`.

Run: `uv run basedpyright && uv run ruff check`
Expected: `0 errors, 0 warnings, 0 notes` and `All checks passed!`.

- [ ] **Step 7: Commit**

```bash
git add src/project/model tests/model/test_config.py
git commit -m "$(cat <<'EOF'
Add ExoMiner++ config, validation, flat overrides and presets

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Layer building blocks

**Files:**
- Create: `src/project/model/layers.py`
- Test: `tests/model/test_layers.py`

**Interfaces:**
- Consumes: `Activation`, `ConvInit` and `pooled_length` from `project.model.config` (Task 2).
- Produces, in `project.model.layers`:
  - `make_activation(activation: Activation) -> nn.Module`;
  - `ConvStack(in_channels, blocks, layers_per_block, init_filters, kernel_size: tuple[int, int], pool_size: tuple[int, int], activation, pool_every_layer=False)`, an `nn.Sequential` with attribute `out_channels: int`;
  - `BranchHead(in_features, units, activation, dropout)`, with attributes `linear`, `activation` and `dropout`, and `forward(x) -> Tensor`;
  - `FullWidthLinear(nn.Linear)`;
  - `with_scalars(features: Tensor, inputs: Mapping[str, Tensor], scalars: Sequence[str]) -> Tensor`;
  - `init_weights(module: nn.Module, conv_init: ConvInit) -> None`.

- [ ] **Step 1: Write the tests**

Create `tests/model/test_layers.py`:

```python
import pytest
import torch
from torch import nn

from project.model.config import pooled_length
from project.model.layers import (
    BranchHead,
    ConvStack,
    FullWidthLinear,
    init_weights,
    make_activation,
    with_scalars,
)


@pytest.mark.parametrize(
    ("length", "pools", "pool_size", "expected"),
    [(301, 2, 8, 287), (674, 2, 8, 660), (31, 2, 4, 25), (33, 9, 2, 24)],
)
def test_pooled_length(length: int, pools: int, pool_size: int, expected: int) -> None:
    assert pooled_length(length, pools, pool_size) == expected


def test_make_activation_prelu_starts_as_relu() -> None:
    activation = make_activation("prelu")
    x = torch.tensor([-2.0, 3.0])

    assert isinstance(activation, nn.PReLU)
    assert torch.equal(activation(x), torch.tensor([0.0, 3.0]))


def test_conv_stack_time_series_shape() -> None:
    stack = ConvStack(
        in_channels=2,
        blocks=2,
        layers_per_block=3,
        init_filters=8,
        kernel_size=(1, 6),
        pool_size=(1, 4),
        activation="relu",
    )

    out = stack(torch.randn(3, 2, 4, 31))

    assert out.shape == (3, 16, 4, 25)
    assert stack.out_channels == 16
    assert sum(isinstance(layer, nn.Conv2d) for layer in stack) == 6


def test_conv_stack_pool_every_layer_shape() -> None:
    stack = ConvStack(
        in_channels=3,
        blocks=3,
        layers_per_block=3,
        init_filters=4,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        activation="relu",
        pool_every_layer=True,
    )

    assert stack(torch.randn(2, 3, 33, 33)).shape == (2, 16, 24, 24)


def test_conv_stack_even_kernel_keeps_rows_separate() -> None:
    stack = ConvStack(
        in_channels=1,
        blocks=1,
        layers_per_block=1,
        init_filters=1,
        kernel_size=(1, 6),
        pool_size=(1, 1),
        activation="relu",
    )
    x = torch.randn(1, 1, 2, 31)
    changed = x.clone()
    changed[:, :, 1] += 1.0

    out, out_changed = stack(x), stack(changed)

    assert out.shape == (1, 1, 2, 31)
    assert torch.equal(out[:, :, 0], out_changed[:, :, 0])


def test_branch_head_shape_and_dropout() -> None:
    head = BranchHead(in_features=10, units=4, activation="relu", dropout=0.5)

    assert head(torch.randn(3, 10)).shape == (3, 4)
    assert head.dropout.p == 0.5


def test_with_scalars_appends_columns() -> None:
    features = torch.zeros(2, 3, 5)
    inputs = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}

    combined = with_scalars(features, inputs, ("a", "b"))

    assert combined.shape == (2, 17)
    assert torch.equal(combined[:, -2:], torch.tensor([[1.0, 3.0], [2.0, 4.0]]))


def test_init_weights_zeroes_biases_and_uses_he_for_convs() -> None:
    conv = nn.Conv2d(64, 64, kernel_size=3)
    full_width = FullWidthLinear(4096, 64)
    linear = nn.Linear(4096, 64)

    for module in (conv, full_width, linear):
        init_weights(module, conv_init="he")

    assert conv.bias is not None
    assert torch.count_nonzero(conv.bias) == 0
    assert conv.weight.std().item() == pytest.approx((2 / (64 * 9)) ** 0.5, rel=0.1)
    assert full_width.weight.std().item() == pytest.approx((2 / 4096) ** 0.5, rel=0.1)
    xavier_std = (2 / (4096 + 64)) ** 0.5
    assert linear.weight.std().item() == pytest.approx(xavier_std, rel=0.1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/model/test_layers.py -v`
Expected: a collection error, `ModuleNotFoundError: No module named 'project.model.layers'`.

- [ ] **Step 3: Write `src/project/model/layers.py`**

```python
"""Building blocks shared by the ExoMiner++ branches."""

from collections.abc import Mapping, Sequence
from typing import override

import torch
from torch import Tensor, nn

from project.model.config import Activation, ConvInit

__all__ = [
    "BranchHead",
    "ConvStack",
    "FullWidthLinear",
    "init_weights",
    "make_activation",
    "with_scalars",
]


def make_activation(activation: Activation) -> nn.Module:
    """ReLU, or PReLU with a single slope initialized to 0 as in NASA's Keras code."""
    if activation == "prelu":
        return nn.PReLU(init=0.0)
    return nn.ReLU()


class ConvStack(nn.Sequential):
    """Blocks of `layers_per_block` x (Conv2d -> activation), each followed by a pool.

    Block `i` has `init_filters * 2**i` filters. Convolutions use 'same' padding and
    stride 1; max-pools use 'valid' padding and stride 1. With `pool_every_layer`,
    every convolution is followed by a pool instead of every block.
    """

    def __init__(
            self,
            in_channels: int,
            blocks: int,
            layers_per_block: int,
            init_filters: int,
            kernel_size: tuple[int, int],
            pool_size: tuple[int, int],
            activation: Activation,
            pool_every_layer: bool = False,
    ) -> None:
        layers: list[nn.Module] = []
        channels = in_channels
        for block in range(blocks):
            filters = init_filters * 2**block
            for _ in range(layers_per_block):
                layers.append(nn.Conv2d(channels, filters, kernel_size, padding="same"))
                layers.append(make_activation(activation))
                if pool_every_layer:
                    layers.append(nn.MaxPool2d(pool_size, stride=1))
                channels = filters
            if not pool_every_layer:
                layers.append(nn.MaxPool2d(pool_size, stride=1))
        super().__init__(*layers)
        self.out_channels: int = channels


class BranchHead(nn.Module):
    """Linear -> activation -> dropout that ends every branch."""

    def __init__(
            self,
            in_features: int,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(in_features, units)
        self.activation: nn.Module = make_activation(activation)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.linear(x)))


class FullWidthLinear(nn.Linear):
    """Linear layer standing in for NASA's full-width Conv1D; initialized as a conv."""


def with_scalars(
        features: Tensor,
        inputs: Mapping[str, Tensor],
        scalars: Sequence[str],
) -> Tensor:
    """Flattens `features` to `(B, n)` and appends the named scalars as columns."""
    columns = [features.flatten(1), *(inputs[name].unsqueeze(1) for name in scalars)]
    return torch.cat(columns, dim=1)


def init_weights(module: nn.Module, conv_init: ConvInit) -> None:
    """Initializes convs per `conv_init`, linear layers with Xavier, biases to 0."""
    if not isinstance(module, nn.Conv2d | nn.Linear):
        return
    is_conv = isinstance(module, nn.Conv2d | FullWidthLinear)
    if is_conv and conv_init == "he":
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
```

- [ ] **Step 4: Run tests and checks**

Run: `uv run pytest tests/model/test_layers.py -v`
Expected: `11 passed`. PyTorch's warning about 'same' padding with even kernels is filtered in `pyproject.toml`.

Run: `uv run basedpyright && uv run ruff check`
Expected: `0 errors, 0 warnings, 0 notes` and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/project/model/layers.py tests/model/test_layers.py
git commit -m "$(cat <<'EOF'
Add ExoMiner++ conv stack, branch head and weight init

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Branches and the ExoMiner++ network

**Files:**
- Create: `src/project/model/branches.py`
- Create: `src/project/model/exominer.py`
- Modify: `src/project/model/__init__.py` (replace with the final version)
- Create: `tests/helpers.py`
- Test: `tests/model/test_branches.py`, `tests/model/test_exominer.py`

**Interfaces:**
- Consumes: Task 1 constants; `ConvBranchConfig`, `ScalarBranchConfig`, `TransitViewsConfig`, `UnfoldedFluxConfig`, `pooled_length` and `required_features` (Task 2); `BranchHead`, `ConvStack`, `FullWidthLinear`, `init_weights`, `make_activation` and `with_scalars` (Task 3).
- Produces, in `project.model.branches`:
  - `ScalarBranch`, `TimeSeriesBranch`, `TransitViewBranches`, `UnfoldedFluxBranch` and `DifferenceImageBranch`, each an `nn.Module` with `out_features: int` and `forward(inputs: Mapping[str, Tensor]) -> Tensor` of shape `(B, out_features)`;
  - the union alias `Branch` of these classes.
  - Head attribute names: single-head branches expose `head: BranchHead`; `TransitViewBranches` exposes `heads: nn.ModuleDict` keyed `flux`, `weak_secondary`, `odd_even`; `DifferenceImageBranch` also exposes `sectors_layer: FullWidthLinear`.
- Produces, in `project.model.exominer`: `ExoMinerPlusPlus(config)`, with attributes `config`, `required_features: frozenset[str]`, `branches: nn.ModuleDict` and `head: nn.Sequential`. `forward(inputs: Mapping[str, Tensor]) -> Tensor` returns shape `(B,)`, and raises `KeyError` for missing and `ValueError` for misshaped features.
- Produces, in `tests.helpers`:
  - `random_inputs(config: ExoMinerConfig, batch_size: int) -> dict[str, Tensor]`;
  - `random_batch(config: ExoMinerConfig, batch_size: int) -> Batch`.

- [ ] **Step 1: Write the test helpers**

Create `tests/helpers.py`:

```python
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
```

- [ ] **Step 2: Write the tests**

Create `tests/model/test_branches.py`:

```python
import pytest
import torch

from project.features import (
    EVEN_VIEWS,
    ODD_VIEWS,
    UNFOLDED_FLUX_VIEW,
)
from project.model import DIAGRAM, PAPER, REPO_V1, ExoMinerConfig, ExoMinerPlusPlus
from project.model.branches import (
    DifferenceImageBranch,
    ScalarBranch,
    TimeSeriesBranch,
    TransitViewBranches,
    UnfoldedFluxBranch,
)
from project.model.layers import BranchHead, with_scalars
from tests.helpers import random_inputs

# Input width of every branch head per preset; derived in spec section 5.2.
HEAD_INPUTS = {
    "PAPER": {
        "global_flux": 4593,
        "flux_trend": 4594,
        "flux_periodogram": 10562,
        "local_centroid": 404,
        "momentum_dump": 400,
        "transit_views.flux": 401,
        "transit_views.weak_secondary": 404,
        "transit_views.odd_even": 402,
        "unfolded_flux": 1602,
        "diff_img": 5,
        "stellar": 6,
        "dv_tce_fit": 9,
    },
    "DIAGRAM": {
        "global_flux": 4593,
        "flux_trend": 4722,
        "flux_periodogram": 10690,
        "local_centroid": 404,
        "momentum_dump": 400,
        "transit_views.flux": 401,
        "transit_views.weak_secondary": 404,
        "transit_views.odd_even": 402,
        "unfolded_flux": 1602,
        "diff_img": 4,
        "stellar": 6,
        "dv_tce_fit": 9,
    },
    "REPO_V1": {
        "global_flux": 4593,
        "flux_trend": 4722,
        "flux_periodogram": 10690,
        "local_centroid": 401,
        "momentum_dump": 400,
        "transit_views.flux": 201,
        "transit_views.weak_secondary": 204,
        "transit_views.odd_even": 202,
        "unfolded_flux": 1602,
        "diff_img": 6,
        "stellar": 7,
        "dv_tce_fit": 9,
    },
}
PRESETS = {"PAPER": PAPER, "DIAGRAM": DIAGRAM, "REPO_V1": REPO_V1}
SINGLE_HEAD_BRANCHES = (
    ScalarBranch,
    TimeSeriesBranch,
    UnfoldedFluxBranch,
    DifferenceImageBranch,
)


def _head_inputs(model: ExoMinerPlusPlus) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name, branch in model.branches.items():
        if isinstance(branch, TransitViewBranches):
            for diagnostic, head in branch.heads.items():
                assert isinstance(head, BranchHead)
                sizes[f"{name}.{diagnostic}"] = head.linear.in_features
        else:
            assert isinstance(branch, SINGLE_HEAD_BRANCHES)
            sizes[name] = branch.head.linear.in_features
    return sizes


@pytest.mark.parametrize("preset", PRESETS)
def test_head_input_sizes_match_spec(preset: str) -> None:
    model = ExoMinerPlusPlus(PRESETS[preset])

    assert _head_inputs(model) == HEAD_INPUTS[preset]


@pytest.mark.parametrize("preset", PRESETS)
def test_branch_outputs_have_branch_units(preset: str) -> None:
    config = PRESETS[preset]
    model = ExoMinerPlusPlus(config)
    inputs = random_inputs(config, 3)

    for name, branch in model.branches.items():
        heads = 3 if name == "transit_views" else 1
        assert branch(inputs).shape == (3, heads * config.branch_units), name


def test_difference_image_full_width_layer_sees_all_sectors() -> None:
    branch = DifferenceImageBranch(PAPER.diff_img, 4, "relu", 0.0)

    assert branch.sectors_layer.in_features == 5 * 16 * 24 * 24 + 5


def test_identical_odd_and_even_views_give_zero_difference() -> None:
    config = PAPER.transit_views
    branch = TransitViewBranches(config, 4, "relu", 0.0).eval()
    inputs = random_inputs(PAPER, 2)
    for odd, even in zip(ODD_VIEWS, EVEN_VIEWS, strict=True):
        inputs[even] = inputs[odd].clone()
    zeros = torch.zeros(2, 16, 25)
    expected = branch.heads["odd_even"](
        with_scalars(zeros, inputs, config.odd_even_scalars)
    )

    assert torch.allclose(branch(inputs)[:, 8:12], expected)


def test_unfolded_branch_ignores_transit_order() -> None:
    branch = UnfoldedFluxBranch(PAPER.unfolded_flux, 4, "relu", 0.0).eval()
    inputs = random_inputs(PAPER, 2)
    shuffled = dict(inputs)
    shuffled[UNFOLDED_FLUX_VIEW] = inputs[UNFOLDED_FLUX_VIEW][:, torch.randperm(20)]

    assert torch.allclose(branch(inputs), branch(shuffled), atol=1e-6)


def test_repo_unfolded_branch_mixes_transits() -> None:
    config: ExoMinerConfig = REPO_V1
    branch = UnfoldedFluxBranch(config.unfolded_flux, 3, "relu", 0.0).eval()
    inputs = random_inputs(config, 2)
    shuffled = dict(inputs)
    shuffled[UNFOLDED_FLUX_VIEW] = inputs[UNFOLDED_FLUX_VIEW][:, torch.randperm(20)]

    assert not torch.allclose(branch(inputs), branch(shuffled), atol=1e-6)
```

Create `tests/model/test_exominer.py`:

```python
import pytest
import torch

from project.model import (
    DIAGRAM,
    PAPER,
    REPO_V1,
    ConfigValue,
    ExoMinerConfig,
    ExoMinerPlusPlus,
    with_overrides,
)
from tests.helpers import random_inputs

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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/model/test_branches.py tests/model/test_exominer.py -v`
Expected: collection errors, `ImportError: cannot import name 'ExoMinerPlusPlus' from 'project.model'`.

- [ ] **Step 4: Write `src/project/model/branches.py`**

```python
"""The ExoMiner++ branches; each maps the inputs to a `(B, out_features)` tensor.

All convolutions are 2D. Time series are fed as `(B, channels, 1, bins)` with kernels
of height 1, so rows never mix. That lets the transit views stack as rows and share
one convolution, which is exactly the paper's shared feature extractor.
"""

from collections.abc import Mapping
from typing import override

import torch
from torch import Tensor, nn

from project.features import (
    DIFF_IMG_VIEWS,
    EVEN_VIEWS,
    IMAGE_SIZE,
    LOCAL_FLUX_VIEWS,
    ODD_VIEWS,
    QUALITY,
    SECTORS,
    TRANSIT_BINS,
    UNFOLDED_FLUX_VIEW,
    WEAK_SECONDARY_VIEWS,
)
from project.model.config import (
    Activation,
    ConvBranchConfig,
    ScalarBranchConfig,
    TransitViewsConfig,
    UnfoldedFluxConfig,
    pooled_length,
)
from project.model.layers import (
    BranchHead,
    ConvStack,
    FullWidthLinear,
    make_activation,
    with_scalars,
)

__all__ = [
    "Branch",
    "DifferenceImageBranch",
    "ScalarBranch",
    "TimeSeriesBranch",
    "TransitViewBranches",
    "UnfoldedFluxBranch",
]


class ScalarBranch(nn.Module):
    """Stellar parameters or DV fit statistics through one FC layer (Sec. 3.1, 8)."""

    def __init__(
            self,
            config: ScalarBranchConfig,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.scalars: tuple[str, ...] = config.scalars
        in_features = len(config.scalars)
        self.head: BranchHead = BranchHead(in_features, units, activation, dropout)
        self.out_features: int = units

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        return self.head(torch.stack([inputs[name] for name in self.scalars], dim=1))


class TimeSeriesBranch(nn.Module):
    """A phase-folded view plus its uncertainty channel through a conv stack.

    Used for global flux, flux trend, periodogram, centroid and momentum dump.
    """

    def __init__(
            self,
            config: ConvBranchConfig,
            views: tuple[str, ...],
            bins: int,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.views: tuple[str, ...] = views
        self.scalars: tuple[str, ...] = config.scalars
        self.convs: ConvStack = ConvStack(
            in_channels=len(views),
            blocks=config.blocks,
            layers_per_block=config.layers_per_block,
            init_filters=config.init_filters,
            kernel_size=(1, config.kernel_size),
            pool_size=(1, config.pool_size),
            activation=activation,
        )
        length = pooled_length(bins, config.blocks, config.pool_size)
        in_features = self.convs.out_channels * length + len(config.scalars)
        self.head: BranchHead = BranchHead(in_features, units, activation, dropout)
        self.out_features: int = units

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        series = torch.stack([inputs[name] for name in self.views], dim=1)
        maps = self.convs(series.unsqueeze(2))
        return self.head(with_scalars(maps, inputs, self.scalars))


class TransitViewBranches(nn.Module):
    """Flux, weak secondary and odd/even transit views through one shared extractor.

    The views are stacked as rows and convolved together (Sec. 3.1 item 2), then each
    diagnostic gets its own head. Odd and even feature maps are subtracted first.
    """

    def __init__(
            self,
            config: TransitViewsConfig,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.config: TransitViewsConfig = config
        self.rows: dict[str, tuple[str, ...]] = {}
        if config.flux_enabled:
            self.rows["flux"] = LOCAL_FLUX_VIEWS
        if config.weak_secondary_enabled:
            self.rows["weak_secondary"] = WEAK_SECONDARY_VIEWS
        if config.odd_even_enabled:
            self.rows["odd"] = ODD_VIEWS
            self.rows["even"] = EVEN_VIEWS
        self.convs: ConvStack = ConvStack(
            in_channels=2,
            blocks=config.blocks,
            layers_per_block=config.layers_per_block,
            init_filters=config.init_filters,
            kernel_size=(1, config.kernel_size),
            pool_size=(1, config.pool_size),
            activation=activation,
        )
        length = pooled_length(TRANSIT_BINS, config.blocks, config.pool_size)
        per_view = self.convs.out_channels * length
        self.heads: nn.ModuleDict = nn.ModuleDict()
        for name, enabled, scalars in self._diagnostics():
            if enabled:
                self.heads[name] = BranchHead(
                    per_view + len(scalars), units, activation, dropout
                )
        self.out_features: int = units * len(self.heads)

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        rows = [
            torch.stack([inputs[name] for name in views], dim=1)
            for views in self.rows.values()
        ]
        maps = self.convs(torch.stack(rows, dim=2))
        by_row = dict(zip(self.rows, maps.unbind(dim=2), strict=True))
        if self.config.odd_even_enabled:
            by_row["odd_even"] = by_row["odd"] - by_row["even"]
        outputs = [
            self.heads[name](with_scalars(by_row[name], inputs, scalars))
            for name, enabled, scalars in self._diagnostics()
            if enabled
        ]
        return torch.cat(outputs, dim=1)

    def _diagnostics(self) -> list[tuple[str, bool, tuple[str, ...]]]:
        config = self.config
        return [
            ("flux", config.flux_enabled, config.flux_scalars),
            (
                "weak_secondary",
                config.weak_secondary_enabled,
                config.weak_secondary_scalars,
            ),
            ("odd_even", config.odd_even_enabled, config.odd_even_scalars),
        ]


class UnfoldedFluxBranch(nn.Module):
    """The 20 individual transits, reduced to their min, max and mean (Sec. 4.2).

    A 1x1 convolution then mixes the three statistics per feature position.
    """

    def __init__(
            self,
            config: UnfoldedFluxConfig,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.scalars: tuple[str, ...] = config.scalars
        self.convs: ConvStack = ConvStack(
            in_channels=1,
            blocks=config.blocks,
            layers_per_block=config.layers_per_block,
            init_filters=config.init_filters,
            kernel_size=(config.transit_kernel_size, config.kernel_size),
            pool_size=(1, config.pool_size),
            activation=activation,
        )
        self.stat_conv: nn.Conv2d = nn.Conv2d(3, config.stat_filters, kernel_size=1)
        self.stat_activation: nn.Module = (
            make_activation(activation) if config.stat_activation else nn.Identity()
        )
        length = pooled_length(TRANSIT_BINS, config.blocks, config.pool_size)
        stat_features = config.stat_filters * self.convs.out_channels * length
        in_features = stat_features + len(config.scalars)
        self.head: BranchHead = BranchHead(in_features, units, activation, dropout)
        self.out_features: int = units

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        maps = self.convs(inputs[UNFOLDED_FLUX_VIEW].unsqueeze(1))
        stats = torch.stack([maps.amin(dim=2), maps.amax(dim=2), maps.mean(dim=2)], 1)
        mixed = self.stat_activation(self.stat_conv(stats))
        return self.head(with_scalars(mixed, inputs, self.scalars))


class DifferenceImageBranch(nn.Module):
    """Difference, out-of-transit and target images of 5 sectors (Sec. 4.2).

    Each sector is convolved on its own with shared weights. The feature maps of all
    sectors and their quality metrics then go through one full-width layer, as in
    NASA's code, before the scalars are appended.
    """

    def __init__(
            self,
            config: ConvBranchConfig,
            units: int,
            activation: Activation,
            dropout: float,
    ) -> None:
        super().__init__()
        self.scalars: tuple[str, ...] = config.scalars
        self.convs: ConvStack = ConvStack(
            in_channels=len(DIFF_IMG_VIEWS),
            blocks=config.blocks,
            layers_per_block=config.layers_per_block,
            init_filters=config.init_filters,
            kernel_size=(config.kernel_size, config.kernel_size),
            pool_size=(config.pool_size, config.pool_size),
            activation=activation,
            pool_every_layer=True,
        )
        pools = config.blocks * config.layers_per_block
        size = pooled_length(IMAGE_SIZE, pools, config.pool_size)
        sector_features = SECTORS * self.convs.out_channels * size * size + SECTORS
        self.sectors_layer: FullWidthLinear = FullWidthLinear(sector_features, units)
        self.sectors_activation: nn.Module = make_activation(activation)
        in_features = units + len(config.scalars)
        self.head: BranchHead = BranchHead(in_features, units, activation, dropout)
        self.out_features: int = units

    @override
    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        images = torch.stack([inputs[name] for name in DIFF_IMG_VIEWS], dim=2)
        maps = self.convs(images.flatten(0, 1))
        sectors = torch.cat([maps.reshape(len(images), -1), inputs[QUALITY]], dim=1)
        summary = self.sectors_activation(self.sectors_layer(sectors))
        return self.head(with_scalars(summary, inputs, self.scalars))


Branch = (
    ScalarBranch
    | TimeSeriesBranch
    | TransitViewBranches
    | UnfoldedFluxBranch
    | DifferenceImageBranch
)
```

- [ ] **Step 5: Write `src/project/model/exominer.py`**

```python
"""The ExoMiner++ network (Valizadegan et al. 2025)."""

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

__all__ = ["ExoMinerPlusPlus"]


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
```

- [ ] **Step 6: Replace `src/project/model/__init__.py` with the final version**

```python
"""The ExoMiner++ model, its configuration and presets."""

from project.model.config import (
    ConfigValue,
    ExoMinerConfig,
    OptimizerConfig,
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
    "required_features",
    "to_flat",
    "with_overrides",
]
```

- [ ] **Step 7: Run tests and checks**

Run: `uv run pytest tests/model -v`
Expected: `70 passed` (this includes the config and layer tests).

Run: `uv run basedpyright && uv run ruff check`
Expected: `0 errors, 0 warnings, 0 notes` and `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add src/project/model tests/helpers.py tests/model
git commit -m "$(cat <<'EOF'
Add ExoMiner++ branches and network

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Paper metrics

**Files:**
- Create: `src/project/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing from the project (torchmetrics only).
- Produces, in `project.metrics`: `paper_metrics(threshold: float = 0.5) -> MetricCollection`, with keys `accuracy`, `precision`, `recall`, `auc_pr` and `auc_roc`. It expects probabilities and integer targets.

- [ ] **Step 1: Write the tests**

Create `tests/test_metrics.py`:

```python
import subprocess
import sys

import pytest
import torch

from project.metrics import paper_metrics


def test_paper_metrics_on_hand_computed_example() -> None:
    metrics = paper_metrics()
    probabilities = torch.tensor([0.9, 0.8, 0.3, 0.6, 0.1])
    targets = torch.tensor([1, 1, 1, 0, 0])

    result = metrics(probabilities, targets)

    assert result["accuracy"].item() == pytest.approx(3 / 5)
    assert result["precision"].item() == pytest.approx(2 / 3)
    assert result["recall"].item() == pytest.approx(2 / 3)
    assert result["auc_roc"].item() == pytest.approx(5 / 6)
    assert result["auc_pr"].item() == pytest.approx((1 + 1 + 3 / 4) / 3)


def test_shared_modules_do_not_import_the_model() -> None:
    code = (
        "import sys, project.features, project.metrics; "
        "assert 'project.model' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: a collection error, `ModuleNotFoundError: No module named 'project.metrics'`.

- [ ] **Step 3: Write `src/project/metrics.py`**

```python
"""Classification metrics reported in the ExoMiner++ paper (Sec. 6.1)."""

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)

__all__ = ["paper_metrics"]


def paper_metrics(threshold: float = 0.5) -> MetricCollection:
    """Accuracy, precision and recall at `threshold`, plus PR AUC and ROC AUC.

    Expects predicted probabilities and integer targets (1 = planet).
    """
    return MetricCollection(
        {
            "accuracy": BinaryAccuracy(threshold=threshold),
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "auc_pr": BinaryAveragePrecision(),
            "auc_roc": BinaryAUROC(),
        }
    )
```

- [ ] **Step 4: Run tests and checks**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: `2 passed`. The second test enforces the import rules from Global Constraints.

Run: `uv run basedpyright && uv run ruff check`
Expected: `0 errors, 0 warnings, 0 notes` and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/project/metrics.py tests/test_metrics.py
git commit -m "$(cat <<'EOF'
Add ExoMiner++ paper metrics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Lightning classifier

**Files:**
- Create: `src/project/classifier.py`
- Test: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `Batch` and `UnlabeledBatch` (Task 1); `ExoMinerPlusPlus`, `ExoMinerConfig`, `OptimizerConfig`, `PAPER`, `PAPER_OPTIMIZER` and the branch config classes (Tasks 2 and 4); `paper_metrics` (Task 5); `random_batch` (Task 4 helpers).
- Produces, in `project.classifier`: `ExoMinerClassifier(config: ExoMinerConfig = PAPER, optimizer: OptimizerConfig = PAPER_OPTIMIZER)`.
  - Attributes: `model: ExoMinerPlusPlus`, `optimizer_config: OptimizerConfig`, and `train_metrics`, `val_metrics`, `test_metrics`.
  - It logs `{train,val,test}/loss` and `{train,val,test}/{accuracy,precision,recall,auc_pr,auc_roc}`.
  - `predict_step` returns probabilities of shape `(B,)`.
  - Importing the module registers the config classes with `torch.serialization.add_safe_globals`.

- [ ] **Step 1: Write the tests**

Create `tests/test_classifier.py`:

```python
from pathlib import Path
from typing import override

import lightning
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from project.classifier import ExoMinerClassifier
from project.features import Batch, UnlabeledBatch
from project.model import DIAGRAM, PAPER, REPO_V1_OPTIMIZER
from tests.helpers import random_batch


class _RandomBatches(Dataset[Batch]):
    """Pre-collated random batches, served one per index."""

    def __init__(self, batches: int, batch_size: int) -> None:
        self.batches: list[Batch] = [
            random_batch(PAPER, batch_size) for _ in range(batches)
        ]

    def __len__(self) -> int:
        return len(self.batches)

    @override
    def __getitem__(self, index: int) -> Batch:
        return self.batches[index]


def _loader(batches: int = 2, batch_size: int = 4) -> DataLoader[Batch]:
    return DataLoader(_RandomBatches(batches, batch_size), batch_size=None)


def _trainer(
        fast_dev_run: bool = False,
        max_steps: int = -1,
        default_root_dir: Path | None = None,
) -> lightning.Trainer:
    return lightning.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        fast_dev_run=fast_dev_run,
        max_steps=max_steps,
        default_root_dir=default_root_dir,
    )


def test_fast_dev_run_fits_validates_and_predicts() -> None:
    classifier = ExoMinerClassifier()
    trainer = _trainer(fast_dev_run=True)

    trainer.fit(classifier, _loader(), _loader())
    logged = set(trainer.callback_metrics)
    trainer.validate(classifier, _loader())
    predictions = trainer.predict(classifier, _loader())

    assert {"train/loss", "train/auc_pr", "val/loss", "val/auc_pr"} <= logged
    assert predictions is not None
    for probabilities in predictions:
        assert isinstance(probabilities, torch.Tensor)
        assert probabilities.shape == (4,)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_predict_without_labels() -> None:
    classifier = ExoMinerClassifier().eval()
    batch: UnlabeledBatch = {"features": random_batch(PAPER, 3)["features"]}

    probabilities = classifier.predict_step(batch, 0)

    assert probabilities.shape == (3,)


def test_validation_with_a_single_class() -> None:
    classifier = ExoMinerClassifier()
    dataset = _RandomBatches(batches=1, batch_size=4)
    for batch in dataset.batches:
        batch["label"] = torch.zeros(4)
    trainer = _trainer()

    with pytest.warns(UserWarning):
        results = trainer.validate(classifier, DataLoader(dataset, batch_size=None))

    assert "val/loss" in results[0]


def test_overfits_a_single_batch() -> None:
    classifier = ExoMinerClassifier()
    batch = random_batch(PAPER, 8)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    losses: list[float] = []

    for _ in range(30):
        optimizer.zero_grad()
        loss = loss_fn(classifier(batch["features"]), batch["label"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < 0.5 * losses[0]


def test_checkpoint_restores_config(tmp_path: Path) -> None:
    classifier = ExoMinerClassifier(DIAGRAM, REPO_V1_OPTIMIZER)
    trainer = _trainer(max_steps=1, default_root_dir=tmp_path)
    trainer.fit(classifier, _loader(batches=1), _loader(batches=1))
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path)

    restored = ExoMinerClassifier.load_from_checkpoint(path)

    assert restored.model.config == DIAGRAM
    assert restored.optimizer_config == REPO_V1_OPTIMIZER
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: a collection error, `ModuleNotFoundError: No module named 'project.classifier'`.

- [ ] **Step 3: Write `src/project/classifier.py`**

```python
"""Lightning wrapper that trains ExoMiner++ as a binary planet classifier."""

from collections.abc import Mapping
from typing import override

import lightning
import torch
from torch import Tensor
from torch.nn import functional
from torchmetrics import MetricCollection

from project.features import Batch, UnlabeledBatch
from project.metrics import paper_metrics
from project.model import (
    PAPER,
    PAPER_OPTIMIZER,
    ExoMinerConfig,
    ExoMinerPlusPlus,
    OptimizerConfig,
)
from project.model.config import (
    ConvBranchConfig,
    ScalarBranchConfig,
    TransitViewsConfig,
    UnfoldedFluxConfig,
)

__all__ = ["ExoMinerClassifier"]

# Checkpoints hold the configs; torch.load only unpickles allowlisted classes.
torch.serialization.add_safe_globals(
    [
        ExoMinerConfig,
        OptimizerConfig,
        ConvBranchConfig,
        ScalarBranchConfig,
        TransitViewsConfig,
        UnfoldedFluxConfig,
    ]
)


class ExoMinerClassifier(lightning.LightningModule):
    """ExoMiner++ trained with binary cross-entropy and Adam (paper Sec. 5).

    Logs `{stage}/loss` and the paper metrics as `{stage}/<metric>` for the stages
    train, val and test.
    """

    def __init__(
            self,
            config: ExoMinerConfig = PAPER,
            optimizer: OptimizerConfig = PAPER_OPTIMIZER,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_config: OptimizerConfig = optimizer
        self.model: ExoMinerPlusPlus = ExoMinerPlusPlus(config)
        metrics = paper_metrics()
        self.train_metrics: MetricCollection = metrics.clone(prefix="train/")
        self.val_metrics: MetricCollection = metrics.clone(prefix="val/")
        self.test_metrics: MetricCollection = metrics.clone(prefix="test/")

    @override
    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        return self.model(features)

    @override
    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        return self._shared_step(batch, self.train_metrics, "train")

    @override
    def validation_step(self, batch: Batch, batch_idx: int) -> Tensor:
        return self._shared_step(batch, self.val_metrics, "val")

    @override
    def test_step(self, batch: Batch, batch_idx: int) -> Tensor:
        return self._shared_step(batch, self.test_metrics, "test")

    @override
    def predict_step(self, batch: UnlabeledBatch, batch_idx: int) -> Tensor:
        return torch.sigmoid(self(batch["features"]))

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        config = self.optimizer_config
        return torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

    def _shared_step(
            self,
            batch: Batch,
            metrics: MetricCollection,
            stage: str,
    ) -> Tensor:
        logits = self(batch["features"])
        labels = batch["label"]
        loss = functional.binary_cross_entropy_with_logits(logits, labels)
        metrics.update(torch.sigmoid(logits), labels.long())
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=len(labels),
        )
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss
```

- [ ] **Step 4: Run tests and checks**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: `5 passed`.

Run: `uv run basedpyright && uv run ruff check`
Expected: `0 errors, 0 warnings, 0 notes` and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/project/classifier.py tests/test_classifier.py
git commit -m "$(cat <<'EOF'
Add ExoMiner++ Lightning classifier

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Full verification, including the GPU

**Files:** none are created or changed.

**Interfaces:**
- Consumes: everything above.
- Produces: evidence that the whole suite passes and that training runs on the local GPU.

- [ ] **Step 1: Run the whole suite and both checkers**

Run: `uv run pytest -v && uv run basedpyright && uv run ruff check`
Expected: `80 passed`, `0 errors, 0 warnings, 0 notes`, `All checks passed!`.

- [ ] **Step 2: Smoke-train every preset on the GPU**

Run:

```bash
uv run python - <<'EOF'
import lightning
import torch
from torch.utils.data import DataLoader

from project.classifier import ExoMinerClassifier
from project.model import DIAGRAM, PAPER, REPO_V1
from tests.helpers import random_batch

for config in (PAPER, DIAGRAM, REPO_V1):
    batches = [random_batch(config, 32) for _ in range(4)]
    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(
        ExoMinerClassifier(config),
        DataLoader(batches, batch_size=None),
        DataLoader(batches, batch_size=None),
    )
    print(config.activation, config.branch_units, trainer.callback_metrics["val/loss"])
print(torch.cuda.get_device_name(0))
EOF
```

Expected, after Lightning's log output:
- three lines like `relu 4 tensor(0.69, device='cuda:0')`, `prelu 3 tensor(...)`, `prelu 3 tensor(...)`, each with a finite loss near 0.69;
- then the GPU name (e.g. `AMD Radeon RX 9070 XT`);
- no exceptions.

A CPU dry run of this script printed losses 0.6918, 0.6943 and 0.6851.

- [ ] **Step 3: Confirm the placeholders are untouched**

Run: `git diff main --stat -- src/project/__init__.py src/project/abc.py scripts/test.py`
Expected: no output.

- [ ] **Step 4: Hand over**

Report the test count, the checker output, the GPU smoke-run losses, and the basedpyright rules disabled in `pyproject.toml`:
- `reportAny`, `reportUnknownMemberType`, `reportUnknownVariableType`, `reportUnknownArgumentType`, because torch/Lightning are typed as `Any` or left untyped;
- `reportUnusedCallResult`, because of in-place torch APIs.
