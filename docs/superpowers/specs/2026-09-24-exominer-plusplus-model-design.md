# ExoMiner++ Model: Design

Date: 2026-09-24 (revised 2026-09-25: configurable architecture and presets)
Status: awaiting review

## 1. Goal

Reimplement the ExoMiner++ transit classifier (Valizadegan et al. 2025, AJ 170(5), arXiv:2502.09790) in PyTorch Lightning.
Write it from scratch in `src/project/`, easy to read, fully typed, and compliant with `style.md`.

The architecture is **configurable**. Presets reproduce three reference models:

- `PAPER`: the paper text;
- `DIAGRAM`: the Zenodo architecture image of the trained paper model;
- `REPO_V1`: NASA's ExoMiner v1.0.0 code and YAML.

The config can also be driven by an Optuna search.

In scope:

- config dataclasses, presets, and a flat key/value view for Optuna and loggers;
- the `nn.Module` for the model, built from small branch modules;
- a `LightningModule` that defines the loss, the optimizer and the paper's metrics;
- tests for shapes, presets, config validation, and a smoke test that the model trains on random tensors;
- a `pyproject.toml` that installs torch for either ROCm or CUDA.

Out of scope (later work):

- the dataset / `LightningDataModule` and normalization;
- training, evaluation and Optuna scripts (no `optuna` dependency yet);
- cross-validation, ensembling, TensorBoard setup, multi-sector aggregation, per-subclass recall, precision@k.

The old attempt in `../BSc-Project` is not ported and not imported.

Success criteria:

1. With each preset, every branch produces the tensor shapes listed in section 5.
2. A full forward and backward pass works on random inputs for batch sizes 1 and 8.
3. Lightning's `fast_dev_run` completes on random data.
4. A config with impossible shapes cannot be constructed: it raises `ValueError` naming the problem.
5. `basedpyright` and `ruff check` report no errors or warnings. `ruff format` is not used: it would undo the extra indentation `style.md` requires for `def` arguments.

## 2. Sources

Three references, and how they disagree:

1. **Paper text**: Sec. 3.1, Sec. 4.2, Sec. 5, App. A and Table 18.
2. **Zenodo architecture diagram**: `exominer_plusplus_architecture.png`, transcribed in `~/Downloads/exominer_plusplus_torchview.gv`.
   It shows the model trained for the paper. It shows shapes, but not kernel sizes or initialization.
3. **NASA ExoMiner v1.0.0**: `models/models_keras.py`, `models/exominer_plus_plus.yaml` and `src_cv/train/config_cv_train.yaml`.
   The paper-era container images (commit `fec6ab1`) use the same ExoMiner++ YAML values.

`PAPER` is the default preset. Where the paper text is silent, `PAPER` falls back to the diagram and then to the code, which gives:

- the scalars fed to each branch;
- the min/max/mean step in the unfolded branch;
- the difference-image quality input and full-width layer;
- the stat conv kernel size of 1;
- Xavier init for linear layers;
- Adam's β and ε.

`DIAGRAM` falls back to the paper text for what the image cannot show (kernel sizes, initialization).

One v1.0.0 bug is never replicated. NASA reshapes images stored as `(5, 33, 33)` into `(33, 33, 5)` with `tf.reshape` instead of a transpose, which scrambles the pixels.

## 3. Input contract

`ExoMinerPlusPlus.forward` takes a `Mapping[str, Tensor]` keyed by NASA's feature names and returns logits of shape `(B,)`.
Inputs must already be normalized; that is the future data module's job.
`forward` raises `KeyError` naming missing features, and `ValueError` naming any feature whose per-example shape differs from `FEATURE_SHAPES` (e.g. NASA-style `(B, 1)` scalars).

Per-sample shapes (the batch adds a leading `B`):

- phase-folded views: `(L,)`, where L is 31 for transit views, 301 for full-orbit views, 674 for periodograms;
- unfolded flux: `(20, 31)` (transit, bin);
- images: `(5, 33, 33)` (sector, row, column);
- `quality`: `(5,)`;
- scalars: `()`, which batch to `(B,)`.

**Views are fixed per branch.** All three sources agree on them, and changing them changes what a branch means.

**Scalars are configurable per branch,** because the sources disagree on them (section 4.3).

Fixed views:

| Branch | Views |
|---|---|
| `global_flux` | `global_flux_view_fluxnorm`, `global_flux_view_fluxnorm_var` |
| `flux_trend` | `flux_trend_global_norm`, `flux_trend_global_norm_var` |
| `flux_periodogram` | `pgram_smooth_norm`, `pgram_tpm_smooth_norm` |
| `local_centroid` | `local_centr_view_std_noclip`, `local_centr_view_std_noclip_var` |
| `momentum_dump` | `local_momentum_dump_view`, `local_momentum_dump_view_var` |
| `transit_views`: flux | `local_flux_view_fluxnorm`, `local_flux_view_fluxnorm_var` |
| `transit_views`: weak secondary | `local_weak_secondary_view_selfnorm`, `local_weak_secondary_view_selfnorm_var` |
| `transit_views`: odd / even | `local_flux_odd_view_fluxnorm`, `local_flux_odd_view_fluxnorm_var`, `local_flux_even_view_fluxnorm`, `local_flux_even_view_fluxnorm_var` |
| `unfolded_flux` | `unfolded_local_flux_view_fluxnorm` |
| `diff_img` | `diff_imgs_std_trainset`, `oot_imgs_std_trainset`, `target_imgs`, and per-sector `quality` |
| `stellar`, `dv_tce_fit` | none |

`FEATURE_SHAPES` in `features.py` lists every known feature and its per-sample shape.
`required_features(config)` in `model/config.py` returns the subset a given config reads, so a future data module loads only what the model uses.

## 4. Configuration

### 4.1 Structure

All config classes are frozen dataclasses (`slots=True`). Their fields are only `bool`, `int`, `float`, `str` literals and `tuple[str, ...]`.

Every conv branch describes its layers the way Table 18 does:

- `blocks`: number of blocks;
- `layers_per_block`: number of conv layers in each block;
- `init_filters`: filters in the first block; block `i` has `init_filters · 2**i` filters;
- `kernel_size`;
- `pool_size`.

Parametrizing the filters this way, instead of listing them as a tuple, keeps every field a single scalar Optuna can suggest.

```python
Activation = Literal["relu", "prelu"]
ConvInit = Literal["he", "glorot"]

@dataclass(frozen=True, slots=True)
class ScalarBranchConfig:            # stellar, dv_tce_fit
    enabled: bool
    scalars: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class ConvBranchConfig:              # global_flux, flux_trend, flux_periodogram,
    enabled: bool                    # local_centroid, momentum_dump, diff_img
    scalars: tuple[str, ...]
    blocks: int
    layers_per_block: int
    init_filters: int
    kernel_size: int
    pool_size: int

@dataclass(frozen=True, slots=True)
class TransitViewsConfig:            # shared extractor for flux / secondary / odd-even
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

@dataclass(frozen=True, slots=True)
class UnfoldedFluxConfig:
    enabled: bool
    scalars: tuple[str, ...]
    blocks: int
    layers_per_block: int
    init_filters: int
    kernel_size: int                 # along the 31 bins
    transit_kernel_size: int         # along the 20 transits; 1 = each transit separately
    pool_size: int                   # along the bins only
    stat_filters: int                # 1x1 conv over the min/max/mean statistics
    stat_activation: bool

@dataclass(frozen=True, slots=True)
class ExoMinerConfig:
    activation: Activation
    conv_init: ConvInit
    branch_units: int
    branch_dropout: float
    scalar_branch_dropout: bool      # dropout after the stellar / DV FC layer
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
```

The optimizer settings live separately, in `OptimizerConfig(learning_rate, beta1, beta2, eps)`, which the `LightningModule` takes.

### 4.2 Deliberately not configurable

These are fixed, because all sources agree on them or because changing them breaks the design:

- **Conv stride 1, 'same' padding; pool stride 1, 'valid' padding.** PyTorch only supports `padding="same"` with stride 1. For an even kernel, PyTorch and TF pad the same way: 2 left and 3 right for k=6.
- **Views per branch** (section 3).
- **How branches are wired:**
  - the shared transit-view extractor;
  - the odd − even subtraction on feature maps;
  - the min/max/mean reduction in the unfolded branch;
  - pooling after every conv in the difference-image branch, with the difference image's full-width layer having `branch_units` outputs (as in v1.0.0).
- **Output:** one logit. Multi-class is out of scope.
- **PReLU shape:** a single learnable α, initialized to 0. Keras used one α per channel on some 2D/3D layers. That difference doesn't change the architecture's capacity in any meaningful way, and configuring it would add a field nobody tunes.
- **Linear init:** always Xavier-uniform with zero bias. Only the conv init differs between sources.

### 4.3 Presets

`PAPER`, `DIAGRAM` and `REPO_V1` are module-level constants in `presets.py`. `DIAGRAM` and `REPO_V1` are built from `PAPER` with `dataclasses.replace`, so each one shows only how it differs.

Settings that differ between presets:

| Setting | `PAPER` | `DIAGRAM` | `REPO_V1` |
|---|---|---|---|
| `activation` | relu | prelu | prelu |
| `conv_init` | he | he | glorot |
| `branch_units` | 4 | 3 | 3 |
| `branch_dropout` | 0.1211 | 0.1211 | 0.1211390996398814 |
| `scalar_branch_dropout` | True | False | False |
| `head_dropout` | 0.0215 | 0.0215 | 0.021491238286347598 |
| `global_flux.layers_per_block` | 3 | 3 | 5 |
| `flux_trend.pool_size` | 8 | 4 | 4 |
| `flux_periodogram.pool_size` | 8 | 4 | 4 |
| `local_centroid.kernel_size` | 6 | 6 | 5 |
| `momentum_dump.kernel_size` | 6 | 6 | 5 |
| `transit_views.layers_per_block` | 3 | 3 | 1 |
| `transit_views.init_filters` | 8 | 8 | 4 |
| `unfolded_flux.kernel_size` | 6 | 6 | 1 |
| `unfolded_flux.transit_kernel_size` | 1 | 1 | 6 |
| `unfolded_flux.stat_activation` | True | False | False |
| `local_centroid.scalars` | dikco_msky, dikco_msky_err, mag_shift, ruwe | same as `PAPER` | mag_shift |
| `stellar.scalars` | sdens, steff, smet, slogg, smass, sradius | same as `PAPER` | the 6 + ruwe |
| `diff_img.scalars` | mag_shift | mag_shift | mag_shift, dikco_msky, dikco_msky_err |

The scalar names above are shortened; the presets use the full `_norm` feature names.

Settings shared by all presets:

- **Head:** 4 layers × 512 units.
- **Global flux:** 2 blocks, 8 initial filters, kernel 5, pool 8.
- **Flux trend and periodogram:** 2 blocks, 3 layers per block, 8 initial filters, kernel 5.
- **Centroid and momentum dump:** 2 blocks, 3 layers per block, 8 initial filters, pool 4.
- **Transit views:** 2 blocks, kernel 6, pool 4.
- **Unfolded flux:** 2 blocks, 3 layers per block, 8 initial filters, pool 4, 4 stat filters.
- **Difference image:** 3 blocks, 3 layers per block, 4 initial filters, kernel 3, pool 2.
- **Every branch** is enabled.

Scalars shared by all presets:

| Branch | Scalars |
|---|---|
| global flux | `flux_global_stat_abs_min_norm` |
| flux trend | `flux_trend_global_stat_max_norm`, `flux_trend_global_stat_min_norm` |
| periodogram | `pgram_smooth_max_power_norm`, `pgram_tpm_smooth_max_power_norm` |
| momentum dump | none |
| local flux | `flux_local_stat_abs_min` |
| weak secondary | `tce_albedo_stat_norm`, `tce_ptemp_stat_norm`, `tce_maxmes_norm`, `flux_weak_secondary_local_stat_abs_min_norm` |
| odd / even | `flux_odd_local_stat_abs_min_norm`, `flux_even_local_stat_abs_min_norm` |
| unfolded flux | `tce_num_transits_obs_norm`, `tce_num_transits_norm` |
| DV fit | `boot_fap_norm`, `tce_cap_stat_norm`, `tce_hap_stat_norm`, `tce_period_norm`, `tce_max_mult_ev_norm`, `tce_max_sngle_ev_norm`, `tce_robstat_norm`, `tce_model_chisq_norm`, `tce_prad_norm` |

`OptimizerConfig` presets: `PAPER_OPTIMIZER` with lr 4.18e-5 (App. A), and `REPO_V1_OPTIMIZER` with lr 4.176171931455995e-05. Both use β = (0.9, 0.999) and ε = 1e-8. `DIAGRAM` uses the paper's optimizer.

`ExoMinerConfig` fields have no defaults. Every config comes from a preset, so no half-specified config can exist.

### 4.4 Flat view for Optuna and loggers

In `config.py`:

- `to_flat(config) -> dict[str, ConfigValue]` turns a config into dotted keys, for example `{"activation": "relu", "global_flux.pool_size": 8, "transit_views.flux_scalars": (...)}`.
- `with_overrides(config, overrides: Mapping[str, ConfigValue]) -> ExoMinerConfig` returns a copy with only the given dotted keys replaced.
  - An unknown key raises `KeyError`.
  - A value of the wrong type raises `TypeError`. Ints are accepted for floats, and lists of names for tuples, so a flat config survives a round trip through JSON.
- `ConfigValue = bool | int | float | str | tuple[str, ...]`.

A future Optuna objective then looks like this:

```python
overrides = {
    "global_flux.blocks": trial.suggest_int("global_flux.blocks", 1, 5),
    "global_flux.init_filters": 2 ** trial.suggest_int("global_flux.log2_filters", 2, 6),
    "branch_dropout": trial.suggest_float("branch_dropout", 1e-3, 0.2),
    "activation": trial.suggest_categorical("activation", ["relu", "prelu"]),
}
try:
    config = with_overrides(PAPER, overrides)
except ValueError as error:
    raise optuna.TrialPruned(str(error)) from error
```

`to_flat` also gives TensorBoard's hparams view the flat, loggable form it needs.

### 4.5 Validation

`__post_init__` rejects invalid configs with a `ValueError` that names the field:

- counts and sizes must be ≥ 1; dropouts must satisfy 0 ≤ p < 1;
- scalar names must exist in `FEATURE_SHAPES` and must have shape `()`;
- at least one branch must be enabled;
- the pooled length must stay ≥ 1 for each branch's input length, where `pooled_length(n, blocks, pool) = n − blocks · (pool − 1)`;
- the difference-image branch pools `blocks · layers_per_block` times.

Because of this, an Optuna trial with an impossible geometry fails at construction, and the objective can prune it.

## 5. Architecture

### 5.1 Shared building blocks

- **`ConvStack`**: `blocks` blocks of `layers_per_block × (Conv → activation)` followed by a max-pool.
  - It is always 2D, with `(height, width)` kernel and pool sizes. A time series enters as `(B, channels, 1, bins)` with kernels of height 1, so rows never mix.
  - Option `pool_every_layer` is used by the difference image only.
- **`BranchHead`**: concatenates the flattened features with the branch's scalars, then Linear(→`branch_units`) → activation → Dropout(`branch_dropout`).
  The stellar and DV heads skip the dropout unless `scalar_branch_dropout` is set.
- **Weight sharing:**
  - The 4 transit views are stacked as rows of one `(B, 2, 4, 31)` input. A `(1, k)` kernel convolves each row separately with the same weights: the paper's shared extractor.
  - The 5 image sectors are folded into the batch dimension around the `ConvStack`.
  - The 20 unfolded transits use kernel `(transit_kernel_size, kernel_size)`. That one layer covers both the paper's per-transit conv (`(1, 6)`) and v1.0.0's across-transit conv (`(6, 1)`).

### 5.2 Branches and shapes

Shapes below are for `PAPER` (8→16 filters, transit pool 4, full-orbit pool 8).

| Branch | Conv input | Conv output | Flattened | Head input |
|---|---|---|---|---|
| global flux | `(B, 2, 1, 301)` | `(B, 16, 1, 287)` | 4592 | 4593 |
| flux trend | `(B, 2, 1, 301)` | `(B, 16, 1, 287)` | 4592 | 4594 |
| periodogram | `(B, 2, 1, 674)` | `(B, 16, 1, 660)` | 10560 | 10562 |
| centroid | `(B, 2, 1, 31)` | `(B, 16, 1, 25)` | 400 | 404 |
| momentum dump | `(B, 2, 1, 31)` | `(B, 16, 1, 25)` | 400 | 400 |
| transit views (shared) | `(B, 2, 4, 31)` | `(B, 16, 4, 25)` | 400 per view | flux 401, secondary 404, odd−even 402 |
| unfolded flux | `(B, 1, 20, 31)` | `(B, 16, 20, 25)` | 1600 after reduction | 1602 |
| difference image | `(B·5, 3, 33, 33)` | `(B·5, 16, 24, 24)` | 46080 + 5 quality → `branch_units` | 5 |
| stellar | — | — | — | 6 |
| DV fit | — | — | — | 9 |

`DIAGRAM` reproduces the diagram's flatten sizes: flux trend 4720, periodogram 10688, all others the same.
`REPO_V1` gives:

- transit views 25 × 8 = 200 per view;
- centroid and momentum dump 400;
- unfolded 1600;
- head input: 3 per branch × 12 = 36.

Tests pin these numbers.

Branch-specific steps:

- **Transit views:**
  1. The enabled views are stacked (flux, secondary, odd, even) and passed through one `ConvStack`.
  2. The output is split: flux → head; secondary → head; odd feature map − even feature map → head.
- **Unfolded flux:**
  1. `ConvStack` over `(B, 1, 20, 31)`, with pooling `(1, pool_size)` along the bins only.
  2. Min, max and mean over the transit axis, stacked as channels: `(B, 3, C, L)`.
  3. Conv2d(3 → `stat_filters`, kernel 1), followed by the activation if `stat_activation` is set.
  4. Flatten, then the head.
- **Difference image:**
  1. Stack diff, oot and target as 3 channels per sector.
  2. `ConvStack` with `pool_every_layer`.
  3. Unfold the sectors, flatten, and concatenate `quality`.
  4. Linear(→`branch_units`) → activation. This is equivalent to NASA's full-width Conv1D and is initialized as a conv.
  5. Concatenate the scalars, then the head.
- **Stellar and DV fit:** the head applied to the stacked scalars.

A disabled branch is not built. The size of the classifier head's input follows from the enabled branches.

### 5.3 Classifier head and initialization

- **Head:**
  1. Concatenate the branch outputs.
  2. `head_layers` × [Linear(`head_units`) → activation → Dropout(`head_dropout`)].
  3. Linear(→1), squeezed to `(B,)` logits.
  The sigmoid is applied only by the loss and by prediction.
- **Initialization:**
  - convolutions, including the difference image's full-width layer: `kaiming_normal_` if `conv_init == "he"`, otherwise `xavier_uniform_`;
  - linear layers: `xavier_uniform_`;
  - biases: zero;
  - PReLU α: 0.

## 6. Code layout

The layout is chosen so that the data and metrics code can be added later without moving anything. Files marked *(later)* are not part of this work.

```
src/project/
    features.py         data <-> model contract: FEATURE_SHAPES, fixed views per branch,
                        Batch, UnlabeledBatch
    metrics.py          paper_metrics() -> MetricCollection
    classifier.py       ExoMinerClassifier (LightningModule)
    model/
        __init__.py     re-exports ExoMinerPlusPlus, ExoMinerConfig, presets, with_overrides
        config.py       config dataclasses, validation, to_flat(), with_overrides(),
                        required_features(), pooled_length()
        presets.py      PAPER, DIAGRAM, REPO_V1, PAPER_OPTIMIZER, REPO_V1_OPTIMIZER
        layers.py       ConvStack, BranchHead, FullWidthLinear, make_activation(),
                        with_scalars(), init_weights()
        branches.py     ScalarBranch, TimeSeriesBranch, TransitViewBranches,
                        UnfoldedFluxBranch, DifferenceImageBranch
        exominer.py     ExoMinerPlusPlus
    data/               (later) dataset, datamodule, normalization, CV splits
scripts/                (later) train, evaluate, optuna search
tests/
    conftest.py         seeds torch before every test
    helpers.py          random_inputs() / random_batch() built from required_features()
    test_features.py
    model/
        test_config.py
        test_layers.py
        test_branches.py
        test_exominer.py
    test_metrics.py
    test_classifier.py
```

Dependency rules. These are what keep future additions local:

| Module | May import |
|---|---|
| `features` | torch only |
| `model` | `features` |
| `metrics` | torchmetrics only |
| `data` *(later)* | `features` |
| `classifier` | `model`, `metrics`, `features` |
| `scripts` *(later)* | everything |

- **`data` never imports `model`.** A script passes `required_features(config)` (a `frozenset[str]`) to the data module, which then reads only the columns the model uses. Normalization, splits and loading can change without touching the model, and the model can be tested without data.
- **`features.py` is the contract both sides share:**
  - It defines `Batch`, a `TypedDict` with `features: dict[str, Tensor]` and `label: Tensor` (float, `(B,)`, 1 = KP/CP).
  - Later, the data module can add `NotRequired` fields such as `uid` or `disposition`, for predictions and per-subclass recall. The model is unaffected.
- **`metrics.py` holds the paper's metrics from the start.** The classifier doesn't define its own.
  - `paper_metrics()` returns a torchmetrics `MetricCollection`: accuracy, precision and recall at 0.5, `auc_pr` (`BinaryAveragePrecision`), `auc_roc` (`BinaryAUROC`).
  - Precision@k and per-disposition recall go into the same module later, where evaluation scripts can reuse them without Lightning.

Module details:

- **`model/exominer.py`:** `ExoMinerPlusPlus(config)` builds the enabled branches in an `nn.ModuleDict`, plus the classifier head. `forward` checks its inputs as described in section 3.
- **`classifier.py`:** `ExoMinerClassifier(config: ExoMinerConfig = PAPER, optimizer: OptimizerConfig = PAPER_OPTIMIZER)`.
  - It saves the configs as hyperparameters in flat form (`to_flat`, with tuples written as lists, plus `asdict(optimizer)`). Checkpoints and `hparams.yaml` then hold plain values keyed by field name, which `torch.load`'s `weights_only` default accepts. `__init__` also accepts the flat form and rebuilds it with `from_flat`, which raises `KeyError` for a saved config that predates a new field instead of silently shifting values.
  - `predict_step` takes an `UnlabeledBatch`, so unlabeled TCEs can be scored.
  - The loss is `binary_cross_entropy_with_logits`, unweighted (v1.0.0 uses no class weights). The optimizer is Adam.
  - Each stage logs `{stage}/loss` and its own clone of `paper_metrics()` as `{stage}/<metric>`, per epoch.
  - `val/auc_pr` is what a future `EarlyStopping` callback (patience 20, mode max) will monitor.
  - `predict_step` returns sigmoid probabilities.

Each module declares `__all__`. Comments are single-line and only where something is not obvious. Docstrings explain what each branch does and where it comes from in the paper.

## 7. Tests

- **`test_config.py`:**
  - `to_flat` → `with_overrides` round-trips each preset;
  - an unknown key raises `KeyError`, a wrongly typed value raises `TypeError`;
  - bad geometry (e.g. 5 blocks with pool 16 on the transit views) raises `ValueError`;
  - an unknown scalar name raises `ValueError`, and so does disabling every branch;
  - `DIAGRAM` and `REPO_V1` differ from `PAPER` in exactly the fields of the section 4.3 table.
- **`test_layers.py`:**
  - `pooled_length`;
  - `ConvStack` shapes in 1D and 2D, with int and pair kernels;
  - 'same' padding with an even kernel.
- **`test_branches.py`:**
  - with each preset, the branch output width and each head's `in_features` match section 5.2;
  - identical odd and even inputs give an odd−even feature map of zeros;
  - with `transit_kernel_size = 1`, permuting the 20 transits leaves the unfolded output unchanged (eval mode).
- **`test_exominer.py`**, for each preset:
  - forward returns `(B,)` for B = 1 and 8;
  - every parameter gets a gradient;
  - eval mode is deterministic;
  - a missing feature raises `KeyError`;
  - disabling a branch removes its features from `required_features` and shrinks the head input.
- **`test_metrics.py`:**
  - on a small hand-computed example, `paper_metrics()` returns the expected accuracy, precision, recall and ROC-AUC.
- **Dependency rules:** a test imports `project.features` and `project.metrics` in a fresh interpreter and asserts that `project.model` was not imported. This enforces the section 6 table.
- **`test_classifier.py`:**
  - `Trainer(fast_dev_run=True, accelerator="cpu")` runs fit, validate and predict;
  - 30 steps on one fixed batch of 8 reduce the loss;
  - a checkpoint round trip restores the same config.

## 8. Project setup

`pyproject.toml` changes. The existing file is edited; its name and version stay the same.

```toml
[project]
requires-python = ">=3.12"
dependencies = ["lightning>=2.6", "torchmetrics>=1.9"]

[dependency-groups]
dev = ["ai-edge-model-explorer>=0.1.33", "basedpyright>=1.40", "pytest>=9.1", "ruff>=0.16"]
rocm = ["torch>=2.14", "triton-rocm"]
cuda = ["torch>=2.14"]

[tool.uv]
default-groups = ["dev", "rocm"]
environments = ["sys_platform == 'linux'"]
conflicts = [[{ group = "rocm" }, { group = "cuda" }]]

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
    { index = "pytorch-rocm", group = "rocm" },
    { index = "pytorch-cu130", group = "cuda" },
]
triton-rocm = [{ index = "pytorch-rocm", group = "rocm" }]
```

The user already made these changes in part. Still to do:

- Two leftovers from switching cu126 to cu130: the index `url` still ends in `cu126`, and `[tool.uv.sources]` still names `pytorch-cu126`.
- The ROCm torch 2.14 wheel depends on `triton-rocm`, which only the ROCm index serves, so that package, not `pytorch-triton-rocm`, must be routed there. Without it `uv lock` fails.
- `environments` restricts resolution to Linux. Otherwise uv also tries to resolve for macOS and Windows, where no ROCm build exists.
- The remaining sections (`dependencies`, `[dependency-groups]`, and the tool config below) are still missing.

Usage:

- ROCm is the default: plain `uv sync` and `uv run` install `torch+rocm7.2` on AMD.
- NVIDIA (CUDA 13.0 wheels, driver ≥ 580) opts in on every command with `--no-group rocm --group cuda`. Enabling both groups fails with a conflict error instead of silently switching builds.
- The torch builds were first extras (`--extra rocm`). That was changed to dependency groups because uv has no default for extras, and a plain `uv run` then silently re-synced to PyPI's CUDA torch, which also broke the ROCm `triton` install.
- One `uv.lock` covers both. The CUDA path can only be exercised on an NVIDIA machine.

Tooling config:

- **ruff:** line length 88. Rules `E, W, F, I, N, UP, B, Q`. Double quotes.
- **basedpyright:** mode `recommended`, `pythonVersion = "3.12"`. Disabled rules:
  - `reportAny`, `reportUnknownMemberType`, `reportUnknownVariableType` and `reportUnknownArgumentType`, because `nn.Module.__call__` and parts of torch/Lightning are typed as `Any` or left untyped;
  - `reportUnusedCallResult`, because in-place torch APIs (`nn.init.*_`, `backward`, `optimizer.step`) return values nobody uses.
  
  Everything else stays on, including `reportImplicitOverride` (so every override carries `@override`).
- **pytest:** `testpaths = ["tests"]` and `pythonpath = ["."]`, so tests can import `tests.helpers`. Three known-harmless warnings are filtered: torch's even-kernel 'same' padding note, Lightning's `num_workers` hint, and a Lightning `LeafSpec` deprecation.

Other files:

- `README.md` gets the two `uv sync` commands and `uv run pytest`.
- The placeholder files (`src/project/__init__.py`, `src/project/abc.py`, `scripts/test.py`) are left untouched.

## 9. Risks

- **Parameter count:** `PAPER` has about 1.1 M parameters: the FC head ~0.8 M, the difference-image full-width layer ~0.18 M. This is small enough for CPU tests.
- **Weight compatibility:** not a goal. `DIAGRAM` and `REPO_V1` reproduce the architectures, not NASA's weights.
- **Config size:** the config has 18 top-level fields, 10 of them per-branch configs built from 4 small branch config classes. That is the cost of being able to express all three sources. The presets and `with_overrides` keep day-to-day use to a one-liner.
- **Library versions:** newer torch or Lightning releases may change how their types show up in basedpyright. Versions are pinned through `uv.lock`.
