This project uses:

- uv
- Pytorch Lighting
- TensorBoard
- Polars
- matplotlib
- numpy

Dataset, Model, Metrics, etc should all be defined inside `src/project` while `scripts` holds files that execute the training, evaluation, etc of the model

## Setup

Install the dependencies with the torch build that matches your GPU:

```bash
uv sync --extra rocm   # AMD, ROCm 7.2
uv sync --extra cuda   # NVIDIA, CUDA 13.0 (driver >= 580)
```

Always pass one of the extras, also to `uv run`: a plain `uv run` re-syncs the
environment and swaps in PyPI's default torch build. Alternatively, set
`UV_NO_SYNC=1` after syncing to make `uv run` keep the installed build.
If that swap already happened, `triton` and `triton-rocm` share files and the
ROCm one ends up broken (`module 'triton' has no attribute 'language'`); repair
it with `uv sync --extra rocm --reinstall-package triton-rocm`.

## Checks

Use the same extra you synced with (`rocm` on AMD, `cuda` on NVIDIA):

```bash
uv run --extra rocm pytest         # or: uv run --extra cuda pytest
uv run --extra rocm basedpyright   # or: uv run --extra cuda basedpyright
uv run --extra rocm ruff check     # or: uv run --extra cuda ruff check
```

## Inspecting the model

`scripts/explore_model.py` opens the model in
[Model Explorer](https://github.com/google-ai-edge/model-explorer). Every branch is
a collapsible group, and one forward pass on a random batch adds per-operation
overlays (mean, std, max_abs, zero_fraction for dead ReLUs, non_finite for NaN/inf):

```bash
uv run --extra rocm scripts/explore_model.py                                # PAPER preset
uv run --extra rocm scripts/explore_model.py --preset paper --preset repo_v1
uv run --extra rocm scripts/explore_model.py --override global_flux.pool_size=4
uv run --extra rocm scripts/explore_model.py --checkpoint path/to/model.ckpt
```

Pick an overlay in the viewer's node data panel. Add `--no-browser` on a remote
machine and open the printed URL yourself.
