This project uses:

- uv
- Pytorch Lighting
- TensorBoard
- Polars
- matplotlib
- numpy

Dataset, Model, Metrics, etc should all be defined inside `src/project` while `scripts` holds files that execute the training, evaluation, etc of the model

## Setup

AMD GPUs (ROCm 7.2) are the default, so plain uv commands work:

```bash
uv sync
uv run pytest
```

On NVIDIA (CUDA 13.0, driver >= 580), swap the torch build with the `cuda` group,
in every `uv sync` and `uv run`:

```bash
uv sync --no-group rocm --group cuda
uv run --no-group rocm --group cuda pytest
```

Forgetting `--no-group rocm` there fails with "groups `cuda` and `rocm` are
incompatible" instead of silently switching builds. On an NVIDIA-only machine,
`export UV_NO_GROUP=rocm` saves typing the first flag.

If `triton` ever breaks with `module 'triton' has no attribute 'language'` (the
PyPI `triton` and `triton-rocm` packages share files), repair it with
`uv sync --reinstall-package triton-rocm`.

## Checks

```bash
uv run pytest
uv run basedpyright
uv run ruff check
```

## Inspecting the model

`scripts/explore_model.py` opens the model in
[Model Explorer](https://github.com/google-ai-edge/model-explorer). Every branch is
a collapsible group, and one forward pass on a random batch adds per-operation
overlays (mean, std, max_abs, zero_fraction for dead ReLUs, non_finite for NaN/inf):

```bash
uv run scripts/explore_model.py                                # PAPER preset
uv run scripts/explore_model.py --preset paper --preset repo_v1
uv run scripts/explore_model.py --override global_flux.pool_size=4
uv run scripts/explore_model.py --checkpoint path/to/model.ckpt
```

Pick an overlay in the viewer's node data panel. Add `--no-browser` on a remote
machine and open the printed URL yourself.
