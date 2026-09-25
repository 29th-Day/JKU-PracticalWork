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

## Checks

```bash
uv run --extra rocm pytest
uv run --extra rocm basedpyright
uv run --extra rocm ruff check
```
