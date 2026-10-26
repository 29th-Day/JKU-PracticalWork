"""The ExoMiner++ TCE dataset, stored on disk as Hugging Face `datasets`."""

__all__ = [
    "DEFAULT_ROOT",
    "KNOWN_LABELS",
    "PART_NAMES",
    "PLANET_LABELS",
    "SCHEMA",
    "UNLABELED",
    "ExoMinerDataModule",
    "LabeledDataset",
    "PartName",
    "Split",
    "UnlabeledDataset",
    "load_part",
    "make_split",
    "paper_split",
    "random_split",
]

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, final, override

import datasets as hf  # pyright: ignore[reportMissingTypeStubs]
import lightning
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from project.features import Batch, UnlabeledBatch
from project.normalization import (
    SOURCE_SHAPES,
    STATS_SOURCES,
    NormStats,
    load_or_compute_stats,
    normalize_batch,
)

PartName = Literal["labeled", "unlabeled"]
PART_NAMES: tuple[PartName, PartName] = ("labeled", "unlabeled")

PLANET_LABELS: frozenset[str] = frozenset({"KP", "CP"})
KNOWN_LABELS: tuple[str, ...] = ("KP", "CP", "BD", "EB", "FP", "NTP", "NEB", "NPC")
UNLABELED = "UNK"

DEFAULT_ROOT = Path("data/exominer_plusplus_hf")


def _feature(shape: tuple[int, ...]) -> hf.Value | hf.List | hf.Array2D | hf.Array3D:
    match shape:
        case ():
            return hf.Value("float32")
        case (n,):
            return hf.List(hf.Value("float32"), length=n)
        case (_, _):
            return hf.Array2D(shape, "float32")
        case _:
            return hf.Array3D(shape, "float32")


SCHEMA = hf.Features({
    "uid": hf.Value("string"),
    "target_id": hf.Value("int64"),
    "label": hf.Value("string"),
    "fold": hf.Value("int8"),
    "paper_score": hf.Value("float32"),
    **{name: _feature(shape) for name, shape in SOURCE_SHAPES.items()},
})


def load_part(root: Path, name: PartName) -> hf.Dataset:
    """Open `name` in `root`, memory-mapped and returning numpy arrays."""
    try:
        part = hf.load_from_disk(root / name)
    except FileNotFoundError as error:
        hint = "run scripts/build_dataset.py to build it"
        message = f"Part {name!r} is not built in {root}: {hint}"
        raise FileNotFoundError(message) from error
    assert isinstance(part, hf.Dataset)
    return part.with_format("numpy")


@final
@dataclass(frozen=True, slots=True, eq=False)
class Split:
    """Sorted row indices into the labeled part for each split."""

    train: NDArray[np.int64]
    val: NDArray[np.int64]
    test: NDArray[np.int64]

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Split):
            return NotImplemented
        return (
            np.array_equal(self.train, other.train)
            and np.array_equal(self.val, other.val)
            and np.array_equal(self.test, other.test)
        )

    __hash__ = None  # pyright: ignore[reportAssignmentType]


def paper_split(
        fold: NDArray[np.integer],
        test_fold: int = 0,
        val_fold: int = 1,
) -> Split:
    """Split by the paper's cross-validation folds 0-9."""
    for number in (test_fold, val_fold):
        if not 0 <= number <= 9:
            raise ValueError(f"fold must be in 0-9, got {number}")
    if test_fold == val_fold:
        raise ValueError(f"test and val fold must differ, both are {test_fold}")
    test = np.flatnonzero(fold == test_fold).astype(np.int64)
    val = np.flatnonzero(fold == val_fold).astype(np.int64)
    train = np.flatnonzero((fold != test_fold) & (fold != val_fold)).astype(np.int64)
    return Split(train=train, val=val, test=test)


def random_split(
        target_id: NDArray[np.integer],
        seed: int,
        fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Split:
    """Split randomly by star so that no star is in two splits."""
    if any(f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError(f"fractions must be positive and sum to 1, got {fractions}")
    stars = np.unique(target_id)
    shuffled = stars[np.random.default_rng(seed).permutation(len(stars))]
    n_val = round(fractions[1] * len(stars))
    n_test = round(fractions[2] * len(stars))
    val_stars = shuffled[:n_val]
    test_stars = shuffled[n_val : n_val + n_test]
    val = np.flatnonzero(np.isin(target_id, val_stars)).astype(np.int64)
    test = np.flatnonzero(np.isin(target_id, test_stars)).astype(np.int64)
    rest = np.ones(len(target_id), dtype=bool)
    rest[val] = False
    rest[test] = False
    train = np.flatnonzero(rest).astype(np.int64)
    return Split(train=train, val=val, test=test)


def make_split(
        fold: NDArray[np.integer],
        target_id: NDArray[np.integer],
        seed: int | None,
        fractions: tuple[float, float, float],
        test_fold: int,
        val_fold: int,
) -> Split:
    """The paper's folds when `seed` is None, else a random split by star."""
    if seed is None:
        return paper_split(fold, test_fold, val_fold)
    return random_split(target_id, seed, fractions)


def _features(
        raw: Mapping[str, NDArray[np.float32]],
        stats: NormStats,
) -> dict[str, torch.Tensor]:
    normalized = normalize_batch(raw, stats)
    return {name: torch.from_numpy(value) for name, value in normalized.items()}


@final
class LabeledDataset(Dataset[Batch]):
    """Normalized batches of the given `rows` of the labeled part.

    Indexed by a list of positions, as yielded by a `BatchSampler`.
    """

    def __init__(
            self,
            part: hf.Dataset,
            rows: NDArray[np.int64],
            stats: NormStats,
    ) -> None:
        self.part = part.select(rows).select_columns([*SOURCE_SHAPES, "label"])
        self.stats = stats

    def __len__(self) -> int:
        return len(self.part)

    @override
    def __getitem__(self, index: list[int]) -> Batch:
        raw = self.part[index]
        is_planet = np.isin(raw.pop("label"), list(PLANET_LABELS))
        return {
            "features": _features(raw, self.stats),
            "label": torch.from_numpy(is_planet.astype(np.float32)),
        }


@final
class UnlabeledDataset(Dataset[UnlabeledBatch]):
    """Normalized batches of the unlabeled part, indexed like `LabeledDataset`."""

    def __init__(self, part: hf.Dataset, stats: NormStats) -> None:
        self.part = part.select_columns(list(SOURCE_SHAPES))
        self.stats = stats

    def __len__(self) -> int:
        return len(self.part)

    @override
    def __getitem__(self, index: list[int]) -> UnlabeledBatch:
        return {"features": _features(self.part[index], self.stats)}


@final
class ExoMinerDataModule(lightning.LightningDataModule):
    """Labeled train/val/test splits and the unlabeled TCEs to score.

    Normalization statistics come from the training split and travel in the
    checkpoint through `state_dict`.
    """

    def __init__(
            self,
            root: Path | str = DEFAULT_ROOT,
            seed: int | None = None,
            fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
            test_fold: int = 0,
            val_fold: int = 1,
            batch_size: int = 32,
            num_workers: int = 4,
    ) -> None:
        super().__init__()
        # checkpoints load with weights_only=True, which refuses a pickled Path
        root = str(root)
        self.save_hyperparameters()
        self.root = Path(root)
        self.seed = seed
        self.fractions = fractions
        self.test_fold = test_fold
        self.val_fold = val_fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split: Split | None = None
        self.norm_stats: NormStats | None = None
        self._labeled: hf.Dataset | None = None

    @override
    def setup(self, stage: str) -> None:
        if stage == "predict":
            return
        self._labeled = load_part(self.root, "labeled")
        self.split = make_split(
            np.asarray(self._labeled["fold"], dtype=np.int8),
            np.asarray(self._labeled["target_id"], dtype=np.int64),
            self.seed,
            self.fractions,
            self.test_fold,
            self.val_fold,
        )
        if self.norm_stats is None:
            train = self._labeled.select(self.split.train)

            def load_features() -> dict[str, NDArray[np.float32]]:
                return train.select_columns(list(STATS_SOURCES))[:]

            self.norm_stats = load_or_compute_stats(
                self.root / "norm_stats",
                np.asarray(train["uid"], dtype=np.str_),
                load_features,
            )

    def _loader[T](self, dataset: Dataset[T], shuffle: bool) -> DataLoader[T]:
        order = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)  # pyright: ignore[reportArgumentType]
        return DataLoader(
            dataset,
            sampler=BatchSampler(order, self.batch_size, drop_last=False),
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
        )

    def _labeled_loader(
            self,
            rows: NDArray[np.int64],
            shuffle: bool,
    ) -> DataLoader[Batch]:
        assert self._labeled is not None
        assert self.norm_stats is not None
        dataset = LabeledDataset(self._labeled, rows, self.norm_stats)
        return self._loader(dataset, shuffle)

    @override
    def train_dataloader(self) -> DataLoader[Batch]:
        assert self.split is not None
        return self._labeled_loader(self.split.train, shuffle=True)

    @override
    def val_dataloader(self) -> DataLoader[Batch]:
        assert self.split is not None
        return self._labeled_loader(self.split.val, shuffle=False)

    @override
    def test_dataloader(self) -> DataLoader[Batch]:
        assert self.split is not None
        return self._labeled_loader(self.split.test, shuffle=False)

    @override
    def predict_dataloader(self) -> DataLoader[UnlabeledBatch]:
        if self.norm_stats is None:
            raise RuntimeError(
                "No normalization statistics: predict with ckpt_path=<checkpoint>"
            )
        part = load_part(self.root, "unlabeled")
        return self._loader(UnlabeledDataset(part, self.norm_stats), shuffle=False)

    def predict_uids(self) -> NDArray[np.str_]:
        """The unlabeled `uid`s in `predict_dataloader` order."""
        return np.asarray(load_part(self.root, "unlabeled")["uid"], dtype=np.str_)

    @override
    def state_dict(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        return {"norm_stats": self.norm_stats}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # pyright: ignore[reportExplicitAny]
        self.norm_stats = state_dict["norm_stats"]
