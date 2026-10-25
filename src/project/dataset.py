"""Memory-mapped on-disk storage of the ExoMiner++ TCE dataset."""

__all__ = [
    "DEFAULT_ROOT",
    "KNOWN_LABELS",
    "LABEL_DTYPE",
    "ExoMinerDataModule",
    "LabeledDataset",
    "MANIFEST_NAME",
    "PART_NAMES",
    "PLANET_LABELS",
    "UID_DTYPE",
    "UNLABELED",
    "UnlabeledDataset",
    "Manifest",
    "Part",
    "PartName",
    "PartWriter",
    "Split",
    "load_part",
    "make_split",
    "open_part",
    "paper_split",
    "random_split",
    "read_manifest",
    "write_manifest",
]

import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, final, override

import lightning
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from project.features import Batch, UnlabeledBatch
from project.normalization import (
    SOURCE_SHAPES,
    NormStats,
    load_or_compute_stats,
    normalize_example,
)

PartName = Literal["labeled", "unlabeled"]
PART_NAMES: tuple[PartName, PartName] = ("labeled", "unlabeled")

PLANET_LABELS: frozenset[str] = frozenset({"KP", "CP"})
KNOWN_LABELS: tuple[str, ...] = ("KP", "CP", "BD", "EB", "FP", "NTP", "NEB", "NPC")
UNLABELED = "UNK"

DEFAULT_ROOT = Path("data/exominer_plusplus")

UID_DTYPE = "<U32"
LABEL_DTYPE = "<U3"
MANIFEST_NAME = "manifest.json"

_META_DIR = "meta"
_META_DTYPES: dict[str, str] = {
    "uid": UID_DTYPE,
    "target_id": "int64",
    "label": LABEL_DTYPE,
    "fold": "int8",
    "paper_score": "float32",
}


class Manifest(TypedDict):
    """What a dataset directory was built from."""

    zenodo_record: str
    files: dict[str, str]
    source_shapes: dict[str, list[int]]
    parts: dict[str, int]
    built: str


@dataclass(frozen=True, eq=False)
class Part:
    """One memory-mapped part (`labeled` or `unlabeled`) of the dataset.

    Pickles by path so that DataLoader workers reopen the memory maps instead
    of copying them.
    """

    root: Path
    name: PartName
    uid: NDArray[np.str_]
    target_id: NDArray[np.int64]
    label: NDArray[np.str_]
    fold: NDArray[np.int8]
    paper_score: NDArray[np.float32]
    features: Mapping[str, NDArray[np.float32]]

    def __len__(self) -> int:
        return len(self.uid)

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (load_part, (self.root, self.name))


def write_manifest(root: Path, manifest: Manifest) -> None:
    """Atomically write `manifest` to `root`."""
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f"{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, root / MANIFEST_NAME)


def read_manifest(root: Path) -> Manifest:
    """Read the manifest of the dataset in `root`."""
    manifest: Manifest = json.loads((root / MANIFEST_NAME).read_text())
    return manifest


def load_part(root: Path, name: PartName) -> Part:
    """Open `name` in `root` with every array memory-mapped."""
    hint = "run scripts/build_dataset.py to build it"
    if not (root / MANIFEST_NAME).exists():
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {root}: {hint}")
    if name not in read_manifest(root)["parts"]:
        raise FileNotFoundError(f"Part {name!r} is not built in {root}: {hint}")
    return open_part(root, name)


def open_part(root: Path, name: PartName) -> Part:
    """`load_part` without the manifest check, for the build that writes it."""
    directory = root / name
    meta = {
        key: np.load(directory / _META_DIR / f"{key}.npy", mmap_mode="r")
        for key in _META_DTYPES
    }
    features = {
        column: np.load(directory / f"{column}.npy", mmap_mode="r")
        for column in SOURCE_SHAPES
    }
    return Part(
        root=root,
        name=name,
        uid=meta["uid"],
        target_id=meta["target_id"],
        label=meta["label"],
        fold=meta["fold"],
        paper_score=meta["paper_score"],
        features=features,
    )


@final
class PartWriter:
    """Fills a part in `<name>.partial/` and renames it to `<name>` on `finish`."""

    def __init__(self, root: Path, name: PartName, n_rows: int) -> None:
        self._final = root / name
        self._partial = root / f"{name}.partial"
        if self._partial.exists():
            shutil.rmtree(self._partial)
        (self._partial / _META_DIR).mkdir(parents=True)
        self._meta = {
            key: np.lib.format.open_memmap(
                self._partial / _META_DIR / f"{key}.npy",
                mode="w+",
                dtype=dtype,
                shape=(n_rows,),
            )
            for key, dtype in _META_DTYPES.items()
        }
        self._features = {
            column: np.lib.format.open_memmap(
                self._partial / f"{column}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n_rows, *shape),
            )
            for column, shape in SOURCE_SHAPES.items()
        }

    def write(
            self,
            rows: NDArray[np.int64],
            meta: Mapping[str, NDArray[np.generic]],
            features: Mapping[str, NDArray[np.float32]],
    ) -> None:
        """Store the given examples at row indices `rows`."""
        too_long = [uid for uid in meta["uid"].tolist() if len(uid) > 32]
        if too_long:
            raise ValueError(f"uid longer than 32 characters: {too_long[0]!r}")
        for key, array in self._meta.items():
            array[rows] = meta[key]
        for column, array in self._features.items():
            array[rows] = features[column]

    def finish(self) -> None:
        """Flush everything and publish the part under its final name."""
        for array in (*self._meta.values(), *self._features.values()):
            array.flush()
        self._meta.clear()
        self._features.clear()
        if self._final.exists():
            shutil.rmtree(self._final)
        os.replace(self._partial, self._final)


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


def _features_of(part: Part, row: int, stats: NormStats) -> dict[str, torch.Tensor]:
    raw = {name: np.asarray(column[row]) for name, column in part.features.items()}
    normalized = normalize_example(raw, stats)
    # memory-mapped rows are read-only, so copy before handing them to torch
    return {
        name: torch.from_numpy(np.array(value, dtype=np.float32))
        for name, value in normalized.items()
    }


@final
class LabeledDataset(Dataset[Batch]):
    """Normalized examples of the given `rows` of a labeled part."""

    def __init__(self, part: Part, rows: NDArray[np.int64], stats: NormStats) -> None:
        self.part = part
        self.rows = rows
        self.stats = stats

    def __len__(self) -> int:
        return len(self.rows)

    @override
    def __getitem__(self, index: int) -> Batch:
        row = int(self.rows[index])
        is_planet = self.part.label[row] in PLANET_LABELS
        return {
            "features": _features_of(self.part, row, self.stats),
            "label": torch.tensor(1.0 if is_planet else 0.0),
        }


@final
class UnlabeledDataset(Dataset[UnlabeledBatch]):
    """Normalized examples of every row of an unlabeled part."""

    def __init__(self, part: Part, stats: NormStats) -> None:
        self.part = part
        self.stats = stats

    def __len__(self) -> int:
        return len(self.part)

    @override
    def __getitem__(self, index: int) -> UnlabeledBatch:
        return {"features": _features_of(self.part, index, self.stats)}


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
        self._labeled: Part | None = None

    @override
    def setup(self, stage: str) -> None:
        if stage == "predict":
            return
        self._labeled = load_part(self.root, "labeled")
        self.split = make_split(
            self._labeled.fold,
            self._labeled.target_id,
            self.seed,
            self.fractions,
            self.test_fold,
            self.val_fold,
        )
        if self.norm_stats is None:
            self.norm_stats = load_or_compute_stats(
                self.root / "norm_stats",
                self._labeled.features,
                self._labeled.uid,
                self.split.train,
            )

    def _labeled_loader(
            self,
            rows: NDArray[np.int64],
            shuffle: bool,
    ) -> DataLoader[Batch]:
        assert self._labeled is not None
        assert self.norm_stats is not None
        return DataLoader(
            LabeledDataset(self._labeled, rows, self.norm_stats),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
        )

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
        return DataLoader(
            UnlabeledDataset(load_part(self.root, "unlabeled"), self.norm_stats),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
        )

    def predict_uids(self) -> NDArray[np.str_]:
        """The unlabeled `uid`s in `predict_dataloader` order."""
        return np.array(load_part(self.root, "unlabeled").uid)

    @override
    def state_dict(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        return {"norm_stats": self.norm_stats}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # pyright: ignore[reportExplicitAny]
        self.norm_stats = state_dict["norm_stats"]
