"""Downloads the ExoMiner++ dataset from Zenodo and builds the local parts.

The download is about 19 GB. TensorFlow decodes the TFRecord shards and is only
installed with the `build` dependency group.

Example:
    uv run --group build scripts/build_dataset.py --workers 4
"""

import argparse
import os
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets as hf  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
import pooch  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from project.dataset import DEFAULT_ROOT, PART_NAMES, SCHEMA, UNLABELED, PartName
from project.normalization import SOURCE_SHAPES

# isort: split
# Lightning's torch._dynamo crashes the process when imported after TensorFlow
import tensorflow as tf

EXPECTED_COUNTS: dict[PartName, int] = {"labeled": 57162, "unlabeled": 147567}
_BUILD_SCHEMA = hf.Features({**SCHEMA, "part": hf.Value("string")})


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """A file of the Zenodo record with its expected MD5."""

    name: str
    md5: str

    def fetch(self, raw: Path, processor: pooch.Untar | None = None) -> Any:  # pyright: ignore[reportExplicitAny]
        """Download into `raw` once, verify the MD5 and apply `processor`."""
        return pooch.retrieve(
            f"https://zenodo.org/api/records/15466293/files/{self.name}/content",
            known_hash=f"md5:{self.md5}",
            fname=self.name,
            path=raw,
            processor=processor,
            progressbar=True,
        )


ARCHIVE = RemoteFile(
    "tfrecords_tess-spoc-2min_s1-s67_9-24-2024_1159.tar.xz",
    "1932be3c3504443e165e6192ace86278",
)
LABELED_CATALOG = RemoteFile(
    "exominerplusplus_catalog_labeled_tces_s1-s67_tess-spoc-2min_complete_1-14-2025_1039.csv",
    "c79f3651d556e71f7878a46f960d09b4",
)
UNLABELED_CATALOG = RemoteFile(
    "exominerplusplus_catalog_unk_tces_s1-s67_tess-spoc-2min_complete_1-16-2025_1014.csv",
    "0bad0922ea704cf429fd69bacb13b0c9",
)


def read_catalog(labeled_csv: str, unlabeled_csv: str) -> dict[str, dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
    """Map each uid to its `part`, `label`, `fold` and `paper_score`."""
    labeled = pd.read_csv(labeled_csv, usecols=["uid", "label", "fold", "score"])
    unlabeled = pd.read_csv(unlabeled_csv, usecols=["TCE ID", "ExoMiner Score"])
    unlabeled = unlabeled.rename(columns={"TCE ID": "uid", "ExoMiner Score": "score"})
    catalog = pd.concat([
        labeled.assign(part="labeled"),
        unlabeled.assign(part="unlabeled", label=UNLABELED, fold=-1),
    ])
    repeated = catalog.loc[catalog["uid"].duplicated(), "uid"].tolist()
    if repeated:
        raise ValueError(f"the catalogs repeat uid {repeated[0]!r}")
    catalog = catalog.rename(columns={"score": "paper_score"}).set_index("uid")
    return catalog.to_dict("index")


def _value(feature: Any) -> NDArray[np.float32]:  # pyright: ignore[reportExplicitAny]
    kind = feature.WhichOneof("kind")
    if kind == "bytes_list":
        (blob,) = feature.bytes_list.value
        return tf.io.parse_tensor(blob, tf.float32).numpy()
    return np.asarray(getattr(feature, kind).value, dtype=np.float32)


def examples(
        shards: list[str],
        catalog: Mapping[str, Mapping[str, Any]],  # pyright: ignore[reportExplicitAny]
) -> Iterator[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
    """Yield every TCE of the TFRecord `shards`, joined with its catalog entry."""
    for record in tf.data.TFRecordDataset(shards):
        features = tf.train.Example.FromString(record.numpy()).features.feature
        uid = features["uid"].bytes_list.value[0].decode()
        if uid not in catalog:
            raise ValueError(f"TCE {uid} is in no catalog")
        try:
            columns = {
                name: _value(features[name]).reshape(shape)
                for name, shape in SOURCE_SHAPES.items()
            }
        except (KeyError, ValueError) as error:
            raise ValueError(f"TCE {uid}: {error}") from error
        target_id = features["target_id"].int64_list.value[0]
        yield {"uid": uid, "target_id": target_id, **catalog[uid], **columns}


def build(root: Path, workers: int) -> None:
    """Download, decode and save both parts to `root`; downloads are reused."""
    if all((root / name).exists() for name in PART_NAMES):
        print(f"The dataset in {root} is already built.")
        return
    raw = root / "raw"
    extracted: list[str] = ARCHIVE.fetch(raw, pooch.Untar())
    shards = sorted(path for path in extracted if Path(path).name.startswith("shard-"))
    catalog = read_catalog(LABELED_CATALOG.fetch(raw), UNLABELED_CATALOG.fetch(raw))
    tces = hf.Dataset.from_generator(
        examples,
        features=_BUILD_SCHEMA,
        gen_kwargs={"shards": shards, "catalog": catalog},
        num_proc=workers,
        cache_dir=str(raw / "cache"),
    )
    assert isinstance(tces, hf.Dataset)
    meta = tces.with_format("numpy").select_columns(["uid", "part", "fold"])[:]
    if len(np.unique(meta["uid"])) != len(tces):
        raise ValueError("some TCEs appear twice in the TFRecords")
    part_of = meta["part"]
    folds = meta["fold"][part_of == "labeled"]
    if not ((folds >= 0) & (folds <= 9)).all():
        raise ValueError("labeled folds must all be in 0-9")
    for name in PART_NAMES:
        rows = np.flatnonzero(part_of == name)
        expected = EXPECTED_COUNTS[name]
        if len(rows) != expected:
            raise ValueError(f"expected {expected} {name} TCEs, got {len(rows)}")
        partial = root / f"{name}.partial"
        tces.select(rows).remove_columns("part").save_to_disk(partial)
        os.replace(partial, root / name)
    shutil.rmtree(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="directory of the built dataset",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="processes decoding TFRecord shards in parallel",
    )
    arguments = parser.parse_args()
    build(arguments.root, arguments.workers)


if __name__ == "__main__":
    main()
