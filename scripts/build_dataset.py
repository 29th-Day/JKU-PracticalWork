"""Downloads the ExoMiner++ dataset from Zenodo and builds the local parts.

The download is about 19 GB and resumes after an interruption. The TFRecord
shards are decoded without TensorFlow or protobuf.

Example:
    uv run scripts/build_dataset.py --workers 4
"""

import argparse
import csv
import hashlib
import multiprocessing
import os
import shutil
import struct
import tarfile
import time
import urllib.request
from collections.abc import Collection, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from project.dataset import (
    MANIFEST_NAME,
    PART_NAMES,
    UNLABELED,
    Manifest,
    Part,
    PartName,
    PartWriter,
    open_part,
    read_manifest,
    write_manifest,
)
from project.normalization import SOURCE_SHAPES

type FeatureValue = list[bytes] | NDArray[np.float32] | NDArray[np.int64]

ZENODO_RECORD = "15466293"
EXPECTED_COUNTS: dict[PartName, int] = {"labeled": 57162, "unlabeled": 147567}
_DT_FLOAT = 1
_MASK64 = (1 << 64) - 1


def read_records(path: Path) -> Iterator[bytes]:
    """Yield the payloads of a TFRecord file; CRCs are skipped.

    A truncated trailing record ends the iteration.
    """
    with path.open("rb") as file:
        while True:
            header = file.read(12)
            if len(header) < 12:
                return
            (length,) = struct.unpack_from("<Q", header)
            body = file.read(length + 4)
            if len(body) < length + 4:
                return
            yield body[:length]


def _read_varint(data: bytes | memoryview, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def _fields(data: bytes | memoryview) -> Iterator[tuple[int, int, int | memoryview]]:
    """Yield `(number, wire_type, value)`; delimited and fixed values are views."""
    view = memoryview(data)
    pos = 0
    end = len(view)
    while pos < end:
        key, pos = _read_varint(view, pos)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, pos = _read_varint(view, pos)
            yield number, wire_type, value
        elif wire_type in (1, 2, 5):
            if wire_type == 2:
                length, pos = _read_varint(view, pos)
            else:
                length = 8 if wire_type == 1 else 4
            if pos + length > end:
                raise ValueError("truncated protobuf field")
            yield number, wire_type, view[pos : pos + length]
            pos += length
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")


def _signed(value: int) -> int:
    return value - (1 << 64) if value >> 63 else value


def _decode_feature(data: memoryview) -> FeatureValue:
    for number, _, body in _fields(data):
        assert isinstance(body, memoryview)
        if number == 1:
            return [bytes(v) for n, _, v in _fields(body) if n == 1]
        if number == 2:
            chunks: list[NDArray[np.float32]] = []
            for n, _, v in _fields(body):
                if n != 1:
                    continue
                assert isinstance(v, memoryview)
                chunks.append(np.frombuffer(v, "<f4"))
            if not chunks:
                return np.empty(0, np.float32)
            return np.concatenate(chunks).astype(np.float32)
        if number == 3:
            ints: list[int] = []
            for n, wire_type, v in _fields(body):
                if n != 1:
                    continue
                if wire_type == 0:
                    assert isinstance(v, int)
                    ints.append(_signed(v))
                else:
                    assert isinstance(v, memoryview)
                    pos = 0
                    while pos < len(v):
                        item, pos = _read_varint(v, pos)
                        ints.append(_signed(item))
            return np.array(ints, dtype=np.int64)
    return []


def decode_example(payload: bytes, wanted: Collection[str]) -> dict[str, FeatureValue]:
    """Decode only the `wanted` features of a serialized `Example`."""
    result: dict[str, FeatureValue] = {}
    for number, _, features in _fields(payload):
        if number != 1:
            continue
        assert isinstance(features, memoryview)
        for entry_number, _, entry in _fields(features):
            if entry_number != 1:
                continue
            assert isinstance(entry, memoryview)
            name = ""
            feature: memoryview | None = None
            for n, _, value in _fields(entry):
                assert isinstance(value, memoryview)
                if n == 1:
                    name = bytes(value).decode()
                elif n == 2:
                    feature = value
            if name in wanted and feature is not None:
                result[name] = _decode_feature(feature)
    return result


def decode_tensor(blob: bytes) -> NDArray[np.float32]:
    """Decode a serialized float32 `TensorProto` into an array of its shape."""
    dtype = _DT_FLOAT
    shape: list[int] = []
    content: memoryview | None = None
    floats: list[NDArray[np.float32]] = []
    for number, _, value in _fields(blob):
        if number == 1:
            assert isinstance(value, int)
            dtype = value
        elif number == 2:
            assert isinstance(value, memoryview)
            for n, _, dim in _fields(value):
                if n != 2:
                    continue
                assert isinstance(dim, memoryview)
                shape.extend(size for m, _, size in _fields(dim) if m == 1)  # pyright: ignore[reportArgumentType]
        elif number == 4:
            assert isinstance(value, memoryview)
            content = value
        elif number == 5:
            assert isinstance(value, memoryview)
            floats.append(np.frombuffer(value, "<f4"))
    if dtype != _DT_FLOAT:
        raise ValueError(f"expected a DT_FLOAT tensor, got dtype {dtype}")
    if content is not None:
        flat = np.frombuffer(content, "<f4")
    elif floats:
        flat = np.concatenate(floats)
    else:
        flat = np.empty(0, np.float32)
    return flat.astype(np.float32).reshape(shape)


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Where a TCE lives (`part`, `row`) and what NASA's catalog says about it."""

    part: PartName
    row: int
    label: str
    fold: int
    paper_score: float


def _read_catalog(
    path: Path,
    part: PartName,
    columns: Mapping[str, str],
    catalog: dict[str, CatalogEntry],
    seen_in: dict[str, Path],
) -> None:
    """Add the rows of `path` to `catalog`; `columns` maps roles to CSV headers."""
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        missing = [c for c in columns.values() if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} has no column {missing[0]!r}")
        for row, record in enumerate(reader):
            uid = record[columns["uid"]]
            if uid in catalog:
                raise ValueError(
                    f"{path} repeats uid {uid!r} (first seen in {seen_in[uid]})"
                )
            seen_in[uid] = path
            if part == "labeled":
                label, fold = record["label"], int(record["fold"])
            else:
                label, fold = UNLABELED, -1
            catalog[uid] = CatalogEntry(
                part,
                row,
                label,
                fold,
                float(record[columns["score"]]),
            )


def read_catalogs(labeled: Path, unlabeled: Path) -> dict[str, CatalogEntry]:
    """Read both catalog CSVs into one `uid -> CatalogEntry` mapping."""
    catalog: dict[str, CatalogEntry] = {}
    seen_in: dict[str, Path] = {}
    _read_catalog(
        labeled,
        "labeled",
        {"uid": "uid", "label": "label", "fold": "fold", "score": "score"},
        catalog,
        seen_in,
    )
    _read_catalog(
        unlabeled,
        "unlabeled",
        {"uid": "TCE ID", "score": "ExoMiner Score"},
        catalog,
        seen_in,
    )
    return catalog


def _column(record: Mapping[str, FeatureValue], name: str) -> NDArray[np.float32]:
    shape = SOURCE_SHAPES[name]
    value = record.get(name)
    if value is None:
        raise ValueError(f"feature {name!r} is missing")
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"feature {name!r} holds {len(value)} tensors, expected 1")
        array = decode_tensor(value[0])
    else:
        array = value.astype(np.float32, copy=False)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"feature {name!r} has {array.size} values, expected {shape}")
    return array.reshape(shape)


def _decode_job(shard: Path, out: Path) -> int:
    if out.exists():
        return -1
    wanted = {"uid", "target_id", *SOURCE_SHAPES}
    uids: list[str] = []
    target_ids: list[int] = []
    columns: dict[str, list[NDArray[np.float32]]] = {name: [] for name in SOURCE_SHAPES}
    for index, payload in enumerate(read_records(shard)):
        record = decode_example(payload, wanted)
        try:
            uid, target_id = record["uid"], record["target_id"]
            assert isinstance(uid, list) and isinstance(target_id, np.ndarray)
            uids.append(uid[0].decode())
            target_ids.append(int(target_id[0]))
            for name in SOURCE_SHAPES:
                columns[name].append(_column(record, name))
        except (KeyError, AssertionError, IndexError, ValueError) as error:
            raise ValueError(f"{shard}, record {index}: {error}") from error
    arrays: dict[str, NDArray[np.generic]] = {
        "uid": np.array(uids, dtype=str),
        "target_id": np.array(target_ids, dtype=np.int64),
    }
    for name, shape in SOURCE_SHAPES.items():
        arrays[name] = (
            np.stack(columns[name]) if columns[name]
            else np.empty((0, *shape), np.float32)
        )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with tmp.open("wb") as file:
        np.savez(file, **arrays)  # pyright: ignore[reportArgumentType]
    os.replace(tmp, out)
    return len(uids)


def decode_shard(
    shard: Path,
    _catalog: Mapping[str, CatalogEntry],
    out: Path,
) -> int:
    """Decode a TFRecord shard into `out` (`.npz`); -1 if `out` already exists.

    Checks every feature against `SOURCE_SHAPES`. Catalog membership is checked by
    `assemble`, so the catalog is not needed here and is not sent to workers.
    """
    return _decode_job(shard, out)


def assemble(
    decoded: Sequence[Path],
    catalog: Mapping[str, CatalogEntry],
    root: Path,
) -> dict[PartName, int]:
    """Place the decoded shards' TCEs into the parts missing from `root`."""
    totals: dict[PartName, int] = dict.fromkeys(PART_NAMES, 0)
    for entry in catalog.values():
        totals[entry.part] += 1
    todo: list[PartName] = [n for n in PART_NAMES if not (root / n).exists()]
    # First pass reads only the uids, so nothing is written for a bad input
    seen: set[str] = set()
    unknown: list[str] = []
    filled: dict[PartName, set[int]] = {name: set() for name in PART_NAMES}
    for path in decoded:
        with np.load(path) as npz:
            for uid in npz["uid"].tolist():
                if uid in seen:
                    raise ValueError(f"TCE {uid} appears twice in the TFRecords")
                seen.add(uid)
                entry = catalog.get(uid)
                if entry is None:
                    unknown.append(uid)
                else:
                    filled[entry.part].add(entry.row)
    if unknown:
        raise ValueError(f"{len(unknown)} TCEs are in no catalog, e.g. {unknown[0]}")
    for name in todo:
        if len(filled[name]) != totals[name]:
            absent = [
                uid for uid, e in catalog.items()
                if e.part == name and uid not in seen
            ]  # fmt: skip
            count, example = len(absent), absent[0]
            raise ValueError(
                f"{count} {name} TCEs are missing from the TFRecords, e.g. {example}"
            )
    root.mkdir(parents=True, exist_ok=True)
    writers = {name: PartWriter(root, name, totals[name]) for name in todo}
    for path in decoded:
        with np.load(path) as npz:
            uids = npz["uid"].tolist()
            entries = [catalog[uid] for uid in uids]
            target_ids = npz["target_id"]
            features = {name: npz[name] for name in SOURCE_SHAPES}
        for name, writer in writers.items():
            keep = np.array([e.part == name for e in entries], dtype=bool)
            if not keep.any():
                continue
            chosen = [e for e in entries if e.part == name]
            writer.write(
                np.array([e.row for e in chosen], dtype=np.int64),
                {
                    "uid": np.array(uids)[keep],
                    "target_id": target_ids[keep],
                    "label": np.array([e.label for e in chosen]),
                    "fold": np.array([e.fold for e in chosen], dtype=np.int8),
                    "paper_score": np.array(
                        [e.paper_score for e in chosen],
                        dtype=np.float32,
                    ),
                },
                {column: array[keep] for column, array in features.items()},
            )
    for writer in writers.values():
        writer.finish()
    return totals


def build_from_shards(
    shards: Sequence[Path],
    catalog: Mapping[str, CatalogEntry],
    root: Path,
    work_dir: Path,
    workers: int,
) -> dict[PartName, int]:
    """Decode `shards` in parallel into `work_dir`, then `assemble` them."""
    outputs = [work_dir / f"{shard.name}.npz" for shard in shards]
    work_dir.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [
            pool.submit(_decode_job, shard, out)
            for shard, out in zip(shards, outputs, strict=True)
        ]
        for future in futures:
            future.result()
    return assemble(outputs, catalog, root)


def nan_counts(part: Part) -> dict[str, int]:
    """Rows with any NaN per source column; columns without NaN are left out."""
    counts: dict[str, int] = {}
    for name, array in part.features.items():
        total = 0
        for start in range(0, len(array), 4096):
            chunk = np.asarray(array[start : start + 4096])
            chunk = chunk.reshape(len(chunk), -1)
            total += int(np.isnan(chunk).any(axis=1).sum())
        if total:
            counts[name] = total
    return counts


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """A file of the Zenodo record with its expected size and MD5."""

    name: str
    size: int
    md5: str

    @property
    def url(self) -> str:
        return (
            f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{self.name}/content"
        )


ARCHIVE = RemoteFile(
    "tfrecords_tess-spoc-2min_s1-s67_9-24-2024_1159.tar.xz",
    18954105376,
    "1932be3c3504443e165e6192ace86278",
)
LABELED_CATALOG = RemoteFile(
    "exominerplusplus_catalog_labeled_tces_s1-s67_tess-spoc-2min_complete_1-14-2025_1039.csv",
    8944735,
    "c79f3651d556e71f7878a46f960d09b4",
)
UNLABELED_CATALOG = RemoteFile(
    "exominerplusplus_catalog_unk_tces_s1-s67_tess-spoc-2min_complete_1-16-2025_1014.csv",
    36391080,
    "0bad0922ea704cf429fd69bacb13b0c9",
)
_CHUNK = 1 << 20


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        while chunk := file.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(file: RemoteFile, dest_dir: Path, url: str | None = None) -> Path:
    """Download `file` into `dest_dir`, resuming a `.part` file, and verify its MD5.

    A verified file is marked with an empty `<name>.md5ok` and not requested again.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / file.name
    part = dest_dir / f"{file.name}.part"
    marker = dest_dir / f"{file.name}.md5ok"
    if final.exists() and marker.exists():
        return final
    if final.exists():
        # a crash between the rename and the marker leaves the file unmarked
        if _md5(final) == file.md5:
            marker.touch()
            return final
        final.unlink()
    have = part.stat().st_size if part.exists() else 0
    if have > file.size:
        part.unlink()
        have = 0
    if have < file.size:
        request = urllib.request.Request(url or file.url)
        if have:
            request.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(request, timeout=60) as response:
            if have and response.status != 206:
                have = 0
            last = time.monotonic()
            with part.open("ab" if have else "wb") as out:
                while chunk := response.read(_CHUNK):
                    out.write(chunk)
                    have += len(chunk)
                    if time.monotonic() - last >= 5:
                        last = time.monotonic()
                        print(f"{file.name}: {have / file.size:.1%}", flush=True)
    if part.stat().st_size < file.size:
        raise RuntimeError(f"{file.name}: the download ended early, rerun to resume it")
    if _md5(part) != file.md5:
        part.unlink()
        raise RuntimeError(f"{file.name}: MD5 mismatch, the download was deleted")
    part.replace(final)
    marker.touch()
    return final


def extract(archive: Path, dest: Path) -> list[Path]:
    """Extract a `.tar.xz` into `dest` (once) and return its sorted `shard-*` files."""
    if not dest.exists():
        partial = dest.with_name(dest.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(partial, filter="data")
        partial.rename(dest)
    return sorted(dest.rglob("shard-*"))


def build(root: Path, workers: int) -> Manifest:
    """Download, decode and write the whole dataset to `root`; resumable."""
    if (root / MANIFEST_NAME).exists():
        print(f"The dataset in {root} is already built.")
        return read_manifest(root)
    raw = root / "raw"
    files = [ARCHIVE, LABELED_CATALOG, UNLABELED_CATALOG]
    archive, labeled_csv, unlabeled_csv = (download(f, raw) for f in files)
    shards = extract(archive, raw / "tfrecords")
    size = sum(shard.stat().st_size for shard in shards)
    print(f"{len(shards)} shards, {size / 1e9:.1f} GB")
    catalog = read_catalogs(labeled_csv, unlabeled_csv)
    counts = build_from_shards(shards, catalog, root, raw / "decoded", workers)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"expected {EXPECTED_COUNTS} TCEs, got {counts}")
    labeled = open_part(root, "labeled")
    if not ((labeled.fold >= 0) & (labeled.fold <= 9)).all():
        raise ValueError("labeled folds must all be in 0-9")
    for name in PART_NAMES:
        print(f"NaN rows in {name}: {nan_counts(open_part(root, name))}")
    manifest = Manifest(
        zenodo_record=ZENODO_RECORD,
        files={f.name: f.md5 for f in files},
        source_shapes={k: list(v) for k, v in SOURCE_SHAPES.items()},
        parts={name: n for name, n in counts.items()},
        built=date.today().isoformat(),
    )
    write_manifest(root, manifest)
    shutil.rmtree(raw)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/exominer_plusplus"),
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
