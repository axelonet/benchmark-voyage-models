#!/usr/bin/env python3
"""Restore the checked-in, precomputed starter embeddings without a Voyage API call."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SNAPSHOTS = DATA / "snapshots"
TARGET = DATA / "prepared"
PARTS = (
    "starter-embeddings-20260921.tar.gz.part-aa",
    "starter-embeddings-20260921.tar.gz.part-ab",
    "starter-embeddings-20260921.tar.gz.part-ac",
)
ARCHIVE_SHA256 = "0e9d84705845dcbf9da6c6539ff0ea5d6cc5b17f65185e20d826c00a63536439"


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_safely(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise SystemExit(f"Unsafe archive path: {member.name}") from error
            if member.issym() or member.islnk():
                raise SystemExit(f"Snapshot must not contain links: {member.name}")
        bundle.extractall(destination, members=members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace an existing data/prepared directory")
    args = parser.parse_args()

    missing = [name for name in PARTS if not (SNAPSHOTS / name).is_file()]
    if missing:
        raise SystemExit(f"Missing snapshot part(s): {', '.join(missing)}")
    if TARGET.exists() and not args.force:
        raise SystemExit("data/prepared already exists; use --force only to replace it.")

    with tempfile.TemporaryDirectory(prefix="snapshot-", dir=DATA) as temporary:
        temporary_path = Path(temporary)
        archive = temporary_path / "starter-embeddings-20260921.tar.gz"
        with archive.open("wb") as output:
            for name in PARTS:
                with (SNAPSHOTS / name).open("rb") as source:
                    shutil.copyfileobj(source, output)
        if archive_sha256(archive) != ARCHIVE_SHA256:
            raise SystemExit("Snapshot checksum mismatch; re-clone the repository before retrying.")

        extracted = temporary_path / "extracted"
        extracted.mkdir()
        extract_safely(archive, extracted)
        prepared = extracted / "prepared"
        if not prepared.is_dir():
            raise SystemExit("Snapshot is missing its prepared directory.")
        if TARGET.exists():
            shutil.rmtree(TARGET)
        shutil.move(str(prepared), TARGET)

    print(json.dumps({"status": "restored", "target": str(TARGET), "archive_sha256": ARCHIVE_SHA256}))


if __name__ == "__main__":
    main()
