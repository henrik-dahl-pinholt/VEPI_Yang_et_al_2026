#!/usr/bin/env python3
"""Package large reproducibility inputs/results as GitHub release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAG = "repro-assets-v1"
DEFAULT_REPOSITORY = "henrik-dahl-pinholt/VEPI_Yang_et_al_2026"
DEFAULT_SOURCES = ("Data", "cache", "result")
DEFAULT_MAX_UNCOMPRESSED = int(1.5 * 1024**3)
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_source_files(source_dirs: list[str]) -> list[tuple[Path, int]]:
    files: list[tuple[Path, int]] = []
    for source in source_dirs:
        root = REPO_ROOT / source
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files.append((path, path.stat().st_size))
    return files


def chunk_files(files: list[tuple[Path, int]], max_uncompressed: int) -> list[list[tuple[Path, int]]]:
    chunks: list[list[tuple[Path, int]]] = []
    current: list[tuple[Path, int]] = []
    current_size = 0
    for path, size in files:
        if current and current_size + size > max_uncompressed:
            chunks.append(current)
            current = []
            current_size = 0
        current.append((path, size))
        current_size += size
    if current:
        chunks.append(current)
    return chunks


def make_archive(archive_path: Path, files: list[tuple[Path, int]], compresslevel: int) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz", compresslevel=compresslevel) as archive:
        for path, _size in files:
            archive.add(path, arcname=path.relative_to(REPO_ROOT).as_posix(), recursive=False)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "release_assets" / DEFAULT_TAG)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "repro_assets_manifest.json")
    parser.add_argument("--source-dir", action="append", dest="source_dirs")
    parser.add_argument("--max-uncompressed-gib", type=float, default=1.5)
    parser.add_argument("--compresslevel", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_dirs = args.source_dirs or list(DEFAULT_SOURCES)
    max_uncompressed = int(args.max_uncompressed_gib * 1024**3)
    files = iter_source_files(source_dirs)
    chunks = chunk_files(files, max_uncompressed)
    base_url = f"https://github.com/{args.repository}/releases/download/{args.tag}"

    archives = []
    for index, chunk in enumerate(chunks, start=1):
        name = f"vepi-yang-2026-{args.tag}-part{index:03d}.tar.gz"
        archive_path = args.output_dir / name
        if archive_path.exists() and not args.force:
            print(f"Using existing {archive_path}")
        else:
            print(f"Writing {archive_path}")
            make_archive(archive_path, chunk, args.compresslevel)
        compressed_size = archive_path.stat().st_size
        uncompressed_size = sum(size for _path, size in chunk)
        members = [path.relative_to(REPO_ROOT).as_posix() for path, _size in chunk]
        archives.append(
            {
                "file_count": len(chunk),
                "members": members,
                "name": name,
                "sha256": sha256_file(archive_path),
                "size_bytes": compressed_size,
                "uncompressed_bytes": uncompressed_size,
                "url": f"{base_url}/{name}",
            }
        )

    manifest = {
        "archives": archives,
        "base_url": base_url,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository": args.repository,
        "release_tag": args.tag,
        "schema_version": 1,
        "source_directories": source_dirs,
    }
    write_json(args.manifest, manifest)

    sums_path = args.output_dir / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{asset['sha256']}  {asset['name']}\n" for asset in archives),
        encoding="utf-8",
    )
    print(f"Wrote {args.manifest}")
    print(f"Wrote {sums_path}")
    print(f"{len(archives)} archive(s), {sum(a['size_bytes'] for a in archives) / 1_000_000_000:.2f} GB compressed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
