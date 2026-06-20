#!/usr/bin/env python3
"""Download and unpack reproducibility assets from a GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "repro_assets_manifest.json"
DEFAULT_DOWNLOAD_DIR = REPO_ROOT / ".repro_assets"
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, output_path: Path, expected_size: int | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            downloaded = 0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if expected_size and downloaded % (64 * CHUNK_SIZE) < CHUNK_SIZE:
                    pct = 100 * downloaded / expected_size
                    print(f"  {downloaded / 1_000_000:.1f} MB ({pct:.1f}%)", flush=True)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"failed to download {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to download {url}: {exc.reason}") from exc
    tmp_path.replace(output_path)


def assert_safe_member(member: tarfile.TarInfo, dest_root: Path) -> None:
    member_path = Path(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise RuntimeError(f"unsafe archive member path: {member.name}")
    if member.issym() or member.islnk():
        raise RuntimeError(f"refusing to extract link from archive: {member.name}")
    target = (dest_root / member_path).resolve()
    root = dest_root.resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"archive member escapes destination: {member.name}")


def extract_archive(path: Path, dest_root: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            assert_safe_member(member, dest_root)
        archive.extractall(dest_root, members=members)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "archives" not in manifest:
        raise RuntimeError(f"manifest has no 'archives' field: {path}")
    return manifest


def archive_url(asset: dict, manifest: dict, base_url: str | None) -> str:
    if base_url:
        return base_url.rstrip("/") + "/" + asset["name"]
    if asset.get("url"):
        return asset["url"]
    return manifest["base_url"].rstrip("/") + "/" + asset["name"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--dest", type=Path, default=REPO_ROOT)
    parser.add_argument("--base-url", help="override the release asset base URL")
    parser.add_argument("--force", action="store_true", help="redownload archives even if hashes match")
    parser.add_argument("--no-extract", action="store_true", help="download and verify only")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.dest.mkdir(parents=True, exist_ok=True)

    for asset in manifest["archives"]:
        name = asset["name"]
        output_path = args.download_dir / name
        expected_hash = asset["sha256"]
        expected_size = asset.get("size_bytes")

        if output_path.exists() and not args.force:
            current_hash = sha256_file(output_path)
            if current_hash == expected_hash:
                print(f"Using cached {name}")
            else:
                print(f"Cached {name} hash mismatch; redownloading")
                output_path.unlink()

        if not output_path.exists() or args.force:
            url = archive_url(asset, manifest, args.base_url)
            print(f"Downloading {name}")
            download_file(url, output_path, expected_size)

        actual_hash = sha256_file(output_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA256 mismatch for {name}: {actual_hash} != {expected_hash}")

        if not args.no_extract:
            print(f"Extracting {name}")
            extract_archive(output_path, args.dest)

    print("Reproducibility assets are ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
