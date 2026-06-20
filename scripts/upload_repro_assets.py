#!/usr/bin/env python3
"""Create the reproducibility release and upload packaged assets."""

from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "repro_assets_manifest.json"
DEFAULT_ASSET_DIR = REPO_ROOT / "release_assets"
API_ROOT = "https://api.github.com"
UPLOAD_ROOT = "https://uploads.github.com"
CHUNK_SIZE = 1024 * 1024


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def request_json(method: str, url: str, token: str, payload: dict | None = None) -> tuple[int, dict | list | None]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("User-Agent", "vepi-repro-assets-uploader")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read()
            if not body:
                return response.status, None
            return response.status, json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"message": body}
        return exc.code, parsed


def get_or_create_release(repository: str, tag: str, target: str, token: str, dry_run: bool) -> dict:
    release_url = f"{API_ROOT}/repos/{repository}/releases/tags/{tag}"
    status, body = request_json("GET", release_url, token)
    if status == 200 and isinstance(body, dict):
        print(f"Using existing release {tag}")
        return body
    if status != 404:
        raise RuntimeError(f"failed to look up release {tag}: HTTP {status}: {body}")

    payload = {
        "body": "Large reproducibility assets for regenerating the VEPI figures.",
        "draft": False,
        "name": tag,
        "prerelease": False,
        "tag_name": tag,
        "target_commitish": target,
    }
    if dry_run:
        print(f"Would create release {tag} targeting {target}")
        return {"id": 0, "assets": []}

    status, body = request_json("POST", f"{API_ROOT}/repos/{repository}/releases", token, payload)
    if status != 201 or not isinstance(body, dict):
        raise RuntimeError(f"failed to create release {tag}: HTTP {status}: {body}")
    print(f"Created release {tag}")
    return body


def list_release_assets(repository: str, release_id: int, token: str) -> dict[str, dict]:
    assets: dict[str, dict] = {}
    page = 1
    while True:
        url = f"{API_ROOT}/repos/{repository}/releases/{release_id}/assets?per_page=100&page={page}"
        status, body = request_json("GET", url, token)
        if status != 200 or not isinstance(body, list):
            raise RuntimeError(f"failed to list release assets: HTTP {status}: {body}")
        for asset in body:
            assets[asset["name"]] = asset
        if len(body) < 100:
            return assets
        page += 1


def delete_asset(repository: str, asset_id: int, token: str) -> None:
    status, body = request_json("DELETE", f"{API_ROOT}/repos/{repository}/releases/assets/{asset_id}", token)
    if status != 204:
        raise RuntimeError(f"failed to delete existing asset {asset_id}: HTTP {status}: {body}")


def upload_asset(repository: str, release_id: int, token: str, path: Path) -> None:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    query = urllib.parse.urlencode({"name": path.name})
    url_path = f"/repos/{repository}/releases/{release_id}/assets?{query}"

    connection = http.client.HTTPSConnection(urllib.parse.urlsplit(UPLOAD_ROOT).netloc)
    connection.putrequest("POST", url_path)
    connection.putheader("Accept", "application/vnd.github+json")
    connection.putheader("Authorization", f"Bearer {token}")
    connection.putheader("Content-Length", str(path.stat().st_size))
    connection.putheader("Content-Type", content_type)
    connection.putheader("User-Agent", "vepi-repro-assets-uploader")
    connection.putheader("X-GitHub-Api-Version", "2022-11-28")
    connection.endheaders()

    uploaded = 0
    total = path.stat().st_size
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            connection.send(chunk)
            uploaded += len(chunk)
            if uploaded % (128 * CHUNK_SIZE) < CHUNK_SIZE:
                print(f"  {path.name}: {uploaded / 1_000_000:.1f} / {total / 1_000_000:.1f} MB", flush=True)

    response = connection.getresponse()
    body = response.read().decode("utf-8", errors="replace")
    connection.close()
    if response.status != 201:
        raise RuntimeError(f"failed to upload {path.name}: HTTP {response.status}: {body}")


def upload_files(
    repository: str,
    release_id: int,
    token: str,
    paths: list[Path],
    existing_assets: dict[str, dict],
    replace: bool,
    dry_run: bool,
) -> None:
    for path in paths:
        if not path.exists():
            raise RuntimeError(f"missing asset file: {path}")
        existing = existing_assets.get(path.name)
        if existing and replace:
            if dry_run:
                print(f"Would delete existing {path.name}")
            else:
                print(f"Deleting existing {path.name}")
                delete_asset(repository, existing["id"], token)
        elif existing:
            print(f"Skipping existing {path.name}")
            continue

        if dry_run:
            print(f"Would upload {path.name} ({path.stat().st_size / 1_000_000:.1f} MB)")
        else:
            print(f"Uploading {path.name} ({path.stat().st_size / 1_000_000:.1f} MB)")
            upload_asset(repository, release_id, token, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--target", default="main", help="target commitish for a newly created release tag")
    parser.add_argument("--replace", action="store_true", help="replace existing release assets with the same name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token and not args.dry_run:
        raise RuntimeError(f"set {args.token_env} to a GitHub token with release upload permission")

    manifest = load_json(args.manifest)
    repository = manifest["repository"]
    tag = manifest["release_tag"]
    asset_dir = args.asset_dir or DEFAULT_ASSET_DIR / tag
    paths = [asset_dir / asset["name"] for asset in manifest["archives"]]
    sums_path = asset_dir / "SHA256SUMS"
    if sums_path.exists():
        paths.append(sums_path)

    if args.dry_run and not token:
        print(f"Would use repository {repository}, release {tag}, asset directory {asset_dir}")
        for path in paths:
            print(f"Would upload {path.name} ({path.stat().st_size / 1_000_000:.1f} MB)")
        return 0

    release = get_or_create_release(repository, tag, args.target, token, args.dry_run)
    existing_assets = {} if args.dry_run else list_release_assets(repository, release["id"], token)
    upload_files(repository, release["id"], token, paths, existing_assets, args.replace, args.dry_run)
    print("Release assets are uploaded.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
