"""Authenticated recursive downloader for PhysioNet directory listings.

The password is collected with getpass and is never written to disk or echoed.
Downloads are constrained to the URL subtree supplied on the command line.
Partial files use a .part suffix and are resumed with HTTP Range requests.
"""

from __future__ import annotations

import argparse
import getpass
import os
import time
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlparse

import requests


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def _safe_relative_path(url: str, root_url: str) -> Path | None:
    parsed = urlparse(url)
    root = urlparse(root_url)
    if parsed.scheme != root.scheme or parsed.netloc != root.netloc:
        return None
    if parsed.query or parsed.fragment or not parsed.path.startswith(root.path):
        return None
    relative = unquote(parsed.path[len(root.path) :]).lstrip("/")
    parts = PurePosixPath(relative).parts
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return Path(*parts)


def _remote_metadata(session: requests.Session, url: str) -> tuple[int | None, str | None]:
    response = session.head(url, allow_redirects=True, timeout=60)
    if response.status_code in {401, 403}:
        response.raise_for_status()
    if not response.ok:
        return None, None
    size_text = response.headers.get("Content-Length")
    return (int(size_text) if size_text and size_text.isdigit() else None), response.headers.get(
        "Last-Modified"
    )


def _print_progress(name: str, downloaded: int, total: int | None, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-6)
    speed = downloaded / elapsed / (1024 * 1024)
    if total:
        percentage = downloaded * 100 / total
        print(
            f"\r{name}: {downloaded / 2**20:.1f}/{total / 2**20:.1f} MiB "
            f"({percentage:5.1f}%) {speed:.1f} MiB/s",
            end="",
            flush=True,
        )
    else:
        print(
            f"\r{name}: {downloaded / 2**20:.1f} MiB {speed:.1f} MiB/s",
            end="",
            flush=True,
        )


def _download_file(
    session: requests.Session,
    url: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote_size, last_modified = _remote_metadata(session, url)
    if destination.exists() and remote_size is not None and destination.stat().st_size == remote_size:
        print(f"SKIP {destination} ({remote_size / 2**20:.1f} MiB already complete)")
        return

    partial = destination.with_name(destination.name + ".part")
    if destination.exists() and not partial.exists():
        destination.replace(partial)
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = session.get(url, headers=headers, stream=True, timeout=(30, 180))
    response.raise_for_status()

    if offset and response.status_code == 206:
        mode = "ab"
        downloaded = offset
    else:
        mode = "wb"
        downloaded = 0
        offset = 0

    started = time.monotonic()
    last_report = started
    with partial.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_report >= 0.75:
                _print_progress(destination.name, downloaded, remote_size, started)
                last_report = now
        handle.flush()
        os.fsync(handle.fileno())
    _print_progress(destination.name, downloaded, remote_size, started)
    print()

    if remote_size is not None and partial.stat().st_size != remote_size:
        raise RuntimeError(
            f"size mismatch for {url}: got {partial.stat().st_size}, expected {remote_size}"
        )
    partial.replace(destination)
    if last_modified:
        modified = parsedate_to_datetime(last_modified).timestamp()
        os.utime(destination, (modified, modified))


def download_tree(root_url: str, destination: Path, username: str) -> None:
    root_url = root_url.rstrip("/") + "/"
    destination.mkdir(parents=True, exist_ok=True)
    password = os.environ.pop("PHYSIONET_PASSWORD", None)
    if not password:
        password = getpass.getpass(f"PhysioNet password for {username}: ")
    session = requests.Session()
    session.auth = (username, password)
    session.headers["User-Agent"] = "FemHealthBench-PhysioNet-Downloader/1.0"

    visited: set[str] = set()

    def walk(directory_url: str) -> None:
        if directory_url in visited:
            return
        visited.add(directory_url)
        print(f"INDEX {directory_url}")
        response = session.get(directory_url, timeout=(30, 120))
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            relative = _safe_relative_path(directory_url, root_url)
            if relative is not None:
                _download_file(session, directory_url, destination / relative)
            return

        parser = _LinkParser()
        parser.feed(response.text)
        candidates: list[tuple[str, Path, bool]] = []
        for href in parser.links:
            if href.startswith(("?", "#")) or href in {"../", "./", "/"}:
                continue
            absolute = urljoin(directory_url, href)
            relative = _safe_relative_path(absolute, root_url)
            if relative is None:
                continue
            is_directory = urlparse(absolute).path.endswith("/")
            candidates.append((absolute, relative, is_directory))

        for absolute, relative, is_directory in sorted(candidates, key=lambda item: item[0]):
            if is_directory:
                (destination / relative).mkdir(parents=True, exist_ok=True)
                walk(absolute)
            else:
                _download_file(session, absolute, destination / relative)

    walk(root_url)
    print(f"DONE: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    download_tree(args.url, args.dest.resolve(), args.user)


if __name__ == "__main__":
    main()
