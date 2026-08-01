from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "W3M-dataset-downloader/1.0"})


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(url: str, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    response = SESSION.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def download_single_stream(url: str, output: Path, expected_size: int) -> None:
    temp = output.with_suffix(output.suffix + ".download")
    existing = temp.stat().st_size if temp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with SESSION.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
        if existing and response.status_code != 206:
            existing = 0
            temp.unlink(missing_ok=True)
            response.close()
            return download_single_stream(url, output, expected_size)
        response.raise_for_status()
        mode = "ab" if existing else "wb"
        with temp.open(mode) as stream:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    stream.write(block)
    if temp.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {output.name}: "
            f"{temp.stat().st_size} != {expected_size}"
        )
    os.replace(temp, output)


def download_range(
    url: str,
    part_path: Path,
    start: int,
    end: int,
) -> None:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(8):
        existing = part_path.stat().st_size if part_path.exists() else 0
        if existing == expected:
            return
        if existing > expected:
            part_path.unlink()
            existing = 0

        request_start = start + existing
        headers = {"Range": f"bytes={request_start}-{end}"}
        try:
            with requests.get(
                url,
                headers={**SESSION.headers, **headers},
                stream=True,
                timeout=(30, 240),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Server ignored byte range {request_start}-{end} "
                        f"for {url}: HTTP {response.status_code}"
                    )
                mode = "ab" if existing else "wb"
                with part_path.open(mode) as stream:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            stream.write(block)
            if part_path.stat().st_size == expected:
                return
            last_error = RuntimeError(
                f"Part size mismatch for {part_path.name}: "
                f"{part_path.stat().st_size} != {expected}"
            )
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
        time.sleep(min(30, 2 ** attempt))

    raise RuntimeError(
        f"Part download failed after retries: {part_path.name}"
    ) from last_error


def download_parallel(
    url: str,
    output: Path,
    expected_size: int,
    workers: int,
) -> None:
    # Zenodo throttles individual large-file connections heavily. One request
    # per ~1 MiB (bounded by --workers) gives useful throughput while keeping
    # total concurrent requests explicit and below the repository rate limit.
    worker_count = min(workers, max(1, math.ceil(expected_size / (1024 * 1024))))
    part_dir = output.parent / f".{output.name}.parts-{worker_count}"
    part_dir.mkdir(parents=True, exist_ok=True)
    segment_size = math.ceil(expected_size / worker_count)

    ranges: list[tuple[int, int, Path]] = []
    for index in range(worker_count):
        start = index * segment_size
        end = min(expected_size - 1, start + segment_size - 1)
        ranges.append((start, end, part_dir / f"{index:04d}.part"))

    # Reuse a prior interrupted single-stream download as the beginning of part 0.
    if output.exists() and 0 < output.stat().st_size <= segment_size:
        first_part = ranges[0][2]
        if not first_part.exists():
            shutil.copyfile(output, first_part)

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:
        futures = {
            executor.submit(download_range, url, part_path, start, end): (
                start,
                end,
                part_path,
            )
            for start, end, part_path in ranges
        }
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=15,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                future.result()
            downloaded = sum(
                part.stat().st_size if part.exists() else 0
                for _, _, part in ranges
            )
            elapsed = max(time.monotonic() - started, 0.001)
            mib_s = downloaded / 1024 / 1024 / elapsed
            percent = 100 * downloaded / expected_size
            print(
                f"  {output.name}: {percent:5.1f}% "
                f"({downloaded / 1024 / 1024:.1f}/"
                f"{expected_size / 1024 / 1024:.1f} MiB, "
                f"{mib_s:.2f} MiB/s)",
                flush=True,
            )

    temp = output.with_suffix(output.suffix + ".download")
    with temp.open("wb") as destination:
        for _, _, part_path in ranges:
            with part_path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    if temp.stat().st_size != expected_size:
        raise RuntimeError(
            f"Merged size mismatch for {output.name}: "
            f"{temp.stat().st_size} != {expected_size}"
        )
    os.replace(temp, output)
    shutil.rmtree(part_dir)


def download_file(
    *,
    url: str,
    output: Path,
    expected_size: int,
    expected_md5: str,
    workers: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == expected_size:
        if md5sum(output) == expected_md5.lower():
            print(f"SKIP verified: {output}", flush=True)
            return

    print(
        f"DOWNLOAD: {output.name} "
        f"({expected_size / 1024 / 1024:.1f} MiB)",
        flush=True,
    )
    if expected_size < 8 * 1024 * 1024 or workers == 1:
        download_single_stream(url, output, expected_size)
    else:
        download_parallel(url, output, expected_size, workers)

    actual_md5 = md5sum(output)
    if actual_md5 != expected_md5.lower():
        raise RuntimeError(
            f"MD5 mismatch for {output}: {actual_md5} != {expected_md5}"
        )
    print(f"VERIFIED: {output}", flush=True)


def download_zenodo(
    record_id: int,
    destination: Path,
    workers: int,
    skip_keys: set[str] | None = None,
) -> None:
    metadata_url = f"https://zenodo.org/api/records/{record_id}"
    record = fetch_json(
        metadata_url,
        destination / f"zenodo_record_{record_id}.json",
    )
    for file in record["files"]:
        if skip_keys and file["key"] in skip_keys:
            continue
        checksum = file["checksum"].removeprefix("md5:")
        file_url = (
            f"https://zenodo.org/records/{record_id}/files/"
            f"{quote(file['key'])}?download=1"
        )
        download_file(
            url=file_url,
            output=destination / file["key"],
            expected_size=int(file["size"]),
            expected_md5=checksum,
            workers=workers,
        )


def download_openmhc(destination: Path, workers: int) -> None:
    metadata_url = (
        "https://dataverse.harvard.edu/api/datasets/:persistentId/"
        "?persistentId=doi:10.7910/DVN/ZYMJF6"
    )
    record = fetch_json(
        metadata_url,
        destination / "dataverse_dataset_metadata.json",
    )
    for entry in record["data"]["latestVersion"]["files"]:
        data_file = entry["dataFile"]
        download_file(
            url=(
                "https://dataverse.harvard.edu/api/access/datafile/"
                f"{data_file['id']}"
            ),
            output=destination / entry["label"],
            expected_size=int(data_file["filesize"]),
            expected_md5=data_file["checksum"]["value"],
            workers=workers,
        )
    marker = {
        "version": "xs",
        "n_users": 593,
        "persistent_id": "doi:10.7910/DVN/ZYMJF6",
    }
    (destination / "dataset_version.json").write_text(
        json.dumps(marker, indent=2),
        encoding="utf-8",
    )


def download_hrv_figshare(
    destination: Path,
    workers: int,
    include_large_raw: bool,
) -> None:
    record = fetch_json(
        "https://api.figshare.com/v2/articles/28509740",
        destination / "figshare_article_28509740.json",
    )
    for file in record["files"]:
        if file["name"] == "raw_data.zip" and not include_large_raw:
            continue
        download_file(
            url=file["download_url"],
            output=destination / file["name"],
            expected_size=int(file["size"]),
            expected_md5=file["computed_md5"],
            workers=workers,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--include-large-raw-hrv", action="store_true")
    parser.add_argument("--include-pregnancy-processed", action="store_true")
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "lifesnaps",
            "ssaqs",
            "openmhc",
            "pregnancy",
            "hrv",
        ),
        default="all",
    )
    args = parser.parse_args()

    if not 1 <= args.workers <= 64:
        parser.error("--workers must be between 1 and 64")

    raw = args.root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "lifesnaps"):
        download_zenodo(
            6832242,
            raw / "lifesnaps_zenodo_6832242",
            args.workers,
        )
    if args.only in ("all", "ssaqs"):
        download_zenodo(
            18706837,
            raw / "ssaqs_zenodo_18706837",
            args.workers,
        )
    if args.only in ("all", "openmhc"):
        download_openmhc(
            raw / "openmhc_xs_dvn_zymjf6",
            args.workers,
        )
    if args.only in ("all", "pregnancy"):
        download_zenodo(
            7689724,
            raw / "pregnancy_ga_clock_zenodo_7689724",
            args.workers,
            skip_keys=(
                None
                if args.include_pregnancy_processed
                else {"Ravindra_s2sGAclock_processed_nomd_public.pkl"}
            ),
        )
    if args.only in ("all", "hrv"):
        download_hrv_figshare(
            raw / "wearable_hrv_sleep_figshare_28509740",
            args.workers,
            args.include_large_raw_hrv,
        )
    print("All directly downloadable datasets completed and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
