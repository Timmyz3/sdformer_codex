#!/usr/bin/env python3
"""Small HTTP range downloader for large MVSEC files."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
import urllib.request
from pathlib import Path


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers["Content-Length"])


def download_range(url: str, output: Path, start: int, end: int, retries: int) -> int:
    headers = {"Range": f"bytes={start}-{end}"}
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            expected = end - start + 1
            if len(data) != expected:
                raise IOError(f"short range {start}-{end}: got {len(data)} expected {expected}")
            with output.open("r+b") as handle:
                handle.seek(start)
                handle.write(data)
            return expected
        except Exception as exc:  # pragma: no cover - operational retry path
            last_exc = exc
            time.sleep(2)
    raise RuntimeError(f"failed range {start}-{end}") from last_exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mb", type=int, default=32)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    size = remote_size(args.url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".range_part")
    done = args.output.with_suffix(args.output.suffix + ".range_done")

    completed: set[int] = set()
    if done.exists():
        completed = {int(line.strip()) for line in done.read_text().splitlines() if line.strip()}

    with tmp.open("ab") as handle:
        handle.truncate(size)

    chunk = args.chunk_mb * 1024 * 1024
    ranges = []
    for idx, start in enumerate(range(0, size, chunk)):
        if idx in completed:
            continue
        end = min(start + chunk - 1, size - 1)
        ranges.append((idx, start, end))

    print(f"[range] url={args.url}")
    print(f"[range] output={args.output} size={size} ranges_left={len(ranges)} workers={args.workers}")
    started = time.time()
    bytes_done = len(completed) * chunk
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(download_range, args.url, tmp, start, end, args.retries): (idx, start, end)
            for idx, start, end in ranges
        }
        for future in concurrent.futures.as_completed(future_map):
            idx, start, end = future_map[future]
            nbytes = future.result()
            bytes_done += nbytes
            with done.open("a", encoding="utf-8") as handle:
                handle.write(f"{idx}\n")
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[range] chunk={idx} {start}-{end} done "
                f"{bytes_done / size * 100:.2f}% speed={bytes_done / elapsed / 1024 / 1024:.2f} MiB/s",
                flush=True,
            )

    if tmp.stat().st_size != size:
        raise RuntimeError(f"size mismatch {tmp.stat().st_size} != {size}")
    os.replace(tmp, args.output)
    done.unlink(missing_ok=True)
    print(f"[range] complete {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
