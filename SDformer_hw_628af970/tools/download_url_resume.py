#!/usr/bin/env python3
import argparse
import math
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def format_bytes(num: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num} B"


def get_total_size(url: str, timeout: int, max_retries: int) -> int | None:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.head(url, allow_redirects=True, timeout=timeout)
            resp.raise_for_status()
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
        except Exception as exc:
            last_error = exc
            time.sleep(min(5 * attempt, 30))
    raise RuntimeError(f"failed to fetch remote size after {max_retries} attempts: {last_error}")


class Progress:
    def __init__(self, initial: int) -> None:
        self.downloaded = initial
        self._lock = threading.Lock()

    def add(self, amount: int) -> int:
        with self._lock:
            self.downloaded += amount
            return self.downloaded

    def value(self) -> int:
        with self._lock:
            return self.downloaded


def download_range(
    *,
    url: str,
    part_path: str,
    start: int,
    end: int,
    chunk_size: int,
    timeout: int,
    max_retries: int,
    progress: Progress,
) -> None:
    target_size = end - start + 1
    existing = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    if existing > target_size:
        raise RuntimeError(f"part larger than target size: {part_path}")
    progress.add(existing)
    if existing == target_size:
        return

    attempts = 0
    while existing < target_size:
        range_start = start + existing
        headers = {"Range": f"bytes={range_start}-{end}"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                if resp.status_code != 206:
                    raise RuntimeError(f"server did not honor range request: status={resp.status_code}")
                with open(part_path, "ab") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        existing += len(chunk)
                        progress.add(len(chunk))
            attempts = 0
        except Exception as exc:
            attempts += 1
            if attempts > max_retries:
                raise RuntimeError(f"failed downloading part {part_path}: {exc}") from exc
            time.sleep(min(5 * attempts, 60))


def merge_parts(output: str, part_paths: list[str]) -> None:
    with open(output, "ab") as out:
        for part_path in part_paths:
            with open(part_path, "rb") as part_file:
                shutil.copyfileobj(part_file, out, length=8 * 1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume-capable concurrent URL downloader")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--part-size", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=20)
    args = parser.parse_args()

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    parts_dir = output + ".parts"
    os.makedirs(parts_dir, exist_ok=True)

    existing_prefix = os.path.getsize(output) if os.path.exists(output) else 0
    total = get_total_size(args.url, timeout=args.timeout, max_retries=args.max_retries)
    if total is not None and existing_prefix > total:
        raise RuntimeError(f"existing file larger than remote file: {existing_prefix} > {total}")
    if total is not None and existing_prefix == total:
        print(f"already complete: {output} ({format_bytes(total)})", flush=True)
        return 0
    if total is None:
        raise RuntimeError("remote size is required for concurrent download")

    print(f"url: {args.url}", flush=True)
    print(f"output: {output}", flush=True)
    print(f"resume_prefix: {format_bytes(existing_prefix)}", flush=True)
    print(f"remote_size: {format_bytes(total)}", flush=True)
    print(f"workers: {args.workers}", flush=True)
    print(f"part_size: {format_bytes(args.part_size)}", flush=True)

    progress = Progress(existing_prefix)
    started = time.time()
    last_report = started
    stop_event = threading.Event()
    error_holder: list[BaseException] = []
    tasks: list[tuple[int, int, str]] = []

    range_start = existing_prefix
    while range_start < total:
        range_end = min(total - 1, range_start + args.part_size - 1)
        part_path = os.path.join(parts_dir, f"part_{range_start:015d}_{range_end:015d}.bin")
        tasks.append((range_start, range_end, part_path))
        range_start = range_end + 1

    def report_progress() -> None:
        nonlocal last_report
        while not stop_event.wait(1):
            now = time.time()
            if now - last_report < args.progress_every:
                continue
            downloaded = progress.value()
            elapsed = max(now - started, 1e-6)
            speed = max(downloaded - existing_prefix, 0) / elapsed
            pct = downloaded / total * 100
            remaining = max(total - downloaded, 0)
            eta = remaining / speed if speed > 0 else float("inf")
            eta_str = f"{eta / 60:.1f} min" if eta != float("inf") else "inf"
            print(
                f"progress: {pct:.2f}% "
                f"({format_bytes(downloaded)}/{format_bytes(total)}), "
                f"speed: {format_bytes(speed)}/s, eta: {eta_str}",
                flush=True,
            )
            last_report = now

    reporter = threading.Thread(target=report_progress, daemon=True)
    reporter.start()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for part_start, part_end, part_path in tasks:
                futures.append(
                    executor.submit(
                        download_range,
                        url=args.url,
                        part_path=part_path,
                        start=part_start,
                        end=part_end,
                        chunk_size=args.chunk_size,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        progress=progress,
                    )
                )
            for future in futures:
                future.result()
        merge_parts(output, [part_path for _, _, part_path in tasks])
    except BaseException as exc:
        error_holder.append(exc)
    finally:
        stop_event.set()
        reporter.join(timeout=2)

    if error_holder:
        raise error_holder[0]

    final_size = os.path.getsize(output)
    print(f"done: {output} ({format_bytes(final_size)})", flush=True)
    if final_size != total:
        print(f"warning: final size {final_size} does not match remote size {total}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
