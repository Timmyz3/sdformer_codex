#!/usr/bin/env python3
"""Safely list or extract a .tar.zst handoff without a system zstd binary."""

import argparse
import os
from pathlib import Path, PurePosixPath
import tarfile

import zstandard


def validate_member(member):
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe archive member: {}".format(member.name))
    if member.issym() or member.islnk() or member.isdev():
        raise ValueError("unsupported archive member type: {}".format(member.name))


def members(archive):
    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                for member in tar:
                    validate_member(member)
                    yield tar, member


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    if not args.archive.is_file():
        raise SystemExit("archive not found: {}".format(args.archive))
    if not args.list_only and args.destination.exists() and any(args.destination.iterdir()):
        raise SystemExit("destination must be absent or empty: {}".format(args.destination))

    count = 0
    total_bytes = 0
    preview = []
    if not args.list_only:
        args.destination.mkdir(parents=True, exist_ok=True)
    for tar, member in members(args.archive):
        count += 1
        total_bytes += member.size
        if len(preview) < 30:
            preview.append(member.name)
        if not args.list_only:
            tar.extract(member, str(args.destination))

    for name in preview:
        print(name)
    print("PASS archive members={} payload_bytes={} mode={}".format(
        count, total_bytes, "list" if args.list_only else "extract"))


if __name__ == "__main__":
    main()
