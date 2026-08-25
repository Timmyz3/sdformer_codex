#!/usr/bin/env python3
"""Verify an extracted handoff against its packaged manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    manifest_path = root / "handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    checked_bytes = 0
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            errors.append("missing: {}".format(entry["path"]))
            continue
        checked_bytes += path.stat().st_size
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            errors.append("identity mismatch: {}".format(entry["path"]))
    if errors:
        raise SystemExit("FAIL\n" + "\n".join(errors))
    if checked_bytes != manifest["total_bytes"]:
        raise SystemExit("FAIL total bytes: {} != {}".format(
            checked_bytes, manifest["total_bytes"]))
    print("PASS files={} bytes={}".format(len(manifest["files"]), checked_bytes))


if __name__ == "__main__":
    main()
