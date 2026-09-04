#!/usr/bin/env python3
"""Read-only verifier for an M1317 minimum ep34 handoff tar."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
from typing import Any, BinaryIO


MANIFEST_SCHEMA = "m1317_m1249_remote_minimum_ep34_handoff_pack_manifest_r1_v1"
MANIFEST_STATUS = "SEALED_EXACT_62_FILE_SELF_CONTAINED_HANDOFF__NO_REMOTE_EXECUTION"
MANIFEST_KEYS = {
    "schema", "status", "date", "archive_path", "manifest_member_path",
    "nonmanifest_entries", "counts", "claim_boundary",
}
ENTRY_KEYS = {"path", "size_bytes", "sha256", "role"}
PASS_TOKEN = "PASS_M1317_HANDOFF_PACK_READ_ONLY_VERIFY"


class VerifyError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise VerifyError(message)


def stream_sha(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_sha(stream)


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise VerifyError("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " is not regular non-symlink")


def safe_relative(value: Any) -> str:
    require(type(value) is str and value and not value.startswith("/") and
            "\\" not in value and "//" not in value, "unsafe archive path")
    pure = PurePosixPath(value)
    require(str(pure) == value and all(part not in {"", ".", ".."} for part in pure.parts),
            "traversing or non-canonical archive path")
    return value


def strict_json_bytes(payload: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            require(key not in value, "duplicate manifest key: " + key)
            value[key] = item
        return value

    def reject(value: str) -> Any:
        raise VerifyError("non-finite JSON: " + value)

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=reject)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerifyError("invalid manifest JSON") from exc
    require(isinstance(value, dict), "manifest root is not an object")
    return value


def parse_sha_file(path: Path, archive: Path) -> str:
    regular(path, "archive SHA file")
    rows = path.read_text(encoding="ascii").splitlines()
    require(len(rows) == 1, "archive SHA file must have one row")
    parts = rows[0].split("  ", 1)
    require(len(parts) == 2 and len(parts[0]) == 64 and parts[1] == archive.name,
            "archive SHA row mismatch")
    return parts[0]


def validate_manifest(value: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    require(set(value) == MANIFEST_KEYS and value["schema"] == MANIFEST_SCHEMA and
            value["status"] == MANIFEST_STATUS, "manifest schema/status/shape mismatch")
    manifest_member = safe_relative(value["manifest_member_path"])
    entries = value["nonmanifest_entries"]
    require(isinstance(entries, list) and len(entries) == 61, "manifest needs 61 entries")
    mapped: dict[str, dict[str, Any]] = {}
    for row in entries:
        require(isinstance(row, dict) and set(row) == ENTRY_KEYS, "entry shape mismatch")
        path = safe_relative(row["path"])
        require(path not in mapped and path != manifest_member, "duplicate/self manifest entry")
        require(type(row["size_bytes"]) is int and row["size_bytes"] >= 0 and
                type(row["sha256"]) is str and len(row["sha256"]) == 64 and
                type(row["role"]) is str and row["role"], "entry identity malformed")
        mapped[path] = row
    require(value["counts"] == {
        "inventory_remote_missing": 38,
        "M1313_files": 10,
        "remote_preflight": 1,
        "M1314_hammer_files": 10,
        "production_release": 1,
        "verifier": 1,
        "manifest": 1,
        "total_files": 62,
    }, "manifest count ledger mismatch")
    require(value["claim_boundary"] == {
        "remote_transfer_executed": False,
        "gpu": False,
        "capture": False,
        "eda": False,
        "paper_metric": False,
    }, "manifest claim boundary mismatch")
    return mapped, manifest_member


def verify(archive: Path, manifest_path: Path, sha_file: Path,
           expected_archive_sha256: str, expected_manifest_sha256: str) -> dict[str, Any]:
    for value, label in ((expected_archive_sha256, "expected archive SHA"),
                         (expected_manifest_sha256, "expected manifest SHA")):
        require(len(value) == 64 and all(char in "0123456789abcdef" for char in value),
                label + " malformed")
    regular(archive, "archive")
    regular(manifest_path, "sidecar manifest")
    require(file_sha(archive) == expected_archive_sha256 == parse_sha_file(sha_file, archive),
            "archive SHA mismatch")
    manifest_payload = manifest_path.read_bytes()
    require(hashlib.sha256(manifest_payload).hexdigest() == expected_manifest_sha256,
            "sidecar manifest SHA mismatch")
    entries, manifest_member = validate_manifest(strict_json_bytes(manifest_payload))

    with tarfile.open(archive, mode="r:") as handle:
        members = handle.getmembers()
        names = [member.name for member in members]
        require(len(names) == 62 and len(set(names)) == 62,
                "tar member population is not 62 unique files")
        require(set(names) == set(entries) | {manifest_member}, "tar member set mismatch")
        for member in members:
            safe_relative(member.name)
            require(member.isfile(), "tar contains a non-regular member")
            stream = handle.extractfile(member)
            require(stream is not None, "cannot read tar member")
            with stream:
                digest = stream_sha(stream)
            if member.name == manifest_member:
                require(member.size == len(manifest_payload) and digest == expected_manifest_sha256,
                        "embedded manifest differs from sidecar")
            else:
                row = entries[member.name]
                require(member.size == row["size_bytes"] and digest == row["sha256"],
                        "tar member size/SHA mismatch: " + member.name)
    return {
        "archive_sha256": expected_archive_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "files": 62,
        "remote_transfer_executed": False,
        "capture_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive-sha-file", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    result = verify(args.archive.resolve(), args.manifest.resolve(),
                    args.archive_sha_file.resolve(), args.expected_archive_sha256,
                    args.expected_manifest_sha256)
    print(PASS_TOKEN + " " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
