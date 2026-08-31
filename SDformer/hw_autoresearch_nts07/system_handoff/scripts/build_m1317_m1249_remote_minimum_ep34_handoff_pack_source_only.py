#!/usr/bin/env python3
"""M1317 fail-closed builder for the minimum M1249 ep34 remote handoff.

The checked-in builder source is inert without an exact production release.
The M1314 independent hammer is now sealed, but both its exact entry and a
matching production release are mandatory before ``build_once`` can
consume its one-shot marker and create the local tar.  This source contains no
network or GPU operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
INVENTORY = HW / (
    "system_handoff/inventory/"
    "m1249_remote_minimum_recursive_dependency_inventory_r1_20260831.json")
INVENTORY_SHA256 = "e9ff5d02d28912f2ec741c2aa812f93e3b8a2626a59c495ed2b764a2994b959e"
SOURCE_CONTRACT = HW / (
    "contracts/m1317_m1249_remote_minimum_ep34_handoff_pack_builder_"
    "source_contract_r1_20260831.json")
TEST = HW / (
    "tests/test_m1317_m1249_remote_minimum_ep34_handoff_pack_builder_source.py")

M1313_CONTRACT = HW / (
    "contracts/m1313_motion_ep34_final_unified_capture_production_launch_r1_20260831.json")
M1313_CHECKER = HW / (
    "system_simulator/scripts/"
    "check_m1313_motion_ep34_final_unified_capture_production_launch.py")
M1313_TEST = HW / (
    "tests/test_m1313_motion_ep34_final_unified_capture_production_launch.py")
M1313_RECEIPT_ROOT = HW / (
    "reviews/m1313_motion_ep34_final_unified_capture_production_launch_author_r1_20260831")
REMOTE_PREFLIGHT = HW / (
    "system_handoff/scripts/preflight_m1317_m1249_ep34_remote_capture_read_only.py")
REMOTE_PREFLIGHT_SHA256 = "068161855ea6ba49604ef38701aa237317cfe89c45216a99bdfe709da39d87f8"
VERIFIER = HW / (
    "system_handoff/scripts/verify_m1317_m1249_remote_minimum_ep34_handoff_pack.py")
VERIFIER_SHA256 = "55a3c1ad38085e204e2ca4da3750c0284c54dbed56cbe1fd0ccea552872de70d"
PAYLOAD_MANIFEST = HW / (
    "system_handoff/packs/m1317_m1249_remote_minimum_ep34_handoff_manifest_r1_20260831.json")
M1313 = {
    "contract": {
        "path": str(M1313_CONTRACT.relative_to(ROOT)),
        "sha256": "eeb0a8380e51610652ec6cdf1c2bb58c22395c9d72608e98f6a88a18f5c6bbda",
    },
    "checker": {
        "path": str(M1313_CHECKER.relative_to(ROOT)),
        "sha256": "b1241b66d589281e29dcedc5a535a2c3cee4bc2f0dee18fb1774e8b5efcd35f4",
    },
    "test": {
        "path": str(M1313_TEST.relative_to(ROOT)),
        "sha256": "908166d9cd214bff220881b3a14f602e4990247484a81a3af63f030946413abe",
    },
    "author_receipt": {
        "path": str(M1313_RECEIPT_ROOT.relative_to(ROOT)),
        "manifest_sha256": "6ba2bbb32d611f9a04995315289549798cb8d26342330e56a1bf1e72c9d827c8",
        "outer_file_sha256": "2d0f27bc06b42a972e1035dcfe346509bea04406d7690bd081a0e727f3410073",
        "author_receipt_sha256": "dca15e9a207d831dfd5c4d45ebabf25815bfddb70a382a9c0c57ae61bfeefd3e",
    },
}

M1314_SCHEMA = "m1314_m1313_motion_ep34_final_unified_capture_production_launch_blind_hammer_r1_v1"
M1314_STATUS = "PASS_M1314_M1313_BLIND_HAMMER__ROOT_AGENT_SINGLE_REMOTE_CAPTURE_ONLY__NO_RETRY"
M1314_ENTRY_KEYS = {"path", "manifest_sha256", "outer_file_sha256", "review_sha256"}
M1314_REVIEW_KEYS = {
    "schema", "status", "verdict", "date", "reviewed_identity", "independence",
    "verification", "authorization", "hammer_execution", "claim_boundary", "docs359_sha256",
}
M1314_AUTHORIZATION = {
    "authorized_actor": "root_agent",
    "production_capture": True,
    "remote_capture_runs": 1,
    "automatic_retry": False,
    "authorization_transferable": False,
    "exact_M1313_contract_only": True,
    "exact_canonical_namespaces_only": True,
}
M1314_BOUNDARY = {
    "capture_complete": False,
    "paper_metric": False,
    "hardware_speedup": False,
    "system_speedup": False,
    "energy": False,
    "ppa": False,
}
M1314_ENTRY = {
    "path": "hw_autoresearch_nts07/reviews/m1314_m1313_motion_ep34_final_unified_capture_production_launch_blind_hammer_r1_20260831",
    "manifest_sha256": "1fbd77896e91241df5b1ffa32efdbd76fdc145b5af3823ad79272fc9241db1d5",
    "outer_file_sha256": "44cf8e5f8babf96346878cfbe8efb83929f13fa4c81fe180fd38646b82d3cef2",
    "review_sha256": "26a01134f4089f67ae3c74ca4633939f26d0b3b0d29d5ebf7b31bdb96d0027b6",
}

RELEASE_SCHEMA = "m1317_m1249_remote_minimum_ep34_handoff_pack_production_release_r1_v1"
RELEASE_STATUS = "M1314_BOUND__ONE_LOCAL_MINIMUM_HANDOFF_PACK_BUILD_AUTHORIZED"
RELEASE_PATH = HW / (
    "contracts/m1317_m1249_remote_minimum_ep34_handoff_pack_production_release_r1_20260831.json")
OUTPUT = HW / (
    "system_handoff/packs/m1317_m1249_remote_minimum_ep34_handoff_r1_20260831.tar")
ATTEMPT = HW / (
    "system_handoff/packs/.m1317_m1249_remote_minimum_ep34_handoff_r1_20260831."
    "attempt_consumed")
PARTIAL = HW / (
    "system_handoff/packs/.m1317_m1249_remote_minimum_ep34_handoff_r1_20260831.partial")
ATTEMPT_TOKEN = "M1317_PACK_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
SOURCE_SCHEMA = "m1317_m1249_remote_minimum_ep34_handoff_pack_builder_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1314_HAMMER_REQUIRED__NO_PACK_NO_TRANSFER_NO_GPU"
PASS_TOKEN = "PASS_M1317_MINIMUM_EP34_HANDOFF_PACK_BUILT__NO_REMOTE_TRANSFER"

IDENTITY_KEYS = {"path", "sha256"}
RECEIPT_KEYS = {"path", "manifest_sha256", "outer_file_sha256", "author_receipt_sha256"}
RELEASE_KEYS = {
    "schema", "status", "contract_path", "builder_identity", "inventory",
    "m1313", "m1314_hammer", "verifier", "payload_manifest", "one_shot", "output",
}
BUILDER_KEYS = {
    "source_path", "source_sha256", "test_path", "test_sha256",
    "source_contract_path", "source_contract_sha256",
}


class M1317Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1317Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    def reject(value: str) -> Any:
        raise M1317Error("non-finite JSON constant: " + value)

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                           parse_constant=reject)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M1317Error("invalid JSON: " + str(path)) from exc
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def safe_relative(value: Any) -> str:
    require(type(value) is str and value and not value.startswith("/") and
            "\\" not in value and "//" not in value, "unsafe repository path")
    pure = PurePosixPath(value)
    require(str(pure) == value and all(part not in {"", ".", ".."} for part in pure.parts),
            "non-canonical or traversing repository path")
    return value


def regular_exact(path: Path, expected_sha: str, expected_size: int | None = None,
                  label: str = "file") -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise M1317Error("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " is not regular non-symlink")
    if expected_size is not None:
        require(path.stat().st_size == expected_size, label + " size mismatch")
    require(sha256(path) == expected_sha, label + " SHA mismatch")


def _manifest_rows(root: Path, manifest_sha: str, outer_sha: str) -> dict[str, str]:
    regular_exact(root / "SHA256SUMS", manifest_sha, label="sealed manifest")
    regular_exact(root / "SHA256SUMS.seal.sha256", outer_sha, label="sealed outer")
    rows: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        require(len(parts) == 2 and len(parts[0]) == 64, "malformed manifest row")
        digest, name = parts
        safe_relative(name)
        require("/" not in name and name not in rows, "nested or duplicate manifest member")
        rows[name] = digest
        regular_exact(root / name, digest, label="sealed member " + name)
    population = {path.name for path in root.iterdir()
                  if path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(population == set(rows), "sealed directory population mismatch")
    return rows


def validate_inventory(value: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    regular_exact(INVENTORY, INVENTORY_SHA256, label="minimum dependency inventory")
    inventory = strict_json(INVENTORY) if value is None else value
    require(inventory.get("schema") ==
            "m1249_remote_minimum_recursive_dependency_inventory_r1_v1" and
            inventory.get("status") ==
            "PASS_READ_ONLY_INVENTORY__38_TRANSFER_REQUIRED__55_REMOTE_EXACT__NO_TRANSFER_EXECUTED",
            "inventory schema/status mismatch")
    require(inventory.get("counts", {}).get("transfer_required_files") == 38 and
            inventory.get("counts", {}).get("remote_missing_files") == 38 and
            inventory.get("transfer_required", {}).get("all_remote_status") == "MISSING",
            "inventory transfer count/status mismatch")

    entries: list[dict[str, Any]] = []
    for row in inventory["transfer_required"]["singleton_files"]:
        require({"path", "size_bytes", "sha256", "role"} <= set(row),
                "singleton inventory row incomplete")
        path = safe_relative(row["path"])
        require(type(row["size_bytes"]) is int and row["size_bytes"] >= 0 and
                type(row["sha256"]) is str and len(row["sha256"]) == 64,
                "singleton identity malformed")
        regular_exact(ROOT / path, row["sha256"], row["size_bytes"], path)
        entries.append({"path": path, "size_bytes": row["size_bytes"],
                        "sha256": row["sha256"], "role": row["role"]})

    for directory in inventory["transfer_required"]["sealed_directories"]:
        base = safe_relative(directory["path"])
        root = ROOT / base
        rows = _manifest_rows(root, directory["manifest_sha256"],
                              directory["outer_file_sha256"])
        require(set(rows) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"} ==
                set(directory["members"]), "inventory sealed member names mismatch")
        for name, member in directory["members"].items():
            safe_relative(name)
            path = base + "/" + name
            require(rows.get(name, directory["manifest_sha256"] if name == "SHA256SUMS" else
                             directory["outer_file_sha256"]) == member["sha256"],
                    "inventory sealed member SHA mismatch")
            regular_exact(ROOT / path, member["sha256"], member["size_bytes"], path)
            entries.append({"path": path, "size_bytes": member["size_bytes"],
                            "sha256": member["sha256"], "role": directory["role"]})

    paths = [row["path"] for row in entries]
    require(len(entries) == 38 and len(set(paths)) == 38,
            "inventory closure is not exactly 38 unique files")
    require(sum(row["size_bytes"] for row in entries) == 223543,
            "inventory transfer byte count mismatch")
    return entries


def validate_m1313() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for role in ("contract", "checker", "test"):
        row = M1313[role]
        require(set(row) == IDENTITY_KEYS, "M1313 identity shape mismatch")
        safe_relative(row["path"])
        regular_exact(ROOT / row["path"], row["sha256"], label="M1313 " + role)
        entries.append({"path": row["path"], "size_bytes": (ROOT / row["path"]).stat().st_size,
                        "sha256": row["sha256"], "role": "M1313_" + role})

    receipt = M1313["author_receipt"]
    require(set(receipt) == RECEIPT_KEYS, "M1313 author receipt entry shape mismatch")
    base = safe_relative(receipt["path"])
    rows = _manifest_rows(ROOT / base, receipt["manifest_sha256"],
                          receipt["outer_file_sha256"])
    require(rows.get("author_receipt.json") == receipt["author_receipt_sha256"],
            "M1313 author receipt member mismatch")
    for name, digest in sorted(rows.items()):
        path = base + "/" + name
        entries.append({"path": path, "size_bytes": (ROOT / path).stat().st_size,
                        "sha256": digest, "role": "M1313_author_receipt"})
    for name, digest in (("SHA256SUMS", receipt["manifest_sha256"]),
                         ("SHA256SUMS.seal.sha256", receipt["outer_file_sha256"])):
        path = base + "/" + name
        entries.append({"path": path, "size_bytes": (ROOT / path).stat().st_size,
                        "sha256": digest, "role": "M1313_author_receipt"})
    require(len(entries) == 10, "M1313 handoff addition must be 3 files plus 7 receipt files")
    return entries


def base_payload() -> list[dict[str, Any]]:
    regular_exact(REMOTE_PREFLIGHT, REMOTE_PREFLIGHT_SHA256, label="remote read-only preflight")
    entries = validate_inventory() + validate_m1313() + [{
        "path": str(REMOTE_PREFLIGHT.relative_to(ROOT)),
        "size_bytes": REMOTE_PREFLIGHT.stat().st_size,
        "sha256": REMOTE_PREFLIGHT_SHA256,
        "role": "remote_read_only_preflight",
    }]
    verify_m1314(M1314_ENTRY)
    base = M1314_ENTRY["path"]
    rows = _manifest_rows(ROOT / base, M1314_ENTRY["manifest_sha256"],
                          M1314_ENTRY["outer_file_sha256"])
    for name, digest in sorted(rows.items()):
        path = base + "/" + name
        entries.append({"path": path, "size_bytes": (ROOT / path).stat().st_size,
                        "sha256": digest, "role": "M1314_independent_hammer"})
    for name, digest in (("SHA256SUMS", M1314_ENTRY["manifest_sha256"]),
                         ("SHA256SUMS.seal.sha256", M1314_ENTRY["outer_file_sha256"])):
        path = base + "/" + name
        entries.append({"path": path, "size_bytes": (ROOT / path).stat().st_size,
                        "sha256": digest, "role": "M1314_independent_hammer"})
    regular_exact(VERIFIER, VERIFIER_SHA256, label="M1317 pack verifier")
    entries.append({
        "path": str(VERIFIER.relative_to(ROOT)),
        "size_bytes": VERIFIER.stat().st_size,
        "sha256": VERIFIER_SHA256,
        "role": "pack_read_only_verifier",
    })
    paths = [row["path"] for row in entries]
    require(len(entries) == 60 and len(set(paths)) == 60,
            "base handoff payload must be exactly 60 unique files")
    return sorted(entries, key=lambda row: row["path"])


def exact_payload(release_path: Path = RELEASE_PATH) -> list[dict[str, Any]]:
    require(release_path.resolve() == RELEASE_PATH and release_path.exists(),
            "exact production release is required for final payload")
    entries = base_payload()
    regular_exact(release_path, sha256(release_path), label="M1317 production release")
    entries.append({
        "path": str(release_path.relative_to(ROOT)),
        "size_bytes": release_path.stat().st_size,
        "sha256": sha256(release_path),
        "role": "M1317_production_release",
    })
    require(PAYLOAD_MANIFEST.exists(), "exact payload manifest is required")
    manifest = strict_json(PAYLOAD_MANIFEST)
    require(manifest.get("schema") ==
            "m1317_m1249_remote_minimum_ep34_handoff_pack_manifest_r1_v1" and
            manifest.get("status") ==
            "SEALED_EXACT_62_FILE_SELF_CONTAINED_HANDOFF__NO_REMOTE_EXECUTION",
            "payload manifest schema/status mismatch")
    require(manifest.get("archive_path") == str(OUTPUT.relative_to(ROOT)) and
            manifest.get("manifest_member_path") == str(PAYLOAD_MANIFEST.relative_to(ROOT)),
            "payload manifest path mismatch")
    expected = sorted(entries, key=lambda row: row["path"])
    require(manifest.get("nonmanifest_entries") == expected,
            "payload manifest does not exactly enumerate 61 nonmanifest members")
    require(manifest.get("counts") == {
        "inventory_remote_missing": 38, "M1313_files": 10,
        "remote_preflight": 1, "M1314_hammer_files": 10,
        "production_release": 1, "verifier": 1, "manifest": 1,
        "total_files": 62}, "payload manifest count ledger mismatch")
    require(manifest.get("claim_boundary") == {
        "remote_transfer_executed": False, "gpu": False, "capture": False,
        "eda": False, "paper_metric": False}, "payload manifest boundary mismatch")
    entries.append({
        "path": str(PAYLOAD_MANIFEST.relative_to(ROOT)),
        "size_bytes": PAYLOAD_MANIFEST.stat().st_size,
        "sha256": sha256(PAYLOAD_MANIFEST),
        "role": "M1317_payload_manifest",
    })
    require(len(entries) == 62 and len({row["path"] for row in entries}) == 62,
            "final handoff payload must be exactly 62 unique files")
    return sorted(entries, key=lambda row: row["path"])


def verify_m1314(entry: Any) -> dict[str, Any]:
    require(isinstance(entry, dict) and set(entry) == M1314_ENTRY_KEYS,
            "exact sealed M1314 hammer entry is required")
    root = ROOT / safe_relative(entry["path"])
    require(root.parent == HW / "reviews", "M1314 hammer must be a direct reviews child")
    rows = _manifest_rows(root, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get("review.json") == entry["review_sha256"], "M1314 review SHA mismatch")
    review = strict_json(root / "review.json")
    require(set(review) == M1314_REVIEW_KEYS and review["schema"] == M1314_SCHEMA and
            review["status"] == M1314_STATUS, "M1314 review schema/status/shape mismatch")
    require(entry == M1314_ENTRY, "M1314 exact sealed entry mismatch")
    expected_identity = {
        "contract_path": M1313["contract"]["path"],
        "contract_sha256": M1313["contract"]["sha256"],
        "checker_path": M1313["checker"]["path"],
        "checker_sha256": M1313["checker"]["sha256"],
        "test_path": M1313["test"]["path"],
        "test_sha256": M1313["test"]["sha256"],
        "author_review_path": M1313["author_receipt"]["path"],
        "author_manifest_sha256": M1313["author_receipt"]["manifest_sha256"],
        "author_outer_file_sha256": M1313["author_receipt"]["outer_file_sha256"],
        "author_receipt_sha256": M1313["author_receipt"]["author_receipt_sha256"],
    }
    require(review["reviewed_identity"] == expected_identity,
            "M1314 reviewed M1313 identity mismatch")
    require(review["verdict"] == "GO_ROOT_AGENT_ONE_REMOTE_M1249_CAPTURE_ONLY",
            "M1314 verdict mismatch")
    require(review["independence"] == {"different_author": True},
            "M1314 independence mismatch")
    require(review["authorization"] == M1314_AUTHORIZATION,
            "M1314 authorization mismatch")
    require(review["claim_boundary"] == M1314_BOUNDARY,
            "M1314 claim boundary mismatch")
    require(review["hammer_execution"] == {
        "remote": False, "gpu": False, "eda": False, "production_capture": False},
        "M1314 hammer execution boundary mismatch")
    require(review["docs359_sha256"] ==
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "M1314 docs359 pin mismatch")
    return review


def expected_builder_identity() -> dict[str, str]:
    return {
        "source_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_sha256": sha256(Path(__file__).resolve()),
        "test_path": str(TEST.relative_to(ROOT)),
        "test_sha256": sha256(TEST),
        "source_contract_path": str(SOURCE_CONTRACT.relative_to(ROOT)),
        "source_contract_sha256": sha256(SOURCE_CONTRACT),
    }


def ensure_fresh() -> None:
    for path in (OUTPUT, ATTEMPT, PARTIAL):
        require(not os.path.lexists(str(path)), "pack namespace is not fresh: " + str(path))


def validate_release(value: dict[str, Any], release_path: Path) -> list[dict[str, Any]]:
    require(isinstance(value, dict) and set(value) == RELEASE_KEYS, "release keys mismatch")
    require(value["schema"] == RELEASE_SCHEMA and value["status"] == RELEASE_STATUS,
            "source-only or unhammered release cannot build")
    require(release_path.resolve() == RELEASE_PATH and
            value["contract_path"] == str(RELEASE_PATH.relative_to(ROOT)),
            "production release path mismatch")
    require(isinstance(value["builder_identity"], dict) and
            set(value["builder_identity"]) == BUILDER_KEYS and
            value["builder_identity"] == expected_builder_identity(),
            "builder identity mismatch")
    require(value["inventory"] == {
        "path": str(INVENTORY.relative_to(ROOT)), "sha256": INVENTORY_SHA256},
        "inventory identity mismatch")
    require(value["m1313"] == M1313, "M1313 binding mismatch")
    verify_m1314(value["m1314_hammer"])
    require(value["verifier"] == {
        "path": str(VERIFIER.relative_to(ROOT)), "sha256": VERIFIER_SHA256},
        "verifier identity mismatch")
    require(value["payload_manifest"] == {"path": str(PAYLOAD_MANIFEST.relative_to(ROOT))},
            "payload manifest identity mismatch")
    require(value["one_shot"] == {
        "attempt_marker": str(ATTEMPT.relative_to(ROOT)), "automatic_retry": False},
        "pack one-shot policy mismatch")
    require(value["output"] == {"path": str(OUTPUT.relative_to(ROOT)),
                                 "format": "deterministic_posix_tar"},
            "pack output mismatch")
    ensure_fresh()
    return exact_payload(release_path)


def _consume_attempt() -> None:
    descriptor = os.open(str(ATTEMPT), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        os.write(descriptor, ATTEMPT_TOKEN.encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tarinfo(archive: tarfile.TarFile, source: Path, arcname: str) -> tarfile.TarInfo:
    info = archive.gettarinfo(str(source), arcname=arcname)
    require(info.isfile(), "tar member is not a regular file")
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    info.mode = 0o444
    return info


def build_once(release: dict[str, Any], release_path: Path) -> Path:
    entries = validate_release(release, release_path)
    _consume_attempt()
    try:
        with tarfile.open(PARTIAL, mode="x", format=tarfile.PAX_FORMAT,
                          dereference=False) as archive:
            for row in entries:
                source = ROOT / row["path"]
                regular_exact(source, row["sha256"], row["size_bytes"], row["path"])
                with source.open("rb") as stream:
                    archive.addfile(_tarinfo(archive, source, row["path"]), stream)
        descriptor = os.open(str(PARTIAL), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(PARTIAL, OUTPUT)
    except Exception:
        if PARTIAL.exists() and PARTIAL.is_file() and not PARTIAL.is_symlink():
            PARTIAL.unlink()
        raise
    return OUTPUT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    release_path = args.release.resolve()
    release = strict_json(release_path)
    output = build_once(release, release_path)
    print(PASS_TOKEN + " " + str(output.relative_to(ROOT)) + " " + sha256(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
