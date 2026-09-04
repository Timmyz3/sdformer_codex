#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Additive lexical-path successor to failed M1351.

The sole semantic change is path validation: every lexical component from the
workspace root through the candidate leaf is inspected with lstat before any
resolve call, and resolved containment is then proved independently.  The
production allowlist remains empty; this source cannot emit a production row.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterator


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
HW = REPO / "hw_autoresearch_nts07"
M1351_PATH = HW / "system_simulator/scripts/build_m1351_ep34_table_a_memory_timed_authority_compiler.py"
M1351_SHA256 = "eac9e57a93c5c13346dcf236a7e1eb78c1319dcebf20f69793278f900d794215"
M1351_TEST = HW / "system_simulator/tests/test_m1351_ep34_table_a_memory_timed_authority_compiler.py"
M1351_TEST_SHA256 = "dd8af408d97a48cb0c974ccd57cb1f0cf959b7444de7c15342dd5561e34a7d71"
M1351_CONTRACT = HW / "contracts/m1351_ep34_table_a_memory_timed_authority_compiler_source_contract_r1_20260831.json"
M1351_CONTRACT_SHA256 = "2fbfb750929a53055af70ce17fb0ac63d3cf72a609eda45a283fc68a156e0be1"
M1353_FAIL = HW / "reviews/m1353_m1351_ep34_table_a_memory_timed_authority_compiler_source_blind_hammer_r1_20260831"
M1353_REVIEW_SHA256 = "42313bad5e126717de802ee4b88fe02fe800db75b0b0047f9ed493b6efe26d05"
M1353_MANIFEST_SHA256 = "3583f866e6d94de8b46ef0a23ae4fdbaf4d99009c43340d80ef7012d2adc87e1"
M1353_OUTER_SHA256 = "2ee31718569d878ed4b5d82d77e756bdba369bf32b383ff4cc07e960aa892c61"
TEST = HW / "system_simulator/tests/test_m1354_ep34_table_a_lexical_path_authority_compiler.py"
CONTRACT = HW / "contracts/m1354_ep34_table_a_lexical_path_authority_compiler_source_contract_r1_20260831.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOURCE_SCHEMA = "m1354.ep34.table_a.lexical_path.authority.compiler.source.r1"
SOURCE_STATUS = "SOURCE_ONLY_UNPOPULATED__FRESH_DIFFERENT_AUTHOR_HAMMER_REQUIRED"


class CompileError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise CompileError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise CompileError("missing " + label) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be regular non-symlink")
    require(sha256(path) == expected, label + " SHA drift")


regular_exact(M1351_PATH, M1351_SHA256, "M1351 source")
SPEC = importlib.util.spec_from_file_location("m1354_frozen_m1351", M1351_PATH)
require(SPEC is not None and SPEC.loader is not None, "M1351 import spec")
M1351 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M1351
SPEC.loader.exec_module(M1351)


def strict_json(path: Path) -> Any:
    return M1351.strict_json(path)


def verify_recursive_seal(root: Path, expected_review: str,
                          expected_manifest: str, expected_outer: str) -> None:
    require(root.is_dir() and not root.is_symlink(), "sealed root invalid")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular_exact(manifest, expected_manifest, "M1353 manifest")
    regular_exact(outer, expected_outer, "M1353 outer seal")
    require(outer.read_text().split() == [expected_manifest, "SHA256SUMS"],
            "M1353 outer semantics drift")
    rows: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        member = Path(name)
        require(not member.is_absolute() and ".." not in member.parts and
                name not in rows, "unsafe M1353 manifest member")
        path = root / member
        regular_exact(path, digest, "M1353 member " + name)
        rows[name] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "M1353 recursive population drift")
    require(rows.get("review.json") == expected_review,
            "M1353 review member drift")


def lexical_lstat_then_resolved_containment(root: Path, path: Path,
                                            leaf_must_exist: bool = True) -> Path:
    """Reject lexical symlinks first, then independently prove containment."""
    raw_root = Path(root)
    root_abs = Path(os.path.abspath(os.fspath(raw_root)))
    try:
        root_mode = root_abs.lstat().st_mode
    except OSError as exc:
        raise CompileError("workspace root missing") from exc
    require(stat.S_ISDIR(root_mode) and not stat.S_ISLNK(root_mode),
            "workspace root must be a real directory")
    try:
        root_resolved = root_abs.resolve(strict=True)
    except OSError as exc:
        raise CompileError("workspace root resolution failed") from exc
    require(root_resolved == root_abs,
            "workspace root must not traverse symlinks")

    raw = Path(path)
    require(".." not in raw.parts, "parent traversal forbidden: %s" % path)
    candidate_lexical = raw if raw.is_absolute() else root_abs / raw
    candidate_abs = Path(os.path.abspath(os.fspath(candidate_lexical)))
    try:
        relative = candidate_abs.relative_to(root_abs)
    except ValueError as exc:
        raise CompileError("lexical path escapes workspace: %s" % path) from exc

    current = root_abs
    for index, part in enumerate(relative.parts):
        current = current / part
        is_leaf = index == len(relative.parts) - 1
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            if is_leaf and not leaf_must_exist:
                continue
            raise CompileError("lexical path component missing: %s" % current) from exc
        require(not stat.S_ISLNK(mode),
                "symlink lexical component forbidden: %s" % current)
        if not is_leaf:
            require(stat.S_ISDIR(mode),
                    "non-directory lexical ancestor: %s" % current)

    try:
        resolved = candidate_abs.resolve(strict=leaf_must_exist)
    except OSError as exc:
        raise CompileError("path resolution failed: %s" % path) from exc
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise CompileError("resolved path escapes workspace: %s" % path) from exc
    return resolved


@contextlib.contextmanager
def patched_path_validator() -> Iterator[None]:
    original_m1351 = M1351.secure_no_symlink_ancestry
    original_m1342 = M1351.M1342.no_symlink_ancestry
    M1351.secure_no_symlink_ancestry = lexical_lstat_then_resolved_containment
    M1351.M1342.no_symlink_ancestry = lexical_lstat_then_resolved_containment
    try:
        yield
    finally:
        M1351.secure_no_symlink_ancestry = original_m1351
        M1351.M1342.no_symlink_ancestry = original_m1342


def build(config_path: Path, workspace_root: Path,
          fixture_allowlist: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    with patched_path_validator():
        result = M1351.build(config_path, workspace_root, fixture_allowlist)
    require(result.get("status") == "PASS_SOURCE_FIXTURE_MEMORY_TIMED_NOT_PRODUCTION",
            "M1354 cannot admit production")
    result = dict(result)
    result["schema"] = "m1354.ep34.table_a.lexical_path.authority.output.r1"
    result["status"] = "PASS_SOURCE_FIXTURE_LEXICAL_PATH_NOT_PRODUCTION"
    result["m1354_source_sha256"] = sha256(SCRIPT)
    result["claim_boundary"] = {**result["claim_boundary"],
        "lexical_lstat_including_leaf": True,
        "resolved_containment_separate": True,
        "production_rows": 0,
        "paper_headline_admitted": False}
    return result


def validate_source_policy() -> dict[str, Any]:
    for path, digest, label in (
        (M1351_TEST, M1351_TEST_SHA256, "M1351 test"),
        (M1351_CONTRACT, M1351_CONTRACT_SHA256, "M1351 contract"),
        (DOCS359, DOCS359_SHA256, "protected docs359"),
    ):
        regular_exact(path, digest, label)
    verify_recursive_seal(M1353_FAIL, M1353_REVIEW_SHA256,
                          M1353_MANIFEST_SHA256, M1353_OUTER_SHA256)
    failed = strict_json(M1353_FAIL / "review.json")
    require(failed.get("status") == "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED"
            and failed.get("fresh_hammer", {}).get("false_negative_count") == 1
            and failed.get("fresh_hammer", {}).get("false_negatives") ==
            ["symlink_config_escape"], "M1353 failure semantics drift")
    require(M1351.M1342.PRODUCTION_AUTHORITY_ALLOWLIST == {},
            "production authority allowlist must remain empty")
    contract = strict_json(CONTRACT)
    require(contract.get("schema") == SOURCE_SCHEMA and
            contract.get("status") == SOURCE_STATUS and
            contract.get("production_authority_allowlist_entries") == 0 and
            contract.get("production_authorized") is False,
            "M1354 source policy drift")
    require(contract.get("source") == {
        "path": str(SCRIPT.relative_to(REPO)), "sha256": sha256(SCRIPT)},
        "M1354 source identity drift")
    require(contract.get("test") == {
        "path": str(TEST.relative_to(REPO)), "sha256": sha256(TEST)},
        "M1354 test identity drift")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args()
    try:
        require(args.source_self_check, "author stage permits source self-check only")
        validate_source_policy()
        print("PASS_M1354_SOURCE_SELF_CHECK__NO_PRODUCTION_NO_TABLE_A_NO_EDA")
        return 0
    except (CompileError, M1351.CompileError, M1351.M1342.CompileError,
            M1351.M.CompileError, OSError, ValueError) as exc:
        print("M1354_FAIL_CLOSED: %s" % exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
