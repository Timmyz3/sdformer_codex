#!/usr/bin/env python3
"""Read-only mechanical checks for the consumed M2166 startup failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNS = HW / "dc_handoff/runs"
QUARANTINE = RUNS / (
    "m2166_m2164_icc2_library_import_preflight_raw_r1_20260904."
    "failed_or_incomplete.2744243.quarantine"
)
ATTEMPT = RUNS / ".m2166_m2164_icc2_library_import_preflight_attempt_consumed"
CANONICAL = RUNS / "m2166_m2164_icc2_library_import_preflight_raw_r1_20260904"
RUNNER = HW / "dc_handoff/scripts/run_m2164_m2154_icc2_library_import_preflight_one_shot.sh"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PRIOR_COLLATERAL = HW.parent / "icc2_output.txt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition: bool, label: str) -> None:
    if not condition:
        raise SystemExit(f"M2167_FAIL {label}")
    print(f"PASS {label}")


def verify_seal(directory: Path) -> None:
    need(directory.is_dir() and not directory.is_symlink(), f"dir_{directory.name}")
    need(not any(node.is_symlink() for node in directory.rglob("*")), f"no_symlink_{directory.name}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and outer.is_file(), f"seal_files_{directory.name}")
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        need(name not in listed, f"unique_manifest_name_{directory.name}_{name}")
        listed[name] = digest
    actual = sorted(
        str(node.relative_to(directory))
        for node in directory.rglob("*")
        if node.is_file() and node.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    )
    need(sorted(listed) == actual, f"exhaustive_manifest_{directory.name}")
    for name, digest in listed.items():
        need(sha(directory / name) == digest, f"inner_sha_{directory.name}_{name}")
    outer_digest, outer_name = outer.read_text().split()
    need(outer_name == "SHA256SUMS", f"outer_name_{directory.name}")
    need(sha(manifest) == outer_digest, f"outer_sha_{directory.name}")


need(sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4", "docs359")
need(not CANONICAL.exists(), "no_canonical_m2166_result")
need(len(list(RUNS.glob(".m2166_m2164_icc2_library_import_preflight_attempt_consumed"))) == 1, "one_attempt_marker")
need(len(list(RUNS.glob("m2166_m2164_icc2_library_import_preflight_raw_r1_20260904.failed_or_incomplete.*.quarantine"))) == 1, "one_quarantine")
verify_seal(ATTEMPT)
verify_seal(QUARANTINE)

attempt_text = (ATTEMPT / "ATTEMPT_CONSUMED.txt").read_text()
need(
    attempt_text
    == "status=M2166_ATTEMPT_CONSUMED\nlicense_queries=1\n"
    "top_level_icc2_shell_runs=1\npnr_runs=0\nretry=false\n",
    "attempt_marker_exact",
)
failure_text = (QUARANTINE / "RUN_FAILED_OR_INCOMPLETE.txt").read_text()
need(
    failure_text == "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nexit_code=1\nretry=false\n",
    "failure_marker_exact",
)

expected_files = {
    "RUN_FAILED_OR_INCOMPLETE.txt",
    "SHA256SUMS",
    "SHA256SUMS.seal.sha256",
    "prior_m2135_collateral/icc2_output.txt",
    "repo_root_before.json",
    "repo_root_before.log",
}
actual_files = {
    str(node.relative_to(QUARANTINE)) for node in QUARANTINE.rglob("*") if node.is_file()
}
need(actual_files == expected_files, "exact_quarantine_file_set")

isolated = QUARANTINE / "isolated_cwd"
need((isolated / "home").is_dir(), "home_created")
need((isolated / "tmp").is_dir(), "tmp_created")
need(not (isolated / "cache").exists(), "cache_parent_absent")
need(not (isolated / "cache/xdg").exists(), "xdg_child_absent")
need(not (isolated / "cache/library").exists(), "library_child_absent")

post_mkdir_outputs = [
    "execution_contract.json",
    "license_preflight.log",
    "icc2_preflight.log",
    "icc2_preflight.rc",
    "process_tree.json",
    "process_monitor.log",
    "process_monitor.ready",
    "launch.gate",
    "receipt.json",
    "checker.log",
    "RUN_COMPLETE.txt",
]
need(not any((QUARANTINE / name).exists() for name in post_mkdir_outputs), "no_post_mkdir_output")

runner = RUNNER.read_text()
mkdir_token = 'mkdir -- "${ISOLATED}/home" "${ISOLATED}/tmp" "${ISOLATED}/cache/xdg" "${ISOLATED}/cache/library"'
contract_token = '"${WORK}/execution_contract.json"'
license_token = '"${LMUTIL}" lmstat'
launch_token = '"${ICC2}" -no_init -f "${TCL}"'
for token in (mkdir_token, contract_token, license_token, launch_token):
    need(runner.count(token) >= 1, f"runner_token_{hashlib.sha256(token.encode()).hexdigest()[:12]}")
mkdir_at = runner.index(mkdir_token)
contract_at = runner.index(contract_token, mkdir_at)
license_at = runner.index(license_token, contract_at)
launch_at = runner.index(launch_token, license_at)
need(mkdir_at < contract_at < license_at < launch_at, "startup_order_mkdir_contract_license_icc2")

snapshot = json.loads((QUARANTINE / "repo_root_before.json").read_text())
need(snapshot["schema"] == "m2153_repo_root_inventory_r1_v1", "root_snapshot_schema")
need(snapshot["root"] == str(HW.parent), "root_snapshot_root")
need(snapshot["node_count"] == len(snapshot["nodes"]) == 293, "root_snapshot_count")
names = [row["name"] for row in snapshot["nodes"]]
need(len(names) == len(set(names)), "root_snapshot_unique_names")
need({row["node_type"] for row in snapshot["nodes"]} == {"regular", "directory"}, "root_snapshot_node_types")
need(
    (QUARANTINE / "repo_root_before.log").read_text()
    == "PASS_M2153_REPO_ROOT_ALL_NODE_INVENTORY\nnode_count=293\n",
    "root_snapshot_log",
)

copied = QUARANTINE / "prior_m2135_collateral/icc2_output.txt"
need(sha(copied) == "0410c14052c0b18c0f1a92246ecec4f109a9e37130b8f95f5cb4587cbcf863d6", "copied_prior_collateral")
need(PRIOR_COLLATERAL.is_file(), "root_prior_collateral_present")
need(sha(PRIOR_COLLATERAL) == sha(copied), "root_prior_collateral_unchanged")

print("PASS_M2167_M2166_STARTUP_FAILURE_DIAGNOSIS")
print("reserved_attempts=1")
print("observed_license_queries=0")
print("observed_top_level_icc2_shell_runs=0")
print("observed_pnr_runs=0")
print("m2166_permanently_noncitable=true")
