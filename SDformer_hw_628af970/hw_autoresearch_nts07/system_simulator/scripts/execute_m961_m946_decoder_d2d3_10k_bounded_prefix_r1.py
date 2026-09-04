#!/usr/bin/env python3
"""M961 one-attempt D2/D3 10K bounded-prefix execution driver.

The driver is source-only until a separately sealed M969 release and M970
release hammer are supplied.  A released run may execute exactly two rows:
D2/sample0/A1_OSG/t0/10K and D3/sample0/A1_OSG/t0/10K.  It may only recommend
a separate 100K scheduler-prefix release.  It never runs 100K or a full row.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
PYTHON_PATH = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
PYTHON_VERSION = (3, 10, 18)
M946_PATH = HERE / "analyze_m946_decoder_multilayer_bounded_prefix_source_candidate.py"
M946_SHA256 = "0ffd1ee810f24d1a95b0df33ffe8eae43240920e12a2fccb86c947d2be51b6ac"
SOURCE_CONTRACT = HW / "contracts/m961_m946_decoder_d2d3_10k_bounded_prefix_source_contract_r1_20260829.json"
FUTURE_RELEASE = HW / "contracts/m969_m961_decoder_d2d3_10k_bounded_prefix_release_r1_20260829.json"
SOURCE_HAMMER = HW / "reviews/m968_m961_decoder_d2d3_10k_bounded_prefix_source_hammer_r1_20260829"
RELEASE_HAMMER = HW / "reviews/m970_m969_m961_decoder_d2d3_10k_bounded_prefix_release_hammer_r1_20260829"
M950_DIR = HW / "reviews/m950_m946_decoder_multilayer_bounded_prefix_source_fresh_hammer_r1_20260829"
M950_IDENTITY = (
    "2042b1d2f16a29be706a4c413ce3d473b7daedd56cca24dfd6aff57848579cf6",
    "8f749a2f9db1aa49d710765e3d89232b57029d3ed313f2da5299f0dfa3910ee7",
    "389bae76312b4f51655facdb56d6754c3bb6e93821c02b52b68a0f9f84b19e09",
)
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOURCE_SCHEMA = "m961_m946_decoder_d2d3_10k_bounded_prefix_source_contract_v1"
RELEASE_SCHEMA = "m969_m961_decoder_d2d3_10k_bounded_prefix_release_v1"
RESULT = HW / "results/m961_m946_decoder_d2d3_10k_bounded_prefix_r1_20260829"
ATTEMPT = HW / "results/.m961_m946_decoder_d2d3_10k_bounded_prefix_r1_attempt_consumed"
FAILURE_PREFIX = "m961_m946_decoder_d2d3_10k_bounded_prefix_r1_20260829.failed_or_incomplete."
PREFIX_10K = 10000
PREFIX_100K = 100000
FUTURE_100K_TIMEOUT_CAP_SECONDS = 1800
SOURCE_FETCH_REQUESTS = {"D2": 231600, "D3": 465600}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def strict_json(path: Path):
    def reject(value):
        raise ValueError("duplicate JSON key: " + value)
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=lambda pairs: _pairs(pairs),
                         parse_constant=reject)


def _pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key: " + key)
        output[key] = value
    return output


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _validate_interpreter() -> Dict[str, object]:
    executable = Path(sys.executable).resolve()
    require(executable == PYTHON_PATH and sha256(executable) == PYTHON_SHA256,
            "M961 requires exact frozen M925/M946 Python")
    require(tuple(sys.version_info[:3]) == PYTHON_VERSION,
            "M961 Python version identity drift")
    return {"path": str(executable), "sha256": PYTHON_SHA256,
            "version": list(PYTHON_VERSION)}


_validate_interpreter()


def _load_m946():
    require(M946_PATH.is_file() and not M946_PATH.is_symlink() and
            sha256(M946_PATH) == M946_SHA256, "M946 source identity drift")
    spec = importlib.util.spec_from_file_location("m961_frozen_m946", M946_PATH)
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M946")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M946 = _load_m946()


def _regular_exact(path: Path, expected: str, name: str) -> None:
    require(path.is_file() and not path.is_symlink() and sha256(path) == expected,
            name + " identity drift")


def _verify_sealed(directory: Path, identity: Tuple[str, str, str],
                   name: str) -> Dict[str, str]:
    sealed = M946.M785.verify_sealed_directory(directory)
    require(sha256(directory / "review.json") == identity[0] and
            sealed["manifest_sha256"] == identity[1] and
            sealed["outer_seal_file_sha256"] == identity[2],
            name + " recursive identity drift")
    return sealed


def canonical_paths() -> Dict[str, str]:
    return {
        "source_contract": str(SOURCE_CONTRACT.relative_to(REPO)),
        "release": str(FUTURE_RELEASE.relative_to(REPO)),
        "source_hammer": str(SOURCE_HAMMER.relative_to(REPO)),
        "release_hammer": str(RELEASE_HAMMER.relative_to(REPO)),
        "result": str(RESULT.relative_to(REPO)),
        "attempt": str(ATTEMPT.relative_to(REPO)),
        "failure_prefix": "hw_autoresearch_nts07/results/" + FAILURE_PREFIX,
    }


def validate_source_contract(contract_path: Path, runner_path: Path,
                             *, require_fresh_namespace: bool = True
                             ) -> Dict[str, object]:
    _validate_interpreter()
    require(Path(contract_path).resolve() == SOURCE_CONTRACT.resolve(),
            "M961 source contract canonical path drift")
    contract = strict_json(contract_path)
    require(contract.get("schema") == SOURCE_SCHEMA and
            contract.get("status") ==
            "SOURCE_ONLY__SEPARATE_RELEASE_AND_RELEASE_HAMMER_REQUIRED" and
            contract.get("launch_now") is False,
            "M961 source contract authority drift")
    require(contract.get("canonical") == canonical_paths(),
            "M961 canonical path contract drift")
    interpreter = contract.get("required_interpreter", {})
    require(interpreter == {"path": str(PYTHON_PATH),
                            "sha256": PYTHON_SHA256,
                            "version": "3.10.18"},
            "M961 interpreter contract drift")
    auth = contract.get("authorization", {})
    require(auth.get("future_d2_10k") is True and
            auth.get("future_d3_10k") is True and
            all(auth.get(key) is False for key in
                ("execute_now", "d1_prefix", "d2_or_d3_100k",
                 "full_row", "production", "eda_gpu_remote")),
            "M961 source authorization is not fail closed")
    boundary = contract.get("claim_boundary", {})
    require(all(boundary.get(key) is False for key in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "full_row_authorized")),
            "M961 claim boundary drift")
    for name, row in contract["source_identity"].items():
        path = HW / row["path"]
        _regular_exact(path, row["sha256"], name)
    runner = Path(runner_path)
    require(runner.resolve() ==
            (HW / contract["source_identity"]["m961_runner"]["path"]).resolve(),
            "M961 runner canonical path drift")
    _verify_sealed(M950_DIR, M950_IDENTITY, "M950")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    if require_fresh_namespace:
        require(not RESULT.exists() and not RESULT.is_symlink() and
                not ATTEMPT.exists() and not ATTEMPT.is_symlink(),
                "M961 result/attempt namespace is not fresh")
    return {
        "status": "PASS_M961_SOURCE_CONTRACT__NO_EXECUTION_AUTHORIZED",
        "source_contract_sha256": sha256(contract_path),
        "runner_sha256": sha256(runner),
        "interpreter": _validate_interpreter(),
        "m950_review_sha256": M950_IDENTITY[0],
        "result_absent": not RESULT.exists() and not RESULT.is_symlink(),
        "attempt_absent": not ATTEMPT.exists() and not ATTEMPT.is_symlink(),
    }


def validate_release(release_path: Path, runner_path: Path,
                     expected_release_sha256: str,
                     release_hammer_dir: Path,
                     release_hammer_identity: Tuple[str, str, str]
                     ) -> Dict[str, object]:
    source = validate_source_contract(
        SOURCE_CONTRACT, runner_path, require_fresh_namespace=False)
    require(Path(release_path).resolve() == FUTURE_RELEASE.resolve(),
            "M961 release canonical path drift")
    _regular_exact(release_path, expected_release_sha256, "M969 release")
    release = strict_json(release_path)
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") ==
            "AUTHORIZE_ONE_D2_D3_10K_BOUNDED_PREFIX_PAIR" and
            release.get("release") is True and
            release.get("launch_now") is False and
            release.get("max_attempts") == 1,
            "M969 release authority drift")
    require(release.get("canonical") == canonical_paths(),
            "M969 release canonical path drift")
    exact_rows = release.get("exact_rows")
    require(exact_rows == [
        {"layer": "D2", "sample_id": 0, "config": "A1_OSG",
         "timestep": 0, "expanded_prefix": PREFIX_10K,
         "route": "EXACT_BINARY_SUPPORT"},
        {"layer": "D3", "sample_id": 0, "config": "A1_OSG",
         "timestep": 0, "expanded_prefix": PREFIX_10K,
         "route": "EXACT_BINARY_SUPPORT"}],
            "M969 exact row set drift")
    auth = release.get("authorization", {})
    require(auth.get("consume_one_attempt") is True and
            auth.get("run_exact_d2_d3_10k_pair") is True and
            all(auth.get(key) is False for key in
                ("d1_prefix", "d2_or_d3_100k", "automatic_100k",
                 "full_row", "production", "eda_gpu_remote")),
            "M969 release expands authorization")
    binding = release.get("source_binding", {})
    require(binding.get("m961_source_contract_sha256") ==
            source["source_contract_sha256"] and
            binding.get("m961_runner_sha256") == sha256(Path(runner_path)) and
            binding.get("m961_driver_sha256") == sha256(Path(__file__)) and
            binding.get("m946_source_sha256") == M946_SHA256 and
            binding.get("m950_review_sha256") == M950_IDENTITY[0],
            "M969 source binding drift")
    require(Path(release_hammer_dir).resolve() == RELEASE_HAMMER.resolve(),
            "M970 release hammer path drift")
    hammer = _verify_sealed(release_hammer_dir,
                            release_hammer_identity, "M970")
    review = strict_json(release_hammer_dir / "review.json")
    require(review.get("status") ==
            "PASS_M970_M969_M961_BOUNDED_PREFIX_RELEASE_HAMMER" and
            review.get("verdict") == "GO_ONE_D2_D3_10K_PAIR_ONLY" and
            review.get("release_sha256") == expected_release_sha256 and
            review.get("source_binding", {}).get("m961_driver_sha256") ==
            sha256(Path(__file__)),
            "M970 release hammer authority drift")
    return {
        "status": "PASS_M961_RELEASE_AUTHORITY__ONE_10K_PAIR_ONLY",
        "release_sha256": expected_release_sha256,
        "release_hammer_review_sha256": release_hammer_identity[0],
        "release_hammer_manifest_sha256": hammer["manifest_sha256"],
        "release_hammer_outer_sha256": hammer["outer_seal_file_sha256"],
    }


def _safe_stage(path: Path, prefix: str) -> None:
    path = Path(path)
    require(path.parent.resolve() == RESULT.parent.resolve(),
            "M961 stage parent drift")
    require(path.name.startswith(prefix) and "/" not in path.name and
            path.name not in (".", ".."), "unsafe M961 stage basename")


def _write_exclusive(path: Path, data: bytes) -> None:
    path = Path(path)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def seal_directory(directory: Path, members: Sequence[str]) -> Dict[str, str]:
    directory = Path(directory)
    lines = []
    for name in sorted(members):
        member = directory / name
        require(member.is_file() and not member.is_symlink(),
                "missing result member: " + name)
        lines.append(sha256(member) + "  " + name)
    manifest = ("\n".join(lines) + "\n").encode("utf-8")
    _write_exclusive(directory / "SHA256SUMS", manifest)
    outer = (sha256(directory / "SHA256SUMS") +
             "  SHA256SUMS\n").encode("utf-8")
    _write_exclusive(directory / "SHA256SUMS.seal.sha256", outer)
    return M946.M785.verify_sealed_directory(directory)


def consume_attempt(release_path: Path, runner_path: Path,
                    expected_release_sha256: str,
                    release_hammer_dir: Path,
                    release_hammer_identity: Tuple[str, str, str],
                    stage: Path) -> Dict[str, object]:
    authority = validate_release(
        release_path, runner_path, expected_release_sha256,
        release_hammer_dir, release_hammer_identity)
    _safe_stage(stage, ATTEMPT.name + ".stage.")
    require(not stage.exists() and not stage.is_symlink() and
            not ATTEMPT.exists() and not ATTEMPT.is_symlink(),
            "M961 attempt namespace collision")
    stage.mkdir(mode=0o700)
    receipt = {
        "schema": "m961_decoder_d2d3_10k_attempt_v1",
        "status": "CONSUMED_BEFORE_EXECUTION",
        "max_attempts": 1,
        "release_sha256": expected_release_sha256,
        "release_hammer_review_sha256": release_hammer_identity[0],
        "result_path": str(RESULT.relative_to(REPO)),
        "attempt_path": str(ATTEMPT.relative_to(REPO)),
        "full_row_authorized": False,
        "d2_or_d3_100k_authorized": False,
    }
    _write_exclusive(stage / "attempt.json", (json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False) +
        "\n").encode("utf-8"))
    seal_directory(stage, ("attempt.json",))
    os.rename(stage, ATTEMPT)
    sealed = M946.M785.verify_sealed_directory(ATTEMPT)
    return {"authority": authority, "attempt": receipt,
            "sealed_identity": sealed}


def validate_attempt(release_path: Path, runner_path: Path,
                     expected_release_sha256: str,
                     release_hammer_dir: Path,
                     release_hammer_identity: Tuple[str, str, str]
                     ) -> Dict[str, object]:
    authority = validate_release(
        release_path, runner_path, expected_release_sha256,
        release_hammer_dir, release_hammer_identity)
    sealed = M946.M785.verify_sealed_directory(ATTEMPT)
    receipt = strict_json(ATTEMPT / "attempt.json")
    require(receipt.get("schema") == "m961_decoder_d2d3_10k_attempt_v1" and
            receipt.get("status") == "CONSUMED_BEFORE_EXECUTION" and
            receipt.get("release_sha256") == expected_release_sha256 and
            receipt.get("release_hammer_review_sha256") ==
            release_hammer_identity[0] and
            receipt.get("full_row_authorized") is False and
            receipt.get("d2_or_d3_100k_authorized") is False,
            "M961 consumed attempt drift")
    return {"authority": authority, "attempt": receipt,
            "sealed_identity": sealed}


def _meminfo() -> Dict[str, int]:
    output = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        output[key] = int(value.strip().split()[0]) * 1024
    require(all(key in output for key in
                ("MemAvailable", "CommitLimit", "Committed_AS")),
            "M961 meminfo fields missing")
    return output


def project_100k(row: Mapping[str, object]) -> Dict[str, object]:
    identity = row["row_identity"]
    layer = str(identity["layer"])
    require(layer in ("D2", "D3") and
            identity["numerical_route"] == "EXACT_BINARY_SUPPORT" and
            int(row["prefix"]) == PREFIX_10K,
            "M961 10K row identity drift")
    exact = row["exact_miter"]
    require(exact["status"] == "PASS_M768_M861_M890_M896_EXACT_MITER" and
            int(exact["expanded_request_count"]) == PREFIX_10K and
            int(exact["compressed_transaction_count"]) == 1 and
            int(exact["commit_requests_in_prefix"]) == 0,
            "M961 10K exact/source-fetch boundary drift")
    state = int(exact["combined_live_event_state_bytes"])
    measured_peak = int(row["process_max_rss_kib"]) * 1024
    nonscheduler = max(0, measured_peak - state)
    projected_state = state * (PREFIX_100K // PREFIX_10K)
    projected_peak = nonscheduler + projected_state
    two_x_memory = 2 * projected_peak
    projected_elapsed = float(row["elapsed_seconds"]) * 10.0
    two_x_timeout = math.ceil(2.0 * projected_elapsed)
    mem = _meminfo()
    commit_headroom = max(0, mem["CommitLimit"] - mem["Committed_AS"])
    memory_pass = (two_x_memory <= mem["MemAvailable"] and
                   two_x_memory <= commit_headroom)
    timeout_pass = two_x_timeout <= FUTURE_100K_TIMEOUT_CAP_SECONDS
    source_fetch_count = SOURCE_FETCH_REQUESTS[layer]
    require(PREFIX_100K < source_fetch_count,
            "100K unexpectedly crosses frozen source-fetch transaction")
    return {
        "layer": layer,
        "measured_10k_exact": True,
        "measured_10k_transaction_scope": "SOURCE_FETCH_ONLY",
        "source_fetch_full_request_count": source_fetch_count,
        "future_100k_stays_inside_source_fetch": True,
        "future_100k_contributor_mapper_covered": False,
        "future_100k_commit_covered": False,
        "projected_100k_elapsed_seconds": projected_elapsed,
        "two_x_100k_timeout_seconds": two_x_timeout,
        "timeout_cap_seconds": FUTURE_100K_TIMEOUT_CAP_SECONDS,
        "two_x_timeout_gate_pass": timeout_pass,
        "projected_100k_scheduler_state_bytes": projected_state,
        "projected_100k_peak_bytes": projected_peak,
        "two_x_projected_100k_memory_bytes": two_x_memory,
        "mem_available_bytes": mem["MemAvailable"],
        "commit_headroom_bytes": commit_headroom,
        "two_x_memory_gate_pass": memory_pass,
        "recommend_separate_100k_scheduler_prefix_release":
            bool(memory_pass and timeout_pass),
        "automatic_100k_authorized": False,
        "full_row_authorized": False,
    }


def run_exact_pair(release_path: Path, runner_path: Path,
                   expected_release_sha256: str,
                   release_hammer_dir: Path,
                   release_hammer_identity: Tuple[str, str, str],
                   output_stage: Path) -> Dict[str, object]:
    attempt = validate_attempt(
        release_path, runner_path, expected_release_sha256,
        release_hammer_dir, release_hammer_identity)
    _safe_stage(output_stage, RESULT.name + ".stage.")
    require(not RESULT.exists() and not RESULT.is_symlink() and
            not output_stage.exists() and not output_stage.is_symlink(),
            "M961 result namespace collision")
    started = time.monotonic()
    rows = [M946.run_bounded_prefix(
                layer, 0, "A1_OSG", 0, PREFIX_10K)
            for layer in ("D2", "D3")]
    projections = [project_100k(row) for row in rows]
    recommend = all(row[
        "recommend_separate_100k_scheduler_prefix_release"]
                    for row in projections)
    result = {
        "schema": "m961_decoder_d2d3_10k_bounded_prefix_result_v1",
        "status": "PASS_M961_D2_D3_10K_EXACT_PAIR__NO_100K_OR_FULL_ROW",
        "release_sha256": expected_release_sha256,
        "release_hammer_review_sha256": release_hammer_identity[0],
        "rows": rows,
        "future_100k_projections": projections,
        "next_action": (
            "GO_TO_SEPARATE_100K_SCHEDULER_PREFIX_RELEASE_DESIGN" if recommend
            else "STOP_100K_AND_REPAIR_BOUNDED_SCHEDULER_SCALABILITY"),
        "elapsed_pair_seconds": time.monotonic() - started,
        "transaction_scope": {
            "d2_10k": "SOURCE_FETCH_ONLY",
            "d3_10k": "SOURCE_FETCH_ONLY",
            "d2_d3_contributor_mapper_covered": False,
            "d2_d3_commit_covered": False,
        },
        "claim_boundary": {
            "fresh_result_hammer_required": True,
            "paper_citable": False,
            "production_row": False,
            "d2_or_d3_100k_executed": False,
            "automatic_100k_authorized": False,
            "full_row_authorized": False,
            "decoder_complete": False,
            "table_a_row": False,
            "system_speedup": False,
            "eda_gpu_remote_used": False,
        },
        "attempt_identity": attempt["sealed_identity"],
    }
    output_stage.mkdir(mode=0o700)
    _write_exclusive(output_stage / "result.json", (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) +
        "\n").encode("utf-8"))
    _write_exclusive(output_stage / "RUN_COMPLETE.txt",
                     b"M961_D2_D3_10K_PAIR_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED\n")
    sealed = seal_directory(output_stage, ("RUN_COMPLETE.txt", "result.json"))
    result["stage_sealed_identity"] = sealed
    return result


def publish_no_replace(stage: Path, expected_release_sha256: str,
                       expected_release_hammer_review_sha256: str
                       ) -> Dict[str, object]:
    _safe_stage(stage, RESULT.name + ".stage.")
    require(stage.is_dir() and not stage.is_symlink() and
            not RESULT.exists() and not RESULT.is_symlink(),
            "M961 publish namespace drift")
    sealed = M946.M785.verify_sealed_directory(stage)
    result = strict_json(stage / "result.json")
    require(result.get("schema") ==
            "m961_decoder_d2d3_10k_bounded_prefix_result_v1" and
            result.get("status") ==
            "PASS_M961_D2_D3_10K_EXACT_PAIR__NO_100K_OR_FULL_ROW" and
            result.get("release_sha256") == expected_release_sha256 and
            result.get("release_hammer_review_sha256") ==
            expected_release_hammer_review_sha256 and
            result.get("claim_boundary", {}).get("full_row_authorized") is
            False and
            result.get("claim_boundary", {}).get(
                "automatic_100k_authorized") is False,
            "M961 staged result authority drift")
    os.rename(stage, RESULT)
    require(M946.M785.verify_sealed_directory(RESULT) == sealed,
            "M961 published result identity drift")
    return {"status": "PASS_M961_NO_REPLACE_PUBLICATION",
            "result": str(RESULT.relative_to(REPO)),
            "sealed_identity": sealed}


def source_self_test() -> Dict[str, object]:
    fake = {
        "row_identity": {"layer": "D2", "numerical_route":
                         "EXACT_BINARY_SUPPORT"},
        "prefix": PREFIX_10K,
        "elapsed_seconds": 1.0,
        "process_max_rss_kib": 1024,
        "exact_miter": {
            "status": "PASS_M768_M861_M890_M896_EXACT_MITER",
            "expanded_request_count": PREFIX_10K,
            "compressed_transaction_count": 1,
            "commit_requests_in_prefix": 0,
            "combined_live_event_state_bytes": 4096,
        },
    }
    projection = project_100k(fake)
    require(projection["future_100k_stays_inside_source_fetch"] is True and
            projection["future_100k_contributor_mapper_covered"] is False and
            projection["automatic_100k_authorized"] is False,
            "M961 directed projection boundary drift")
    return {
        "status": "PASS_M961_SOURCE_SELF_TEST__NO_PREFIX_EXECUTED",
        "interpreter": _validate_interpreter(),
        "projection": projection,
        "result_absent": not RESULT.exists() and not RESULT.is_symlink(),
        "attempt_absent": not ATTEMPT.exists() and not ATTEMPT.is_symlink(),
        "prefix_executed": False,
        "full_row_authorized": False,
    }


def _hammer_identity(args) -> Tuple[str, str, str]:
    require(args.expected_release_hammer_review_sha256 and
            args.expected_release_hammer_manifest_sha256 and
            args.expected_release_hammer_outer_sha256,
            "M970 release hammer identity is required")
    return (args.expected_release_hammer_review_sha256,
            args.expected_release_hammer_manifest_sha256,
            args.expected_release_hammer_outer_sha256)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-source-contract", action="store_true")
    parser.add_argument("--validate-release", action="store_true")
    parser.add_argument("--consume-attempt", action="store_true")
    parser.add_argument("--run-exact-pair", action="store_true")
    parser.add_argument("--publish-no-replace", action="store_true")
    parser.add_argument("--contract", type=Path, default=SOURCE_CONTRACT)
    parser.add_argument("--release", type=Path, default=FUTURE_RELEASE)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--release-hammer", type=Path,
                        default=RELEASE_HAMMER)
    parser.add_argument("--expected-release-sha256", default="")
    parser.add_argument("--expected-release-hammer-review-sha256", default="")
    parser.add_argument("--expected-release-hammer-manifest-sha256", default="")
    parser.add_argument("--expected-release-hammer-outer-sha256", default="")
    parser.add_argument("--attempt-stage", type=Path)
    parser.add_argument("--output-stage", type=Path)
    args = parser.parse_args(argv)
    modes = [args.self_test, args.validate_source_contract,
             args.validate_release, args.consume_attempt,
             args.run_exact_pair, args.publish_no_replace]
    require(sum(bool(value) for value in modes) == 1,
            "M961 requires exactly one explicit mode")
    if args.self_test:
        value = source_self_test()
    elif args.validate_source_contract:
        require(args.runner is not None, "runner is required")
        value = validate_source_contract(args.contract, args.runner)
    else:
        require(args.runner is not None and args.expected_release_sha256,
                "release and runner identities are required")
        identity = _hammer_identity(args)
        if args.validate_release:
            value = validate_release(
                args.release, args.runner, args.expected_release_sha256,
                args.release_hammer, identity)
        elif args.consume_attempt:
            require(args.attempt_stage is not None,
                    "attempt stage is required")
            value = consume_attempt(
                args.release, args.runner, args.expected_release_sha256,
                args.release_hammer, identity, args.attempt_stage)
        elif args.run_exact_pair:
            require(args.output_stage is not None,
                    "output stage is required")
            value = run_exact_pair(
                args.release, args.runner, args.expected_release_sha256,
                args.release_hammer, identity, args.output_stage)
        else:
            require(args.output_stage is not None,
                    "output stage is required")
            validate_release(
                args.release, args.runner, args.expected_release_sha256,
                args.release_hammer, identity)
            value = publish_no_replace(
                args.output_stage, args.expected_release_sha256,
                identity[0])
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
