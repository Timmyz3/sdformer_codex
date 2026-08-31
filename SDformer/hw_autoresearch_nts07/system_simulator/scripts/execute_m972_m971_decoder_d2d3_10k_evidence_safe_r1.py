#!/usr/bin/env python3
"""M972 evidence-safe additive wrapper for one future D2/D3 10K pair.

M972 freezes M946/M896 and repairs only the M961 driver/evidence lifecycle.
It derives source-fetch request counts from the generated transaction, keeps
bytes and requests distinct, accepts a prefix spanning arbitrary transaction
classes, persists and double-seals each row independently, and refuses 100K,
full-row, production, EDA, GPU and remote execution.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
PYTHON_PATH = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
PYTHON_VERSION = (3, 10, 18)
M946_PATH = HERE / "analyze_m946_decoder_multilayer_bounded_prefix_source_candidate.py"
M946_SHA256 = "0ffd1ee810f24d1a95b0df33ffe8eae43240920e12a2fccb86c947d2be51b6ac"
M896_PATH = HERE / "analyze_m896_decoder_run_gtls_source_candidate.py"
M896_SHA256 = "c877f70849eb254bd5b227c79e8120773a9c48aa7405a2e6564b7eb4647aae39"
M971_DIR = HW / "reviews/m971_m961_decoder_d2d3_10k_failure_forensic_r1_20260829"
M971_IDENTITY = (
    "36073062ebfeb3c8077cabdd2ebae7bc2053212084432460b742fc5a4bafc1ef",
    "83af03a8768c3728e67d537c585b5b913aecf5c3d86f90e3c71fcccb5601027d",
    "d1a19b066e205abc99cd31eaa58ae9ddab5619cfac2c822c6a67b291667a4c44",
)
M950_DIR = HW / "reviews/m950_m946_decoder_multilayer_bounded_prefix_source_fresh_hammer_r1_20260829"
M950_IDENTITY = (
    "2042b1d2f16a29be706a4c413ce3d473b7daedd56cca24dfd6aff57848579cf6",
    "8f749a2f9db1aa49d710765e3d89232b57029d3ed313f2da5299f0dfa3910ee7",
    "389bae76312b4f51655facdb56d6754c3bb6e93821c02b52b68a0f9f84b19e09",
)
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOURCE_CONTRACT = HW / "contracts/m972_m971_decoder_d2d3_10k_evidence_safe_source_contract_r1_20260829.json"
FUTURE_RELEASE = HW / "contracts/m974_m972_decoder_d2d3_10k_evidence_safe_release_r1_20260829.json"
SOURCE_HAMMER = HW / "reviews/m973_m972_decoder_d2d3_10k_evidence_safe_source_hammer_r1_20260829"
RELEASE_HAMMER = HW / "reviews/m975_m974_m972_decoder_d2d3_10k_evidence_safe_release_hammer_r1_20260829"
RESULT = HW / "results/m972_m946_decoder_d2d3_10k_evidence_safe_r1_20260829"
ATTEMPT = HW / "results/.m972_m946_decoder_d2d3_10k_evidence_safe_r1_attempt_consumed"
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."
SOURCE_SCHEMA = "m972_m971_decoder_d2d3_10k_evidence_safe_source_contract_v1"
RELEASE_SCHEMA = "m974_m972_decoder_d2d3_10k_evidence_safe_release_v1"
PREFIX_10K = 10000
EXPECTED_SOURCE_GEOMETRY = {
    "D2": {"source_bytes": 231600, "source_fetch_requests": 1207},
    "D3": {"source_bytes": 465600, "source_fetch_requests": 2425},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _pairs(pairs):
    output = {}
    for key, value in pairs:
        require(key not in output, "duplicate JSON key: " + key)
        output[key] = value
    return output


def strict_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(
                             ValueError("nonfinite JSON: " + value)))


def _write_exclusive(path: Path, data: bytes) -> None:
    with Path(path).open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _append_fsync(path: Path, text: str) -> None:
    with Path(path).open("ab") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def _validate_interpreter() -> Dict[str, object]:
    executable = Path(sys.executable).resolve()
    require(executable == PYTHON_PATH and sha256(executable) == PYTHON_SHA256,
            "M972 requires exact frozen Python")
    require(tuple(sys.version_info[:3]) == PYTHON_VERSION,
            "M972 Python version identity drift")
    return {"path": str(executable), "sha256": PYTHON_SHA256,
            "version": list(PYTHON_VERSION)}


_validate_interpreter()


def _load_m946():
    require(M946_PATH.is_file() and not M946_PATH.is_symlink() and
            sha256(M946_PATH) == M946_SHA256, "M946 source identity drift")
    require(M896_PATH.is_file() and not M896_PATH.is_symlink() and
            sha256(M896_PATH) == M896_SHA256, "M896 source identity drift")
    spec = importlib.util.spec_from_file_location("m972_frozen_m946", M946_PATH)
    require(spec is not None and spec.loader is not None,
            "cannot import frozen M946")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M946 = _load_m946()


def _verify_flat_seal(directory: Path, identity: Tuple[str, str, str],
                      label: str) -> Dict[str, str]:
    sealed = M946.M785.verify_sealed_directory(directory)
    require(sha256(directory / "review.json") == identity[0] and
            sealed["manifest_sha256"] == identity[1] and
            sealed["outer_seal_file_sha256"] == identity[2],
            label + " sealed identity drift")
    return sealed


def canonical_paths() -> Dict[str, str]:
    return {
        "source_contract": str(SOURCE_CONTRACT.relative_to(REPO)),
        "future_release": str(FUTURE_RELEASE.relative_to(REPO)),
        "source_hammer": str(SOURCE_HAMMER.relative_to(REPO)),
        "release_hammer": str(RELEASE_HAMMER.relative_to(REPO)),
        "result": str(RESULT.relative_to(REPO)),
        "attempt": str(ATTEMPT.relative_to(REPO)),
        "failure_prefix": "hw_autoresearch_nts07/results/" + FAILURE_PREFIX,
    }


def _safe_results_stage(path: Path, prefix: str) -> None:
    path = Path(path)
    require(path.parent.resolve() == RESULT.parent.resolve(),
            "M972 stage parent drift")
    require(path.name.startswith(prefix) and "/" not in path.name and
            path.name not in (".", ".."), "unsafe M972 stage basename")


def _seal_recursive(directory: Path) -> Dict[str, str]:
    directory = Path(directory)
    require(not (directory / "SHA256SUMS").exists() and
            not (directory / "SHA256SUMS.seal.sha256").exists(),
            "M972 directory already sealed")
    members = []
    for member in sorted(directory.rglob("*")):
        if member.is_file() and not member.is_symlink():
            rel = member.relative_to(directory).as_posix()
            members.append(sha256(member) + "  " + rel)
    require(members, "M972 refuses empty seal")
    _write_exclusive(directory / "SHA256SUMS",
                     ("\n".join(members) + "\n").encode("utf-8"))
    outer = sha256(directory / "SHA256SUMS") + "  SHA256SUMS\n"
    _write_exclusive(directory / "SHA256SUMS.seal.sha256",
                     outer.encode("utf-8"))
    return _verify_recursive(directory)


def _verify_recursive(directory: Path) -> Dict[str, str]:
    directory = Path(directory)
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink() and
            outer.is_file() and not outer.is_symlink(),
            "M972 recursive seal missing")
    require(outer.read_text(encoding="utf-8") ==
            sha256(manifest) + "  SHA256SUMS\n", "M972 outer seal mismatch")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        require(rel not in listed and rel not in
                ("SHA256SUMS", "SHA256SUMS.seal.sha256"),
                "M972 recursive manifest duplicate/reserved member")
        member = directory / rel
        require(member.resolve().is_relative_to(directory.resolve()) and
                member.is_file() and not member.is_symlink() and
                sha256(member) == digest, "M972 recursive member drift: " + rel)
        listed.add(rel)
    actual = {item.relative_to(directory).as_posix()
              for item in directory.rglob("*")
              if item.is_file() and not item.is_symlink() and
              item.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(actual == listed, "M972 recursive manifest coverage drift")
    return {"manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": sha256(outer)}


def source_fetch_geometry(layer: str) -> Dict[str, int]:
    require(layer in ("D2", "D3"), "M972 only accepts D2/D3")
    module_index = M946.MODULE_BY_LAYER[layer]
    cin, _, hin, win, _, _ = M946.M785.MODULE_GEOMETRY[module_index]
    source_bytes = math.ceil(int(cin) * int(hin) * int(win) / 8)
    generated = M946.M785._source_read(
        "m972_geometry", "m972_geometry", "A1_OSG", module_index, 0,
        source_bytes)
    value = {"module_index": int(module_index),
             "source_bytes": int(source_bytes),
             "source_fetch_requests": int(generated.count),
             "request_width_bytes": int(generated.width_bytes)}
    require({key: value[key] for key in
             ("source_bytes", "source_fetch_requests")} ==
            EXPECTED_SOURCE_GEOMETRY[layer],
            "generated source geometry drift for " + layer)
    return value


def summarize_row(row: Mapping[str, object]) -> Dict[str, object]:
    identity = row["row_identity"]
    layer = str(identity["layer"])
    require(layer in ("D2", "D3") and
            identity["numerical_route"] == "EXACT_BINARY_SUPPORT" and
            int(row["prefix"]) == PREFIX_10K,
            "M972 row identity drift")
    exact = row["exact_miter"]
    require(exact["status"] == "PASS_M768_M861_M890_M896_EXACT_MITER" and
            int(exact["expanded_request_count"]) == PREFIX_10K,
            "M972 exact miter/prefix drift")
    compressed = int(exact["compressed_transaction_count"])
    commits = int(exact["commit_requests_in_prefix"])
    require(compressed >= 1 and commits >= 0,
            "M972 invalid observed transaction counters")
    geometry = source_fetch_geometry(layer)
    return {
        "layer": layer,
        "prefix_requests": PREFIX_10K,
        "source_bytes": geometry["source_bytes"],
        "generated_source_fetch_requests": geometry["source_fetch_requests"],
        "requests_beyond_first_source_fetch":
            max(0, PREFIX_10K - geometry["source_fetch_requests"]),
        "prefix_stays_inside_first_source_fetch":
            PREFIX_10K <= geometry["source_fetch_requests"],
        "observed_compressed_transaction_count": compressed,
        "observed_commit_requests_in_prefix": commits,
        "contributor_or_later_phase_coverage_must_be_inferred_from_trace": True,
        "elapsed_seconds": float(row["elapsed_seconds"]),
        "process_max_rss_kib": int(row["process_max_rss_kib"]),
        "combined_live_event_state_bytes":
            int(exact["combined_live_event_state_bytes"]),
        "exact_miter_pass": True,
        "automatic_100k_authorized": False,
        "full_row_authorized": False,
    }


def execute_row_to_stage(layer: str, stage: Path,
                         producer: Callable[..., Mapping[str, object]] =
                         M946.run_bounded_prefix) -> Dict[str, object]:
    stage = Path(stage)
    require(stage.parent.name.startswith(RESULT.name + ".work.") and
            stage.name == layer and layer in ("D2", "D3"),
            "M972 row-stage namespace drift")
    require(not stage.exists() and not stage.is_symlink(),
            "M972 row stage collision")
    stage.mkdir(mode=0o700)
    _write_exclusive(stage / "ROW_STARTED.json", (json.dumps({
        "schema": "m972_row_started_v1", "layer": layer,
        "prefix": PREFIX_10K, "status": "STARTED_BEFORE_MODEL_CALL",
    }, sort_keys=True) + "\n").encode("utf-8"))
    _write_exclusive(stage / "row.log",
                     ("M972 {} 10K row stage created before model call\n".
                      format(layer)).encode("utf-8"))
    started = time.monotonic()
    try:
        row = producer(layer, 0, "A1_OSG", 0, PREFIX_10K)
        summary = summarize_row(row)
        payload = {"schema": "m972_decoder_10k_row_result_v1",
                   "status": "PASS_M972_ROW_EXACT__EVIDENCE_SAFE",
                   "row": row, "summary": summary,
                   "wrapper_elapsed_seconds": time.monotonic() - started,
                   "claim_boundary": {"paper_citable": False,
                                      "full_row_authorized": False,
                                      "automatic_100k_authorized": False,
                                      "system_speedup": False}}
        _write_exclusive(stage / "row.json", (json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False) +
            "\n").encode("utf-8"))
        _append_fsync(stage / "row.log", "row complete; sealing before next row\n")
        _write_exclusive(stage / "ROW_COMPLETE.txt",
                         b"M972_ROW_COMPLETE__FRESH_HAMMER_REQUIRED\n")
        sealed = _seal_recursive(stage)
        return {"payload": payload, "sealed_identity": sealed}
    except BaseException as error:
        trace = traceback.format_exc()
        _write_exclusive(stage / "traceback.log", trace.encode("utf-8"))
        _write_exclusive(stage / "failure.json", (json.dumps({
            "schema": "m972_decoder_10k_row_failure_v1",
            "status": "FAILED_QUARANTINE_REQUIRED", "layer": layer,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "wrapper_elapsed_seconds": time.monotonic() - started,
        }, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        _append_fsync(stage / "row.log", "row failed; traceback persisted; sealing\n")
        _write_exclusive(stage / "ROW_FAILED.txt",
                         b"M972_ROW_FAILED__QUARANTINE_REQUIRED\n")
        _seal_recursive(stage)
        raise


def validate_source_contract(contract_path: Path, runner_path: Path,
                             require_fresh: bool = True) -> Dict[str, object]:
    require(Path(contract_path).resolve() == SOURCE_CONTRACT.resolve(),
            "M972 contract canonical path drift")
    contract = strict_json(contract_path)
    require(contract.get("schema") == SOURCE_SCHEMA and
            contract.get("status") ==
            "SOURCE_ONLY__M973_HAMMER_AND_M974_M975_RELEASE_CHAIN_REQUIRED" and
            contract.get("launch_now") is False,
            "M972 source authority drift")
    require(contract.get("canonical") == canonical_paths(),
            "M972 canonical path drift")
    for name, item in contract["source_identity"].items():
        path = HW / item["path"]
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == item["sha256"],
                "M972 source identity drift: " + name)
    require(Path(runner_path).resolve() ==
            (HW / contract["source_identity"]["m972_runner"]["path"]).resolve(),
            "M972 runner path drift")
    _verify_flat_seal(M971_DIR, M971_IDENTITY, "M971")
    _verify_flat_seal(M950_DIR, M950_IDENTITY, "M950")
    require(sha256(HW / "docs/359_DATE终局冻结_20260813.md") ==
            DOCS359_SHA256, "docs359 drift")
    require({layer: {key: source_fetch_geometry(layer)[key] for key in
                     ("source_bytes", "source_fetch_requests")}
             for layer in ("D2", "D3")} == EXPECTED_SOURCE_GEOMETRY,
            "M972 geometry contract drift")
    if require_fresh:
        require(not RESULT.exists() and not ATTEMPT.exists(),
                "M972 result/attempt namespace not fresh")
    return {"status": "PASS_M972_SOURCE_CONTRACT__NO_10K_EXECUTED",
            "contract_sha256": sha256(contract_path),
            "runner_sha256": sha256(Path(runner_path)),
            "geometry": EXPECTED_SOURCE_GEOMETRY,
            "prefix_executed": False}


def validate_release(release_path: Path, runner_path: Path,
                     expected_release_sha256: str,
                     hammer_dir: Path,
                     hammer_identity: Tuple[str, str, str]) -> Dict[str, object]:
    source = validate_source_contract(
        SOURCE_CONTRACT, runner_path, require_fresh=False)
    require(Path(release_path).resolve() == FUTURE_RELEASE.resolve() and
            sha256(release_path) == expected_release_sha256,
            "M972 future release identity drift")
    release = strict_json(release_path)
    require(release.get("schema") == RELEASE_SCHEMA and
            release.get("status") == "AUTHORIZE_ONE_D2_THEN_D3_10K_PAIR" and
            release.get("release") is True and
            release.get("launch_now") is False and
            release.get("max_attempts") == 1,
            "M972 future release authority drift")
    require(release.get("exact_rows") == [
        {"layer": "D2", "sample_id": 0, "config": "A1_OSG",
         "timestep": 0, "expanded_prefix": PREFIX_10K},
        {"layer": "D3", "sample_id": 0, "config": "A1_OSG",
         "timestep": 0, "expanded_prefix": PREFIX_10K}],
            "M972 release row order/scope drift")
    auth = release.get("authorization", {})
    require(auth.get("one_d2_then_d3_10k_pair") is True and
            all(auth.get(key) is False for key in
                ("automatic_retry", "d2_or_d3_100k", "full_row",
                 "production", "eda_gpu_remote")),
            "M972 future release expands authority")
    binding = release.get("source_binding", {})
    require(binding.get("source_contract_sha256") ==
            source["contract_sha256"] and
            binding.get("driver_sha256") == sha256(Path(__file__)) and
            binding.get("runner_sha256") == sha256(Path(runner_path)) and
            binding.get("m946_sha256") == M946_SHA256 and
            binding.get("m896_sha256") == M896_SHA256 and
            binding.get("m971_review_sha256") == M971_IDENTITY[0],
            "M972 future release source binding drift")
    require(Path(hammer_dir).resolve() == RELEASE_HAMMER.resolve(),
            "M972 release hammer path drift")
    sealed = _verify_flat_seal(hammer_dir, hammer_identity, "M975")
    review = strict_json(hammer_dir / "review.json")
    require(review.get("status") ==
            "PASS_M975_M974_M972_EVIDENCE_SAFE_RELEASE_HAMMER" and
            review.get("verdict") == "GO_ONE_D2_THEN_D3_10K_PAIR_ONLY" and
            review.get("release_sha256") == expected_release_sha256,
            "M972 release hammer authority drift")
    return {"status": "PASS_M972_ONE_PAIR_RELEASE_AUTHORITY",
            "release_sha256": expected_release_sha256,
            "release_hammer_review_sha256": hammer_identity[0],
            "release_hammer_manifest_sha256": sealed["manifest_sha256"]}


def consume_attempt(stage: Path, release_sha256: str,
                    release_hammer_review_sha256: str) -> Dict[str, object]:
    _safe_results_stage(stage, ATTEMPT.name + ".stage.")
    require(not stage.exists() and not ATTEMPT.exists() and
            not RESULT.exists(), "M972 attempt/result namespace collision")
    stage.mkdir(mode=0o700)
    receipt = {"schema": "m972_decoder_10k_attempt_v1",
               "status": "CONSUMED_BEFORE_D2_MODEL_CALL",
               "max_attempts": 1, "release_sha256": release_sha256,
               "release_hammer_review_sha256":
                   release_hammer_review_sha256,
               "automatic_retry": False, "d2_or_d3_100k_authorized": False,
               "full_row_authorized": False}
    _write_exclusive(stage / "attempt.json", (json.dumps(
        receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    sealed = _seal_recursive(stage)
    os.rename(stage, ATTEMPT)
    require(_verify_recursive(ATTEMPT) == sealed,
            "M972 attempt publication drift")
    return {"receipt": receipt, "sealed_identity": sealed}


def validate_attempt(release_sha256: str,
                     release_hammer_review_sha256: str) -> Dict[str, object]:
    sealed = _verify_recursive(ATTEMPT)
    receipt = strict_json(ATTEMPT / "attempt.json")
    require(receipt.get("status") == "CONSUMED_BEFORE_D2_MODEL_CALL" and
            receipt.get("max_attempts") == 1 and
            receipt.get("release_sha256") == release_sha256 and
            receipt.get("release_hammer_review_sha256") ==
            release_hammer_review_sha256 and
            receipt.get("automatic_retry") is False,
            "M972 consumed attempt drift")
    return {"receipt": receipt, "sealed_identity": sealed}


def assemble_work_root(work_root: Path, release_sha256: str,
                       release_hammer_review_sha256: str) -> Dict[str, object]:
    _safe_results_stage(work_root, RESULT.name + ".work.")
    require(not (work_root / "SHA256SUMS").exists(),
            "M972 work root already sealed")
    rows = []
    identities = {}
    for layer in ("D2", "D3"):
        stage = work_root / layer
        identities[layer] = _verify_recursive(stage)
        payload = strict_json(stage / "row.json")
        require(payload.get("status") ==
                "PASS_M972_ROW_EXACT__EVIDENCE_SAFE" and
                payload.get("summary", {}).get("layer") == layer,
                "M972 row payload/order drift")
        rows.append(payload)
    result = {"schema": "m972_decoder_d2d3_10k_evidence_safe_result_v1",
              "status": "PASS_M972_D2_THEN_D3_10K_EXACT_PAIR",
              "release_sha256": release_sha256,
              "release_hammer_review_sha256":
                  release_hammer_review_sha256,
              "rows": rows, "row_sealed_identities": identities,
              "claim_boundary": {"fresh_result_hammer_required": True,
                                 "paper_citable": False,
                                 "automatic_100k_authorized": False,
                                 "full_row_authorized": False,
                                 "decoder_complete": False,
                                 "table_a_row": False,
                                 "system_speedup": False}}
    _write_exclusive(work_root / "result.json", (json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) +
        "\n").encode("utf-8"))
    _write_exclusive(work_root / "RUN_COMPLETE.txt",
                     b"M972_D2_THEN_D3_COMPLETE__FRESH_HAMMER_REQUIRED\n")
    sealed = _seal_recursive(work_root)
    return {"result": result, "sealed_identity": sealed}


def seal_failure_root(work_root: Path, return_code: int) -> Dict[str, object]:
    _safe_results_stage(work_root, RESULT.name + ".work.")
    require(work_root.is_dir() and not (work_root / "SHA256SUMS").exists(),
            "M972 failure root absent/already sealed")
    for layer in ("D2", "D3"):
        row_stage = work_root / layer
        if row_stage.is_dir() and not (row_stage / "SHA256SUMS").exists():
            if not (row_stage / "ROW_INTERRUPTED.txt").exists():
                _write_exclusive(
                    row_stage / "ROW_INTERRUPTED.txt",
                    b"M972_ROW_INTERRUPTED__QUARANTINE_AND_NO_RETRY\n")
            _seal_recursive(row_stage)
    _write_exclusive(work_root / "failure.json", (json.dumps({
        "schema": "m972_decoder_pair_failure_v1",
        "status": "FAILED_OR_INCOMPLETE__QUARANTINE_REQUIRED",
        "return_code": int(return_code), "automatic_retry": False,
        "d2_or_d3_100k_authorized": False, "full_row_authorized": False,
    }, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    _write_exclusive(work_root / "RUN_FAILED.txt",
                     b"M972_PAIR_FAILED__QUARANTINE_AND_NO_RETRY\n")
    return {"status": "PASS_M972_FAILURE_ROOT_DOUBLE_SEALED",
            "sealed_identity": _seal_recursive(work_root)}


def publish_no_replace(work_root: Path) -> Dict[str, object]:
    _safe_results_stage(work_root, RESULT.name + ".work.")
    sealed = _verify_recursive(work_root)
    result = strict_json(work_root / "result.json")
    require(result.get("status") ==
            "PASS_M972_D2_THEN_D3_10K_EXACT_PAIR" and
            not RESULT.exists(), "M972 publish authority/namespace drift")
    os.rename(work_root, RESULT)
    require(_verify_recursive(RESULT) == sealed,
            "M972 published result identity drift")
    return {"status": "PASS_M972_NO_REPLACE_PUBLICATION",
            "sealed_identity": sealed}


def source_self_test() -> Dict[str, object]:
    def fake(layer, sample_id, config, timestep, prefix):
        del sample_id, config, timestep
        return {"row_identity": {"layer": layer,
                                 "numerical_route": "EXACT_BINARY_SUPPORT"},
                "prefix": prefix, "elapsed_seconds": 0.1,
                "process_max_rss_kib": 1,
                "exact_miter": {
                    "status": "PASS_M768_M861_M890_M896_EXACT_MITER",
                    "expanded_request_count": prefix,
                    "compressed_transaction_count": 157,
                    "commit_requests_in_prefix": 9,
                    "combined_live_event_state_bytes": 1024}}
    with tempfile.TemporaryDirectory(prefix="m972_static_") as temporary:
        work = Path(temporary) / (RESULT.name + ".work.static")
        work.mkdir()
        success = execute_row_to_stage("D2", work / "D2", fake)
        require(success["payload"]["summary"][
                    "observed_compressed_transaction_count"] == 157 and
                success["payload"]["summary"][
                    "observed_commit_requests_in_prefix"] == 9,
                "M972 multi-transaction/commit directed test drift")
        failed = False
        try:
            execute_row_to_stage(
                "D3", work / "D3",
                lambda *args: (_ for _ in ()).throw(RuntimeError("injected")))
        except RuntimeError:
            failed = True
        require(failed and (work / "D3/failure.json").is_file(),
                "M972 injected failure did not persist")
        _verify_recursive(work / "D2")
        _verify_recursive(work / "D3")
    return {"status": "PASS_M972_SOURCE_SELF_TEST__NO_REAL_PREFIX",
            "geometry": {layer: source_fetch_geometry(layer)
                         for layer in ("D2", "D3")},
            "multi_transaction_accepted": True,
            "commit_coverage_accepted": True,
            "failure_trace_persisted_and_double_sealed": True,
            "real_prefix_executed": False, "full_row_authorized": False}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-source-contract", action="store_true")
    parser.add_argument("--validate-release", action="store_true")
    parser.add_argument("--consume-attempt", action="store_true")
    parser.add_argument("--run-row", choices=("D2", "D3"))
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--seal-failure-root", action="store_true")
    parser.add_argument("--publish-no-replace", action="store_true")
    parser.add_argument("--contract", type=Path, default=SOURCE_CONTRACT)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--row-stage", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--attempt-stage", type=Path)
    parser.add_argument("--release", type=Path, default=FUTURE_RELEASE)
    parser.add_argument("--release-hammer", type=Path,
                        default=RELEASE_HAMMER)
    parser.add_argument("--expected-release-sha256", default="")
    parser.add_argument("--expected-release-hammer-review-sha256", default="")
    parser.add_argument("--expected-release-hammer-manifest-sha256", default="")
    parser.add_argument("--expected-release-hammer-outer-sha256", default="")
    parser.add_argument("--return-code", type=int, default=1)
    args = parser.parse_args(argv)
    modes = [args.self_test, args.validate_source_contract,
             args.validate_release, args.consume_attempt,
             args.run_row is not None, args.assemble,
             args.seal_failure_root, args.publish_no_replace]
    require(sum(bool(item) for item in modes) == 1,
            "M972 requires exactly one explicit mode")
    if args.self_test:
        value = source_self_test()
    elif args.validate_source_contract:
        require(args.runner is not None, "M972 runner required")
        value = validate_source_contract(args.contract, args.runner)
    elif args.seal_failure_root:
        require(args.work_root is not None, "M972 work root required")
        value = seal_failure_root(args.work_root, args.return_code)
    else:
        require(args.runner is not None and args.expected_release_sha256,
                "M972 release/runner identity required")
        hammer_identity = (
            args.expected_release_hammer_review_sha256,
            args.expected_release_hammer_manifest_sha256,
            args.expected_release_hammer_outer_sha256)
        require(all(hammer_identity), "M972 release hammer identity required")
        authority = validate_release(
            args.release, args.runner, args.expected_release_sha256,
            args.release_hammer, hammer_identity)
        if args.validate_release:
            value = authority
        elif args.consume_attempt:
            require(args.attempt_stage is not None,
                    "M972 attempt stage required")
            value = consume_attempt(
                args.attempt_stage, args.expected_release_sha256,
                hammer_identity[0])
        else:
            validate_attempt(args.expected_release_sha256,
                             hammer_identity[0])
            if args.run_row is not None:
                require(args.row_stage is not None,
                        "M972 row stage required")
                value = execute_row_to_stage(args.run_row, args.row_stage)
            elif args.assemble:
                require(args.work_root is not None,
                        "M972 work root required")
                value = assemble_work_root(
                    args.work_root, args.expected_release_sha256,
                    hammer_identity[0])
            else:
                require(args.work_root is not None,
                        "M972 work root required")
                value = publish_no_replace(args.work_root)
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
