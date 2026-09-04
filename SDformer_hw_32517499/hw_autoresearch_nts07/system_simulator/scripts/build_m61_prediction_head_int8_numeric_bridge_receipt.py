#!/usr/bin/env python3
"""Seal the canonical M61 validation and tamper evidence into one receipt."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


RESULT_SHA = "2410ea8b55cf25f60fce2e297480dc865d99521b1e63cecd6f2bcafb541fc9ad"
BLOCKER_SHA = "33110f31555b72c9e175df7244dec5031c2382ef64909e6447a2695d33d92762"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--tamper-runner", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--analyzer", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--m60-result", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--gpu-blocker-observation", required=True, type=Path)
    parser.add_argument("--tamper-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "refusing existing output")
    result_path = arguments.result_dir / (
        "m61_prediction_head_int8_numeric_bridge_result_r1.json")
    require(sha256_path(result_path) == RESULT_SHA and
            sha256_path(arguments.gpu_blocker_observation) == BLOCKER_SHA,
            "result/blocker seal mismatch")
    command = [
        sys.executable, str(arguments.validator.resolve()),
        "--contract", str(arguments.contract.resolve()),
        "--analyzer", str(arguments.analyzer.resolve()),
        "--manifest", str(arguments.manifest.resolve()),
        "--m60-result", str(arguments.m60_result.resolve()),
        "--result-dir", str(arguments.result_dir.resolve()),
        "--gpu-blocker-observation",
        str(arguments.gpu_blocker_observation.resolve()),
        "--expected-result-sha256", RESULT_SHA,
        "--expected-gpu-blocker-sha256", BLOCKER_SHA,
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True)
    require(process.returncode == 0, "canonical validator rerun failed")
    validator_payload = json.loads(process.stdout)
    tamper = strict_json(arguments.tamper_receipt)
    require(tamper["status"] == "PASS_ALL_TAMPERS_REJECTED" and
            int(tamper["tamper_count"]) == 15 and
            all(row["rejected"] and row["exit_code"] != 0
                for row in tamper["attacks"]), "tamper receipt mismatch")
    result = strict_json(result_path)
    blocker = strict_json(arguments.gpu_blocker_observation)
    artifacts = {}
    for path in sorted(arguments.result_dir.iterdir()):
        if path.is_file():
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
    receipt = {
        "artifacts": artifacts,
        "canonical_validator": {
            "argv": command,
            "exit_code": int(process.returncode),
            "stderr": process.stderr,
            "stdout_json": validator_payload,
        },
        "claim_boundary": result["claim_boundary"],
        "gpu_blocker": {
            "compute_apps_stdout": blocker["compute_apps"]["stdout"],
            "gpu_stdout": blocker["gpu"]["stdout"],
            "observation_sha256": BLOCKER_SHA,
            "policy": blocker["policy"],
            "timestamp_utc": blocker["timestamp_utc"],
            "training_config": blocker["training_configs"][0],
            "training_process_line_count": len(blocker[
                "training_process_lines"]),
        },
        "headline": {
            "accumulator_bits": result["accumulator_proof"][
                "declared_signed_bits"],
            "dense_vs_delta_integer_mismatches": result[
                "integer_parent_delta_reconstruction"]["mismatches"],
            "float_dequant_mae": result[
                "numeric_perturbation_ten_real_samples"]["all_outputs"]["mae"],
            "float_dequant_rmse": result[
                "numeric_perturbation_ten_real_samples"]["all_outputs"]["rmse"],
            "quantized_valid825": "OPEN_WITH_OBSERVED_GPU_CONFLICT",
        },
        "schema": "m61_prediction_head_int8_numeric_bridge_validation_receipt_v1",
        "sources": {
            "analyzer_sha256": sha256_path(arguments.analyzer),
            "contract_sha256": sha256_path(arguments.contract),
            "manifest_sha256": sha256_path(arguments.manifest),
            "m60_result_sha256": sha256_path(arguments.m60_result),
            "receipt_builder_sha256": sha256_path(Path(__file__).resolve()),
            "tamper_runner_sha256": sha256_path(arguments.tamper_runner),
            "validator_sha256": sha256_path(arguments.validator),
        },
        "status": (
            "PASS_INT8_NUMERIC_EXACT_TEN_SAMPLE_VALID825_OPEN_"
            "OBSERVED_GPU_CONFLICT"),
        "tamper": {
            "receipt_sha256": sha256_path(arguments.tamper_receipt),
            "status": tamper["status"],
            "tamper_count": tamper["tamper_count"],
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        arguments.output.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.link(str(temporary), str(arguments.output))
    temporary.unlink()
    print(json.dumps({
        "output_sha256": sha256_path(arguments.output),
        "status": receipt["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
