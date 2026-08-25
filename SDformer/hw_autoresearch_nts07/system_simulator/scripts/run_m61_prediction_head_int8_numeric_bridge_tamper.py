#!/usr/bin/env python3
"""Run independent fail-closed tamper attacks against the M61 validator."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def flip_first_byte(path):
    data = bytearray(Path(path).read_bytes())
    require(data, "cannot flip empty file")
    data[0] ^= 1
    Path(path).write_bytes(bytes(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--analyzer", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--m60-result", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--gpu-blocker-observation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "refusing existing output")

    canonical = dict((name, Path(value).resolve()) for name, value in {
        "validator": arguments.validator,
        "contract": arguments.contract,
        "analyzer": arguments.analyzer,
        "manifest": arguments.manifest,
        "m60": arguments.m60_result,
        "result_dir": arguments.result_dir,
        "blocker": arguments.gpu_blocker_observation,
    }.items())

    def invoke(paths, expected_result=RESULT_SHA, expected_blocker=BLOCKER_SHA):
        command = [
            sys.executable, str(paths["validator"]),
            "--contract", str(paths["contract"]),
            "--analyzer", str(paths["analyzer"]),
            "--manifest", str(paths["manifest"]),
            "--m60-result", str(paths["m60"]),
            "--result-dir", str(paths["result_dir"]),
            "--gpu-blocker-observation", str(paths["blocker"]),
            "--expected-result-sha256", expected_result,
            "--expected-gpu-blocker-sha256", expected_blocker,
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, universal_newlines=True)
        return {
            "exit_code": int(process.returncode),
            "stderr_tail": process.stderr[-800:],
            "stdout_tail": process.stdout[-800:],
        }

    baseline = invoke(canonical)
    require(baseline["exit_code"] == 0, "canonical validator baseline failed")
    attacks = []

    def attack(name, mutation, expected_result=RESULT_SHA,
               expected_blocker=BLOCKER_SHA):
        with tempfile.TemporaryDirectory(prefix="m61_tamper_") as temporary:
            root = Path(temporary)
            paths = {
                "validator": root / "validator.py",
                "contract": root / "contract.json",
                "analyzer": root / "analyzer.py",
                "manifest": root / "manifest.json",
                "m60": root / "m60.json",
                "result_dir": root / "result",
                "blocker": root / "blocker.json",
            }
            for key in ("validator", "contract", "analyzer", "manifest",
                        "m60", "blocker"):
                shutil.copy2(str(canonical[key]), str(paths[key]))
            shutil.copytree(str(canonical["result_dir"]),
                            str(paths["result_dir"]))
            mutation(paths)
            observed = invoke(paths, expected_result, expected_blocker)
            require(observed["exit_code"] != 0,
                    "tamper accepted: {}".format(name))
            attacks.append({"name": name, "rejected": True, **observed})

    result_name = "m61_prediction_head_int8_numeric_bridge_result_r1.json"
    attack("result_integer_mismatch_nonzero", lambda p: (
        lambda d: (d["integer_parent_delta_reconstruction"].__setitem__(
            "mismatches", 1), write_json(p["result_dir"] / result_name, d))
    )(read_json(p["result_dir"] / result_name)))
    attack("result_numeric_mae_changed", lambda p: (
        lambda d: (d["numeric_perturbation_ten_real_samples"][
            "all_outputs"].__setitem__("mae", 0.0),
                   write_json(p["result_dir"] / result_name, d))
    )(read_json(p["result_dir"] / result_name)))
    attack("result_accumulator_bits_changed", lambda p: (
        lambda d: (d["accumulator_proof"].__setitem__(
            "declared_signed_bits", 12),
                   write_json(p["result_dir"] / result_name, d))
    )(read_json(p["result_dir"] / result_name)))
    attack("result_per_record_removed", lambda p: (
        lambda d: (d["per_record"].pop(),
                   write_json(p["result_dir"] / result_name, d))
    )(read_json(p["result_dir"] / result_name)))
    attack("weight_bit_flip", lambda p: flip_first_byte(
        p["result_dir"] / "prediction_head_weight_int8.bin"))
    attack("bias_truncated", lambda p: (p["result_dir"] /
        "prediction_head_bias_int32_le.bin").write_bytes(b"\x00" * 4))
    attack("scale_bit_flip", lambda p: flip_first_byte(
        p["result_dir"] / "prediction_head_scale_float32_le.bin"))
    attack("weight_missing", lambda p: (p["result_dir"] /
        "prediction_head_weight_int8.bin").unlink())
    attack("contract_claim_changed", lambda p: (
        lambda d: (d["claim_boundary"]["forbidden"].pop(),
                   write_json(p["contract"], d))
    )(read_json(p["contract"])))
    attack("analyzer_source_changed", lambda p: p["analyzer"].write_text(
        p["analyzer"].read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8"))
    attack("manifest_float_weight_identity_changed", lambda p: (
        lambda d: (d["module_identities"][
            "sttmultires_unet.preds.3.conv.0"]["weight"].__setitem__(
                "content_sha256", "0" * 64), write_json(p["manifest"], d))
    )(read_json(p["manifest"])))
    attack("m60_signed_source_changed", lambda p: (
        lambda d: (d["configurations"][0].__setitem__(
            "source_bits", d["configurations"][0]["source_bits"] + 1),
                   write_json(p["m60"], d))
    )(read_json(p["m60"])))
    attack("gpu_blocker_utilization_changed", lambda p: (
        lambda d: (d["gpu"].__setitem__("stdout", "0, fake, 0, 81920, 0\n"),
                   write_json(p["blocker"], d))
    )(read_json(p["blocker"])))
    attack("expected_result_sha_changed", lambda p: None,
           expected_result="0" * 64)
    attack("expected_blocker_sha_changed", lambda p: None,
           expected_blocker="0" * 64)

    payload = {
        "attacks": attacks,
        "baseline": baseline,
        "canonical": {
            "analyzer_sha256": sha256_path(canonical["analyzer"]),
            "blocker_observation_sha256": sha256_path(canonical["blocker"]),
            "contract_sha256": sha256_path(canonical["contract"]),
            "manifest_sha256": sha256_path(canonical["manifest"]),
            "m60_result_sha256": sha256_path(canonical["m60"]),
            "result_sha256": sha256_path(
                canonical["result_dir"] / result_name),
            "validator_sha256": sha256_path(canonical["validator"]),
        },
        "schema": "m61_prediction_head_int8_numeric_bridge_tamper_receipt_v1",
        "status": "PASS_ALL_TAMPERS_REJECTED",
        "tamper_count": len(attacks),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        arguments.output.name + ".tmp.{}".format(os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.link(str(temporary), str(arguments.output))
    temporary.unlink()
    print(json.dumps({
        "output_sha256": sha256_path(arguments.output),
        "status": payload["status"],
        "tamper_count": payload["tamper_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
