#!/usr/bin/env python3
"""对参数化 Local5 phase-template archive 执行九类定向篡改回归。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mutate_template_event(arrays: dict[str, np.ndarray]) -> None:
    index = int(arrays["template_offsets"][1])
    old = int(arrays["template_event_code"][index])
    arrays["template_event_code"][index] = (old + 1) % len(arrays["event_dictionary"])


def mutate_patch_cycle(arrays: dict[str, np.ndarray]) -> None:
    arrays["patch_cycle"][1000] += np.uint32(1)


def mutate_patch_payload(arrays: dict[str, np.ndarray]) -> None:
    index = int(np.flatnonzero(arrays["patch_payload_code"] != 0)[0])
    old = int(arrays["patch_payload_code"][index])
    arrays["patch_payload_code"][index] = np.uint32(
        (old + 1) % len(arrays["payload_dictionary"])
    )


def mutate_instance_class(arrays: dict[str, np.ndarray]) -> None:
    arrays["instance_class_code"][1] = np.uint8(3)


def mutate_patch_offset(arrays: dict[str, np.ndarray]) -> None:
    arrays["patch_offsets"][1] += np.int64(1)


def mutate_instance_tile(arrays: dict[str, np.ndarray]) -> None:
    arrays["instance_tile"][1] += np.int16(1)


def mutate_instance_head(arrays: dict[str, np.ndarray]) -> None:
    arrays["instance_head"][1] += np.int16(1)


def mutate_dictionary_code(arrays: dict[str, np.ndarray]) -> None:
    arrays["template_origin_code"][0] = np.uint8(255)


def mutate_patch_identity(arrays: dict[str, np.ndarray]) -> None:
    arrays["patch_index"][1000] += np.int32(1)


CASES: tuple[tuple[str, Callable[[dict[str, np.ndarray]], None], str], ...] = (
    ("template_event_flip", mutate_template_event, "expansion differs"),
    ("patch_cycle_flip", mutate_patch_cycle, "expansion differs"),
    ("patch_payload_flip", mutate_patch_payload, "expansion differs"),
    ("instance_class_flip", mutate_instance_class, "typed metadata sequence"),
    ("patch_offset_flip", mutate_patch_offset, "patch length"),
    ("instance_tile_flip", mutate_instance_tile, "typed metadata sequence"),
    ("instance_head_flip", mutate_instance_head, "typed metadata sequence"),
    ("dictionary_code_oob", mutate_dictionary_code, "dictionary code"),
    ("patch_identity_flip", mutate_patch_identity, "expansion differs"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--verifier", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    canary = args.canary_dir.resolve()
    release = args.release_dir.resolve() if args.release_dir else None
    baseline = args.baseline_dir.resolve() if args.baseline_dir else canary
    vectors = args.vector_dir.resolve()
    table = args.table_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    source_archive = canary / "template_patch/phase_template_patch.npz"
    source_manifest = canary / "template_patch/manifest.json"
    if args.verifier:
        verifier = args.verifier.resolve()
    elif release:
        verifier = release / "source/scripts/verify_local5_h3_phase_template_patch_v1.py"
    else:
        raise ValueError("either --verifier or --release-dir is required")
    baseline_trace = (
        baseline / "baseline_trace.csv"
        if (baseline / "baseline_trace.csv").is_file()
        else baseline / "identity_trace.csv"
    )
    baseline_actual = (
        baseline / "baseline_actual.memh"
        if (baseline / "baseline_actual.memh").is_file()
        else baseline / "actual.memh"
    )
    candidate_log = (
        canary / "candidate_verilator.log"
        if (canary / "candidate_verilator.log").is_file()
        else canary / "verilator.log"
    )
    with np.load(source_archive, allow_pickle=False) as handle:
        pristine = {name: np.array(handle[name], copy=True) for name in handle.files}
    rows = []
    with tempfile.TemporaryDirectory(prefix="local5_phase_patch_tamper_") as temporary:
        root = Path(temporary)
        for name, mutator, expected_fragment in CASES:
            arrays = {key: np.array(value, copy=True) for key, value in pristine.items()}
            mutator(arrays)
            archive = root / f"{name}.npz"
            manifest = root / f"{name}.manifest.json"
            report = root / f"{name}.report.json"
            np.savez(archive, **arrays)
            manifest_value = json.loads(source_manifest.read_text(encoding="utf-8"))
            manifest_value["archive_sha256"] = sha256(archive)
            manifest.write_text(
                json.dumps(manifest_value, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            command = [
                "python3", str(verifier),
                "--archive", str(archive),
                "--manifest", str(manifest),
                "--candidate-trace", str(canary / "candidate_trace.csv"),
                "--baseline-trace", str(baseline_trace),
                "--candidate-actual", str(canary / "candidate_actual.memh"),
                "--baseline-actual", str(baseline_actual),
                "--expected", str(vectors / "software_expected/software_expected.npz"),
                "--inputs", str(vectors / "vectors/combined_head_inputs.txt"),
                "--weights", str(vectors / "vectors/projection_weights.txt"),
                "--identity-manifest", str(table / "manifest.json"),
                "--identity-receipt", str(table / "verification_receipt.json"),
                "--verilator-log", str(candidate_log),
                "--output", str(report),
            ]
            completed = subprocess.run(
                command, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False,
            )
            diagnostic = completed.stdout + completed.stderr
            passed = (
                completed.returncode != 0
                and expected_fragment in diagnostic
                and not report.exists()
            )
            rows.append({
                "case": name,
                "status": "PASS_REJECTED" if passed else "FAIL_NOT_REJECTED",
                "returncode": completed.returncode,
                "tampered_archive_sha256": sha256(archive),
                "diagnostic_tail": diagnostic.strip().splitlines()[-1]
                if diagnostic.strip() else "",
            })
            if not passed:
                raise RuntimeError(f"篡改未按预期被拒绝：{name}")
    result = {
        "schema": "local5_phase_template_tamper_regression_v2",
        "status": "PASS_ALL_TEMPLATE_TAMPERS_REJECTED_NOT_G0",
        "evidence": "[independent-expander-negative-test]",
        "formal_g0": "DENY",
        "source_bindings": {
            "canary_complete_sha256": sha256(canary / "complete.json"),
            "sealed_verifier_sha256": sha256(verifier),
            "source_archive_sha256": sha256(source_archive),
        },
        "cases": rows,
        "boundary": [
            "仅证明独立 expander 能拒绝 template/patch 九类定向篡改",
            "不是 formal G0、性能或 PPA 证据",
        ],
    }
    report_path = output / "tamper_regression.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    complete = {
        "schema": "local5_phase_template_tamper_complete_v2",
        "status": result["status"],
        "formal_g0": "DENY",
        "report_sha256": sha256(report_path),
        "runner_source_sha256": sha256(Path(__file__).resolve()),
    }
    (output / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(complete))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
