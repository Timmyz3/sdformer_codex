#!/usr/bin/env python3
"""对 sealed Local5 H3 trace-v2 执行定向篡改反例回归。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


Mutator = Callable[[dict[str, str]], bool]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mutate_trace(source: Path, target: Path, mutator: Mutator) -> None:
    changed = False
    with source.open("r", encoding="ascii", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None:
            raise ValueError("trace 缺少表头")
        with target.open("w", encoding="ascii", newline="") as target_handle:
            writer = csv.DictWriter(target_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if not changed and mutator(row):
                    changed = True
                    if row.get("event") == "__DELETE__":
                        continue
                writer.writerow(row)
    if not changed:
        raise ValueError("定向篡改未命中任何 trace 行")


def payload_flip(row: dict[str, str]) -> bool:
    if row["event"] != "relation_response_accept":
        return False
    old = row["payload"]
    row["payload"] = old[:-1] + ("0" if old[-1] != "0" else "1")
    return True


def metadata_flip(row: dict[str, str]) -> bool:
    if row["event"] != "weight_response_accept":
        return False
    row["lane"] = str(int(row["lane"]) + 1)
    return True


def delete_state(row: dict[str, str]) -> bool:
    if row["event"] != "tx_state":
        return False
    row["event"] = "__DELETE__"
    return True


def state_value_flip(row: dict[str, str]) -> bool:
    if row["event"] != "head_state":
        return False
    row["index"] = str(int(row["index"]) + 1)
    return True


CASES: tuple[tuple[str, Mutator, str], ...] = (
    ("relation_accept_payload_flip", payload_flip, "payload"),
    ("weight_accept_metadata_flip", metadata_flip, "metadata"),
    ("delete_one_tx_state", delete_state, "exact count/order digest"),
    ("head_state_value_flip", state_value_flip, "count/order digest"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    canary = args.canary_dir.resolve()
    release = args.release_dir.resolve()
    package = args.package_dir.resolve()
    expected = args.expected.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")
    output.mkdir(parents=True)
    verifier = release / "source/scripts/verify_local5_identity_service_rtl_trace_v2.py"
    state_reference = (
        release / "source/contracts/local5_identity_service_h3_state_reference_v1.json"
    )
    common = [
        "python3", str(verifier),
        "--package-dir", str(package),
        "--state-reference", str(state_reference),
        "--actual", str(canary / "actual.memh"),
        "--expected", str(expected),
        "--verilator-log", str(canary / "verilator.log"),
    ]
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="local5_trace_v2_tamper_") as temporary:
        temporary_root = Path(temporary)
        for name, mutator, expected_error_fragment in CASES:
            trace = temporary_root / f"{name}.csv"
            report = temporary_root / f"{name}.json"
            mutate_trace(canary / "identity_trace.csv", trace, mutator)
            completed = subprocess.run(
                [*common, "--trace", str(trace), "--output", str(report)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            combined = completed.stdout + completed.stderr
            passed = (
                completed.returncode != 0
                and expected_error_fragment in combined
                and not report.exists()
            )
            rows.append({
                "case": name,
                "status": "PASS_REJECTED" if passed else "FAIL_NOT_REJECTED",
                "returncode": completed.returncode,
                "mutated_trace_sha256": sha256(trace),
                "expected_error_fragment": expected_error_fragment,
                "diagnostic_tail": combined.strip().splitlines()[-1]
                if combined.strip() else "",
            })
            if not passed:
                raise RuntimeError(f"篡改反例未被预期拒绝：{name}")

    complete = json.loads((canary / "complete.json").read_text(encoding="utf-8"))
    binding_failures = []
    for name, binding in complete.get("direct_bindings", {}).items():
        path = Path(binding["path"])
        if not path.is_file() or binding.get("sha256") != sha256(path):
            binding_failures.append(name)
    if binding_failures:
        raise RuntimeError(f"v8 complete direct binding 失配：{binding_failures}")
    result = {
        "schema": "local5_identity_service_trace_v2_tamper_regression_v1",
        "status": "PASS_ALL_TAMPERS_REJECTED_NOT_G0",
        "evidence": "[rtl-verifier-negative-test]",
        "formal_g0": "DENY",
        "source_bindings": {
            "canary_complete_sha256": sha256(canary / "complete.json"),
            "sealed_verifier_sha256": sha256(verifier),
            "sealed_state_reference_sha256": sha256(state_reference),
            "source_trace_sha256": sha256(canary / "identity_trace.csv"),
        },
        "cases": rows,
        "direct_binding_count": len(complete["direct_bindings"]),
        "direct_binding_failures": binding_failures,
        "boundary": [
            "定向负测试只证明 verifier 能拒绝四类篡改",
            "不是 formal G0、性能、PPA 或多样本证据",
        ],
    }
    report_path = output / "tamper_regression.json"
    temporary_report = output / f"tamper_regression.json.tmp.{os.getpid()}"
    temporary_report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)
    source_copy = output / "runner_source.py"
    shutil.copy2(Path(__file__).resolve(), source_copy)
    receipt = {
        "status": result["status"],
        "report_sha256": sha256(report_path),
        "runner_source_sha256": sha256(source_copy),
    }
    (output / "complete.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
