#!/usr/bin/env python3
"""Validate that a SAIF is bound to a frozen design and real trace identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", required=True)
    parser.add_argument("--saif", type=Path, required=True)
    parser.add_argument("--strip-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-paper-power-eligible", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_vcd = Path(str(data.get("source_vcd", "")))
    trace_root = Path(str(data.get("trace_root", "")))
    activity_contract = Path(str(data.get("activity_contract", "")))
    identity_root = Path(str(data.get("identity_root", "")))
    contract_data = {}
    if activity_contract.is_file():
        contract_data = json.loads(activity_contract.read_text(encoding="utf-8"))
    bound_fields = (
        "design_name", "source_vcd", "source_vcd_sha256", "trace_root",
        "trace_sha256", "simulator", "strip_path", "warmup_cycles",
        "measured_cycles", "busy_cycles", "measurement_overhead_cycles",
        "measurement_scope", "activity_purpose", "paper_power_eligible",
        "workload_kind", "trace_scope",
    )

    def canonical_contract_path(name: str) -> Path:
        value = Path(str(contract_data.get(name, "")))
        if not value.is_absolute():
            value = identity_root / value
        return value.resolve()

    contract_fields_match = bool(contract_data) and all(
        (
            Path(str(data.get(name, ""))).resolve()
            == canonical_contract_path(name)
        )
        if name in {"source_vcd", "trace_root"}
        else data.get(name) == contract_data.get(name)
        for name in bound_fields
    )
    checks = {
        "design_name": data.get("design_name") == args.design,
        "saif_sha256": data.get("saif_sha256") == sha256(args.saif),
        "source_vcd_sha256": source_vcd.is_file()
        and SHA256_RE.fullmatch(str(data.get("source_vcd_sha256", ""))) is not None
        and data.get("source_vcd_sha256") == sha256(source_vcd),
        "trace_sha256": trace_root.exists()
        and SHA256_RE.fullmatch(str(data.get("trace_sha256", ""))) is not None
        and data.get("trace_sha256") == sha256_tree(trace_root),
        "simulator": bool(str(data.get("simulator", "")).strip()),
        "strip_path": data.get("strip_path") == args.strip_path,
        "warmup_cycles": isinstance(data.get("warmup_cycles"), int)
        and data["warmup_cycles"] >= 0,
        "measured_cycles": isinstance(data.get("measured_cycles"), int)
        and data["measured_cycles"] > 0,
        "busy_cycles": isinstance(data.get("busy_cycles"), int)
        and data["busy_cycles"] > 0,
        "measurement_overhead_cycles": isinstance(
            data.get("measurement_overhead_cycles"), int
        )
        and data["measurement_overhead_cycles"] >= 0
        and data["measurement_overhead_cycles"]
        == data.get("measured_cycles", -1) - data.get("busy_cycles", 0),
        "measurement_scope": bool(str(data.get("measurement_scope", "")).strip()),
        "activity_purpose": data.get("activity_purpose")
        in {"identity_smoke", "paper_power_compute", "paper_power_with_io"},
        "paper_power_eligible": isinstance(data.get("paper_power_eligible"), bool)
        and (
            not args.require_paper_power_eligible
            or data.get("paper_power_eligible") is True
        ),
        "workload_kind": data.get("workload_kind")
        in {"local5_group", "motion_row"},
        "trace_scope": bool(str(data.get("trace_scope", "")).strip()),
        "activity_contract_sha256": activity_contract.is_file()
        and SHA256_RE.fullmatch(str(data.get("activity_contract_sha256", ""))) is not None
        and data.get("activity_contract_sha256") == sha256(activity_contract),
        "identity_root": identity_root.is_dir(),
        "activity_contract_fields": contract_fields_match,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "boundary": (
            "Identity admission only. PrimeTime annotation coverage is checked after "
            "read_saif and remains a separate requirement."
        ),
    }
    output = args.manifest.with_name(args.manifest.stem + "_audit.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
