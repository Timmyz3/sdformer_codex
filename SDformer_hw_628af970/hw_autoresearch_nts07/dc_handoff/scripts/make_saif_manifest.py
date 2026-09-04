#!/usr/bin/env python3
"""Create the final PTPX SAIF manifest from an admitted wrapper VCD contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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
    parser.add_argument("--activity-contract", type=Path, required=True)
    parser.add_argument("--saif", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    contract = json.loads(args.activity_contract.read_text(encoding="utf-8"))
    if contract.get("status") != "PASS":
        raise SystemExit("activity contract is not PASS")
    source_vcd = Path(contract.get("source_vcd", ""))
    trace_root = Path(contract.get("trace_root", ""))
    if not source_vcd.is_absolute():
        source_vcd = root / source_vcd
    if not trace_root.is_absolute():
        trace_root = root / trace_root
    if not source_vcd.is_file() or sha256(source_vcd) != contract.get("source_vcd_sha256"):
        raise SystemExit("activity contract source VCD identity mismatch")
    if not trace_root.exists() or sha256_tree(trace_root) != contract.get("trace_sha256"):
        raise SystemExit("activity contract trace identity mismatch")
    required = (
        "design_name",
        "source_vcd_sha256",
        "trace_sha256",
        "simulator",
        "strip_path",
        "warmup_cycles",
        "measured_cycles",
        "busy_cycles",
        "measurement_overhead_cycles",
        "measurement_scope",
        "activity_purpose",
        "paper_power_eligible",
        "workload_kind",
        "trace_scope",
        "source_vcd",
        "trace_root",
    )
    missing = [name for name in required if name not in contract]
    if missing:
        raise SystemExit(f"activity contract missing: {missing}")
    manifest = {name: contract[name] for name in required}
    manifest["source_vcd"] = str(source_vcd.resolve())
    manifest["trace_root"] = str(trace_root.resolve())
    manifest["identity_root"] = str(root)
    manifest["saif_sha256"] = sha256(args.saif)
    manifest["activity_contract_sha256"] = sha256(args.activity_contract)
    manifest["activity_contract"] = str(args.activity_contract.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
