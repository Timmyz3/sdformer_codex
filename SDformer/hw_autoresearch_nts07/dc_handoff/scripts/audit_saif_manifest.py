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
    direct_saif = data.get("activity_source_kind") == "direct_vcs_saif"
    source_vcd = Path(str(data.get("source_vcd", "")))
    source_activity = Path(str(data.get("source_activity", "")))
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

    contract_fields_match = direct_saif or (bool(contract_data) and all(
        (
            Path(str(data.get(name, ""))).resolve()
            == canonical_contract_path(name)
        )
        if name in {"source_vcd", "trace_root"}
        else data.get(name) == contract_data.get(name)
        for name in bound_fields
    ))

    def rehashed_file(path_field: str, sha_field: str) -> bool:
        path = Path(str(data.get(path_field, "")))
        return (
            path.is_file()
            and SHA256_RE.fullmatch(str(data.get(sha_field, ""))) is not None
            and data.get(sha_field) == sha256(path)
        )

    direct_activity = data.get("saif_critical_direct_dut_activity", {})
    direct_nonzero_names = (
        "clk_core", "descriptor_valid", "weight_request_valid",
        "weight_response_valid", "output_valid", "controller_state",
    )
    checks = {
        "design_name": data.get("design_name") == args.design,
        "saif_sha256": data.get("saif_sha256") == sha256(args.saif),
        "source_activity_identity": (
            source_activity.is_file()
            and SHA256_RE.fullmatch(
                str(data.get("source_activity_sha256", ""))
            ) is not None
            and data.get("source_activity_sha256") == sha256(source_activity)
        ) if direct_saif else (
            source_vcd.is_file()
            and SHA256_RE.fullmatch(
                str(data.get("source_vcd_sha256", ""))
            ) is not None
            and data.get("source_vcd_sha256") == sha256(source_vcd)
        ),
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
        in {"local5_group", "motion_row", "dual_line_descriptor_batch"},
        "trace_scope": bool(str(data.get("trace_scope", "")).strip()),
        "activity_contract_sha256": direct_saif or (
            activity_contract.is_file()
            and SHA256_RE.fullmatch(
                str(data.get("activity_contract_sha256", ""))
            ) is not None
            and data.get("activity_contract_sha256") == sha256(activity_contract)
        ),
        "identity_root": identity_root.is_dir(),
        "activity_contract_fields": contract_fields_match,
        "direct_toolchain_identity": (not direct_saif) or all(
            rehashed_file(path_field, sha_field)
            for path_field, sha_field in (
                ("simv", "simv_sha256"),
                ("runner", "runner_sha256"),
                ("ucli_script", "ucli_script_sha256"),
                ("manifest_builder", "manifest_builder_sha256"),
            )
        ),
        "direct_dut_scope": (not direct_saif) or (
            isinstance(data.get("saif_direct_dut_signal_record_count"), int)
            and data["saif_direct_dut_signal_record_count"] >= 20
            and all(
                isinstance(direct_activity.get(name), dict)
                and direct_activity[name].get("tc", 0) > 0
                for name in direct_nonzero_names
            )
            and isinstance(direct_activity.get("protocol_error"), dict)
            and direct_activity["protocol_error"].get("tc") == 0
            and direct_activity["protocol_error"].get("tx") == 0
        ),
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
