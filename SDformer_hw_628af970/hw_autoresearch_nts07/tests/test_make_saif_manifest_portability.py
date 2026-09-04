from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "dc_handoff/scripts/make_saif_manifest.py"
AUDIT = ROOT / "dc_handoff/scripts/audit_saif_manifest.py"


def test_relative_activity_paths_are_resolved_from_root(tmp_path: Path) -> None:
    source = tmp_path / "activity/source.vcd"
    trace = tmp_path / "trace/input.memh"
    saif = tmp_path / "activity/output.saif"
    source.parent.mkdir(parents=True)
    trace.parent.mkdir(parents=True)
    source.write_text("vcd\n", encoding="utf-8")
    trace.write_text("trace\n", encoding="utf-8")
    saif.write_text("saif\n", encoding="utf-8")
    contract = {
        "status": "PASS",
        "design_name": "unit",
        "source_vcd": "activity/source.vcd",
        "source_vcd_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "trace_root": "trace/input.memh",
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "simulator": "unit",
        "strip_path": "TOP/unit",
        "warmup_cycles": 0,
        "measured_cycles": 1,
        "busy_cycles": 1,
        "measurement_overhead_cycles": 0,
        "measurement_scope": "unit",
        "activity_purpose": "identity_smoke",
        "paper_power_eligible": False,
        "workload_kind": "motion_row",
        "trace_scope": "unit",
    }
    contract_path = tmp_path / "contract.json"
    output = tmp_path / "manifest.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--activity-contract",
            str(contract_path),
            "--saif",
            str(saif),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["source_vcd"] == str(source.resolve())
    assert manifest["trace_root"] == str(trace.resolve())
    assert manifest["identity_root"] == str(tmp_path.resolve())
    completed = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "--design",
            "unit",
            "--saif",
            str(saif),
            "--strip-path",
            "TOP/unit",
            "--manifest",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    audit = json.loads(
        output.with_name(output.stem + "_audit.json").read_text(encoding="utf-8")
    )
    assert audit["checks"]["activity_contract_fields"] is True
