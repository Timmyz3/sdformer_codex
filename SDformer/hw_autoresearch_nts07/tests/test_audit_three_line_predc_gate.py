from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_three_line_predc_gate.py"


def test_three_line_predc_gate_is_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(ROOT),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["lines"]["motion_h67"]["component_predc"] == "READY_PREMACRO"
    assert report["lines"]["local5"]["component_predc"] == "READY_PREMACRO"
    assert (
        report["lines"]["h81"]["component_predc"]
        == "BLOCKED_UNTIL_SELECTOR"
    )
    assert report["server_handoff"]["paper_ppa_ready"] is False
    assert report["checks"]["docs359_frozen"] is True
    assert report["checks"]["handoff_sources_current"] is True
    assert report["checks"]["h81_contract_status_known"] is True


def test_three_line_gate_rejects_stale_handoff_hash(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    runs = tmp_path / "dc_handoff/runs"
    results = tmp_path / "results/grok_codex_collab"
    docs.mkdir(parents=True)
    runs.mkdir(parents=True)
    results.mkdir(parents=True)
    shutil.copy2(ROOT / "docs/359_DATE终局冻结_20260813.md", docs)
    bound = tmp_path / "bound.txt"
    bound.write_text("current\n", encoding="utf-8")
    checks = {
        "h67_fixed2s_mssb5_dc_top:filelist": True,
        "activity:motion_fixed:contract": True,
        "h67_rqtb2s_mssb5_dc_top:filelist": True,
        "activity:motion_rqtb:contract": True,
        "local5_unified_out2_dc_top:filelist": True,
        "activity:local5_full:contract": True,
        "local5_unified_out2_1rw_dc_top:filelist": True,
        "activity:local5_1rw_full:contract": True,
    }
    (runs / "date_dual_handoff_audit_20260815_v11.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "checks": checks,
                "sha256": {
                    "bound.txt": hashlib.sha256(bound.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    (results / "h81_identity_contract_20260815.json").write_text(
        json.dumps(
            {
                "schema": "h81_hardware_identity_contract_v1",
                "status": "PROPOSED_BOTH_SIDES_AGREE_IN_TEXT",
                "h81_rtl_now": False,
                "if_selector_chooses_h81": {
                    "requires": ["selector_official"]
                },
                "memory_impl_0_label": "pre-macro",
            }
        ),
        encoding="utf-8",
    )
    bound.write_text("stale\n", encoding="utf-8")
    output = tmp_path / "gate.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["checks"]["handoff_sources_current"] is False
