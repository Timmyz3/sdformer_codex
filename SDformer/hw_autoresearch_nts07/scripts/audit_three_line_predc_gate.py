#!/usr/bin/env python3
"""Build a fail-closed Motion/Local5/H81 pre-DC admission matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DOCS359 = "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = (
    "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
HANDOFF_AUDIT = "dc_handoff/runs/date_dual_handoff_audit_20260815_v11.json"
H81_CONTRACT = "results/grok_codex_collab/h81_identity_contract_20260815.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output

    frozen = root / DOCS359
    handoff_path = root / HANDOFF_AUDIT
    h81_path = root / H81_CONTRACT
    handoff = load_json(handoff_path)
    h81 = load_json(h81_path)
    handoff_checks = handoff.get("checks", {})
    handoff_hashes = handoff.get("sha256", {})

    handoff_sources_current = bool(handoff_hashes)
    for relative, expected in handoff_hashes.items():
        source = root / relative
        if not source.is_file() or sha256(source) != expected:
            handoff_sources_current = False
            break

    checks = {
        "docs359_frozen": frozen.is_file()
        and sha256(frozen) == DOCS359_SHA256,
        "handoff_audit_pass": handoff.get("status") == "PASS",
        "handoff_sources_current": handoff_sources_current,
        "motion_fixed_component_present": bool(
            handoff_checks.get("h67_fixed2s_mssb5_dc_top:filelist")
            and handoff_checks.get("activity:motion_fixed:contract")
        ),
        "motion_rqtb_component_present": bool(
            handoff_checks.get("h67_rqtb2s_mssb5_dc_top:filelist")
            and handoff_checks.get("activity:motion_rqtb:contract")
        ),
        "local5_1r1w_component_present": bool(
            handoff_checks.get("local5_unified_out2_dc_top:filelist")
            and handoff_checks.get("activity:local5_full:contract")
        ),
        "local5_1rw_sensitivity_present": bool(
            handoff_checks.get("local5_unified_out2_1rw_dc_top:filelist")
            and handoff_checks.get("activity:local5_1rw_full:contract")
        ),
        "h81_identity_contract_present": h81.get("schema")
        == "h81_hardware_identity_contract_v1",
        "h81_contract_status_known": h81.get("status")
        in {"PROPOSED_BOTH_SIDES_AGREE_IN_TEXT", "FROZEN"},
        "h81_not_current_rtl": h81.get("h81_rtl_now") is False,
        "h81_requires_selector": "selector_official"
        in h81.get("if_selector_chooses_h81", {}).get("requires", []),
        "all_current_memories_are_premacro": h81.get("memory_impl_0_label")
        == "pre-macro",
    }

    motion_component_ready = all(
        checks[name]
        for name in (
            "docs359_frozen",
            "handoff_audit_pass",
            "handoff_sources_current",
            "motion_fixed_component_present",
            "motion_rqtb_component_present",
        )
    )
    local5_component_ready = all(
        checks[name]
        for name in (
            "docs359_frozen",
            "handoff_audit_pass",
            "handoff_sources_current",
            "local5_1r1w_component_present",
            "local5_1rw_sensitivity_present",
        )
    )
    h81_blocked = all(
        checks[name]
        for name in (
            "h81_identity_contract_present",
            "h81_contract_status_known",
            "h81_not_current_rtl",
            "h81_requires_selector",
        )
    )

    payload = {
        "schema": "date_three_line_predc_gate_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "priority": [
            "P0_motion_h67_component_handoff",
            "P1_local5_production_system_closure",
            "P_infinity_h81_until_selector",
        ],
        "lines": {
            "motion_h67": {
                "hardware_identity": "H67_ep35",
                "component_predc": (
                    "READY_PREMACRO" if motion_component_ready else "BLOCKED"
                ),
                "component_boundary": (
                    "matched Fixed2S/RQTB2S T450 attention-row slices; "
                    "not a full block or encoder"
                ),
                "paper_system": "BLOCKED",
                "paper_system_blockers": [
                    "monolithic real-INT8 post-attention/full-block boundary",
                    "cross-head and 12-block system schedule at the paper boundary",
                    "target macro adapters and target-library DC/STA/SAIF/PTPX",
                ],
                "innovation_identity": "MSSB5_to_Q7_to_RQTB_on_H67_only",
            },
            "local5": {
                "hardware_identity": "Local5_independent_checkpoint_rebindable",
                "component_predc": (
                    "READY_PREMACRO" if local5_component_ready else "BLOCKED"
                ),
                "component_boundary": (
                    "OUT_DIM=2 score-to-Acc32 tile; 1RW is a physical "
                    "sensitivity baseline, neither is an encoder"
                ),
                "paper_system": "BLOCKED",
                "paper_system_blockers": [
                    "final algorithm checkpoint selection and trace rebind",
                    "bounded 12-block numerical replay at the current production boundary",
                    "target macro adapters and target-library DC/STA/SAIF/PTPX",
                ],
                "component_system_evidence": (
                    "synthetic deterministic-oracle 3-head OUT32 RTL plus "
                    "12-block structural checks; not final-checkpoint evidence"
                ),
                "innovation_identity": (
                    "QS_to_compiled_inverse_stencil_to_source_owned_TCFM5"
                ),
            },
            "h81": {
                "hardware_identity": "algorithm_control_only",
                "identity_contract_status": h81.get("status"),
                "component_predc": (
                    "BLOCKED_UNTIL_SELECTOR" if h81_blocked else "CONTRACT_ERROR"
                ),
                "paper_system": "BLOCKED_UNTIL_SELECTOR",
                "activation_requirements": h81.get(
                    "if_selector_chooses_h81", {}
                ).get("requires", []),
                "forbidden_inheritance": {
                    "h67_speedup": "1.1865x",
                    "motion_innovation_score": "3.2",
                    "local5_topology": "self_plus_four_not_present_in_H81",
                },
                "innovation_identity": "must_be_requalified_from_real_H81_trace",
            },
        },
        "server_handoff": {
            "premacro_component_queue_ready": bool(
                motion_component_ready and local5_component_ready
            ),
            "paper_ppa_ready": False,
            "missing_external_inputs": [
                "target SRAM/RF macro wrappers and .db files",
                "target PVT and operating condition",
                "Synopsys DC/Formality/PrimeTime/PTPX executables",
            ],
            "forbidden_claim": (
                "READY_PREMACRO does not mean only DC remains for paper submission"
            ),
        },
        "inputs": {
            DOCS359: sha256(frozen) if frozen.is_file() else None,
            HANDOFF_AUDIT: sha256(handoff_path) if handoff_path.is_file() else None,
            H81_CONTRACT: sha256(h81_path) if h81_path.is_file() else None,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
