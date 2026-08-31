#!/usr/bin/env python3
"""Static-only audit of the C2 production SAIF/PTPX admission gap.

This checker deliberately does not invoke VCS, PrimeTime, DC, or any license
tool.  It verifies frozen identities, the two mapped-gate failure boundaries,
the scope of the later K1-only semantic repair, and the absence of a citable
C2 production SAIF before emitting a fail-closed NO-GO receipt.
"""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "dc_handoff/scripts/run_m1046_m1001_c2_mapped_gate_saif_one_shot_r5.sh": "381afcf82fd8a95a2966320cc2fc25d7965d5d0e74f060c8ab6aaef8027e4856",
    "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv": "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
    "dc_handoff/scripts/m979_c2_mapped_gate_per_case_saif.ucli.tcl": "846cd4a1b877803cce986b39cdf0a27ec87b59451ca7e6fc9141c999df85cdad",
    "contracts/m979_m974_c2_three_axis_mapped_gate_saif_source_contract_r1_20260829.json": "d2939e24e587b03680b7b4e0265a8fc8b3dbbea89759e2268e97b118fe32455c",
    "reviews/m1050_m1046_c2_mapped_gate_watchdog_failure_audit_r1_20260829/review.json": "de6802c2dd139f63c90036aed08c35107649c900c4a736e39390fbbd463bcd8b",
    "reviews/m1088_m1080_c2_mapped_gate_failure_audit_r1_20260830/review.json": "ed638e53512852abe3514e7ff211733867c17d8e912241c7f4ce92000233b246",
    "contracts/m1293_c2_semantic_tap_dual_dut_repair_source_contract_r1_20260830.json": "1c50a862e02aeda009d52850f00ba8befa96c19b6599077e61951b36929299f5",
    "reviews/m1305_m1304_m1293_c2_rtl_only_vcs_result_hammer_r1_20260830/review.json": "3724e6a6fe39da1358bc905fd40dbea6df523b45cbe42a04575a517a5383a34d",
    "results/m1046_m1001_c2_ucli_power_preflight.2027456.sealed/preflight.json": "03563668f4245aa2eefba8cce90dda09a5c057119546e40b1bd428ab2c8abd62",
    "results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829.failed_or_incomplete.2027456.quarantine/failure.json": "d14796750f90217092c4e552cc0b056b0e4ebfa4d1a3f2df54f7f4389879c163",
    "results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.2746017.quarantine/failure.json": "5a2e7b4a25c609d3500f887da33eaaeb86c046402df0f2fd4e8d86fc686c2e28",
    "dc_handoff/scripts/run_ptpx.tcl": "879398c8b8708589d42346af10d4825afac19c7c0622601685d1ea3f72245368",
}

AXIS_INPUTS = {
    "k1": (
        "060e7cd00e5a0f79860430c823439424ae88211cd2ff0d71bc787c9e6691d6b3",
        "df2b08e2c8a8faa87f7ab8f738888589f7b7595b386b905388b9428204c5a9bd",
    ),
    "k8": (
        "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
        "70a0d0e7700188f5a80f31b06c2f3d401f56c7d1e2a29428e3837064a722a96c",
    ),
    "k1x8": (
        "65f89c13d0b181fd26708b385fc831bb4493328e24a15bbb07c2dc40f27677dc",
        "24806d5c2d5c0afae2c01d518927e3ca96ec977d000287b0a6bc62fc42a7e317",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL_M1331_STATIC_AUDIT: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hw = Path(__file__).resolve().parents[2]
    checked = {}
    for rel, expected in EXPECTED.items():
        path = hw / rel
        require(path.is_file(), f"missing frozen evidence: {rel}")
        actual = sha256(path)
        require(actual == expected, f"identity drift: {rel}: {actual}")
        checked[rel] = actual

    mapped_root = hw / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
    mapped = {}
    for axis, (net_sha, sdc_sha) in AXIS_INPUTS.items():
        stem = mapped_root / axis / "netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped"
        netlist, sdc = stem.with_suffix(".v"), stem.with_suffix(".sdc")
        require(sha256(netlist) == net_sha, f"{axis} mapped netlist drift")
        require(sha256(sdc) == sdc_sha, f"{axis} mapped SDC drift")
        mapped[axis] = {"netlist_sha256": net_sha, "sdc_sha256": sdc_sha}

    m1050 = load_json(hw / "reviews/m1050_m1046_c2_mapped_gate_watchdog_failure_audit_r1_20260829/review.json")
    require(m1050["failure_boundary"]["production_saif_files"] == 0, "M1046 unexpectedly has production SAIF")
    require(m1050["root_cause"]["class"] == "GATE_LEVEL_UNINITIALIZED_STATE_X_PROPAGATION", "M1050 root-cause drift")
    require(m1050["authorization"]["ptpx"] is False, "M1050 unexpectedly authorizes PTPX")

    m1088 = load_json(hw / "reviews/m1088_m1080_c2_mapped_gate_failure_audit_r1_20260830/review.json")
    require(m1088["failure_boundary"]["completed_cases"] == 0, "M1080 unexpectedly completed a mapped case")
    require(m1088["root_cause_classification"]["primary"] == "REMAINING_UNRESET_CONTROL_OR_PAYLOAD_VALID_BIT_X_ISOLATION_GAP", "M1088 classification drift")
    require(m1088["authorization"]["ptpx"] is False, "M1088 unexpectedly authorizes PTPX")

    m1293 = load_json(hw / "contracts/m1293_c2_semantic_tap_dual_dut_repair_source_contract_r1_20260830.json")
    boundary = m1293["claim_boundary"]
    require(boundary["k1_diagnostic_axis_only"] is True, "M1293 is no longer diagnostic K1-only")
    require(boundary["k8_present"] is False and boundary["equal_bandwidth_k1x8_present"] is False, "M1293 unexpectedly covers headline axes")
    require(boundary["saif"] is False and boundary["ptpx"] is False, "M1293 unexpectedly admits power")

    preflight = load_json(hw / "results/m1046_m1001_c2_ucli_power_preflight.2027456.sealed/preflight.json")
    tiny = hw / "results/m1046_m1001_c2_ucli_power_preflight.2027456.sealed/tiny.saif"
    require(tiny.is_file() and tiny.stat().st_size == 2106, "tiny UCLI preflight identity/size drift")
    require(preflight["status"] == "PASS_M1044_TINY_UCLI_POWER_SAIF_PREFLIGHT", "tiny preflight status drift")

    production_roots = [
        hw / "results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829.failed_or_incomplete.2027456.quarantine",
        hw / "results/m1080_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.2746017.quarantine",
    ]
    production_saif = sorted(str(p.relative_to(hw)) for root in production_roots for p in root.rglob("*.saif"))
    require(not production_saif, f"unexpected production SAIF found: {production_saif}")

    c2_ptpx_sources = sorted(
        str(p.relative_to(hw))
        for p in (hw / "dc_handoff/scripts").glob("*c2*ptpx*")
        if p.is_file()
    )
    require(not c2_ptpx_sources, "a C2-specific PTPX source appeared; audit must be refreshed")
    generic_ptpx = (hw / "dc_handoff/scripts/run_ptpx.tcl").read_text(encoding="utf-8")
    for token in ("read_saif", "update_power", "report_switching_activity", "report_power"):
        require(token in generic_ptpx, f"generic PTPX template missing {token}")

    result = {
        "schema": "m1331_c2_production_saif_ptpx_readonly_gap_audit_r1_v1",
        "status": "NO_GO_DIRECT_PRODUCTION_SAIF_PTPX__MAPPED_ACTIVITY_INPUT_NOT_ADMITTED",
        "score": 100,
        "eda_launched": False,
        "checked_identity": checked,
        "mapped_inputs": mapped,
        "findings": {
            "old_three_axis_mapped_netlists_present": True,
            "old_mapped_compile_and_tiny_ucli_preflight_passed": True,
            "tiny_preflight_is_production_activity": False,
            "m1046_completed_production_cases": 0,
            "m1046_production_saif_files": 0,
            "m1080_reset_hygiene_scope": "K1 diagnostic only",
            "m1080_completed_mapped_cases": 0,
            "m1293_m1304_scope": "RTL-only directed K1 diagnostic",
            "headline_axes_with_repaired_mapped_replay": 0,
            "c2_specific_ptpx_sources": c2_ptpx_sources,
            "generic_ptpx_template_present": True,
            "production_saif_paths": production_saif,
        },
        "root_cause": [
            "The only three-axis production attempt produced zero completed cases and zero production SAIF files after a mapped-gate post-header watchdog.",
            "The additive reset-hygiene mapped K1 attempt still completed zero cases; its audit leaves a remaining unreset valid/control or X-isolation gap.",
            "The later semantic-tap repair is RTL-only and K1-only, so it does not bind the K8 versus equal-bandwidth K1x8 headline comparison.",
            "A generic PrimeTime PX template exists, but no admitted C2 production SAIF or C2-specific annotation/energy execution contract exists.",
        ],
        "minimum_missing_artifacts": [
            "Additive mapped-replay repair that closes the first failing valid/control cone without initreg dependence, applied consistently to K8 and equal-bandwidth K1x8 (K1 remains diagnostic).",
            "Fresh same-source/same-constraint K8 and K1x8 mapped netlists and SDCs after that repair, with exact SHA identities and one frozen power corner.",
            "Fresh mapped-gate VCS replay for the same five frozen workloads on both headline axes, preserving numeric, tuple, weight, unknown, protocol, cycle-window and major-cone gates.",
            "Ten per-case DUT-only production SAIF files (fifteen if diagnostic K1 is retained), each excluding reset/preheader/post-token idle and proving duration equals measured cycles times 3 ns.",
            "C2-specific PrimeTime PX Tcl/runner that pins netlist, SDC, library/operating condition, SAIF strip path, annotation coverage gate, update_power/report_power, and energy equals average power times measured activity duration.",
            "Fresh exact-SHA single-job namespace, independent source/release hammer, and fail-closed result extractor; consumed M1046/M1080 namespaces must not be reused.",
        ],
        "claim_boundary": {
            "production_saif": False,
            "ptpx": False,
            "power": False,
            "energy": False,
            "fair_k8_vs_k1x8_energy": False,
            "system_energy": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
        "docs359_sha256": EXPECTED["docs/359_DATE终局冻结_20260813.md"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(result["status"])


if __name__ == "__main__":
    main()
