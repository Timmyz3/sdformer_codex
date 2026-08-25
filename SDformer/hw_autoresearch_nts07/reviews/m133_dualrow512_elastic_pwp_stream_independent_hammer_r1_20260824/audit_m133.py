#!/usr/bin/env python3
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).resolve().parent

EXPECTED = {
    "rtl_m133/m133_dualrow512_elastic_pwp_stream.sv":
        "84f1b6f6e8d085f14bbe8abe7b2fbfd9dbac586d178ce7e3eb2dff55db92f6de",
    "reviews/m133_dualrow512_elastic_pwp_stream_independent_hammer_r1_20260824/frozen_r1_m133_assertions.sv":
        "ab45fe7d15dd5a57a55461dd92ebd67a2a8a482a57c0e383f35c1a9c0b62a9b4",
    "contracts/m133_dualrow512_elastic_pwp_stream_vcs_contract_r1_20260824.json":
        "c135a995d9405e0d459d92ff6a7d8a12c51fb07d01e25c2723a5781ccf38609f",
    "dc_handoff/filelists/date_m133_dualrow512_elastic_pwp_stream_directed_vcs.f":
        "575a3171e12b701f58709a68703a18eb0a4d111e215e7e4393921c2a4f347c31",
    "contracts/m133_dualrow512_elastic_pwp_stream_logic_only_dc_contract_r1_20260824.json":
        "194e58c4781c667a42a32dd95de83f83ccca50cf56717b063a3017161c70f14d",
    "verif_m133/m133_dualrow512_elastic_pwp_stream_assertions.sv":
        "564fc8184977f352d4d841164583f0dc694ce8ba33fd3d2d6f871a3c2cbc6cea",
    "tb_m133/tb_m133_dualrow512_elastic_pwp_stream.sv":
        "3b73c0ea7d572382521e112a7962febe9c9733899b3a1ca30fa282b97708a742",
    "contracts/m133_r1_stall_fault_composition_correction_r1_20260824.json":
        "a32d3bab8faddf0a318c6ba6a3a1b36cb2ac579c08b3d327b56f4e599f59feff",
    "contracts/m133r2_dualrow512_elastic_pwp_stream_vcs_contract_r1_20260824.json":
        "75d827342d36a82318a29f3efe7149a87b73eeacd576f94c9c533d9cb4c2020f",
    "contracts/m133r2_dc_functional_supersession_overlay_r1_20260824.json":
        "30a1ec5b97117a5315b70e697fe0ef8a95af84d4505d4be5ec9eaa380213b041",
    "dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_logic_only_dc_3p000ns_r1_sealed_20260824/m133_logic_only_dc_receipt_r1.json":
        "b52895549ca4a7da55fbd3cefdb2d1c587777880becc95876591cc3fbf85d07f",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    hashes = {}
    for relative, expected in EXPECTED.items():
        observed = sha256(ROOT / relative)
        hashes[relative] = observed
        require(observed == expected, "exact-SHA mismatch: " + relative)

    lanes = 96
    service_bits = 512
    geometry = {}
    for width in (8, 9, 10, 11):
        payload_bits = lanes * width
        beats = int(math.ceil(payload_bits / float(service_bits)))
        tail_bits = beats * service_bits - payload_bits
        geometry[str(width)] = {
            "payload_bits": payload_bits,
            "accepted_beats": beats,
            "last_beat_valid_bits": payload_bits - (beats - 1) * service_bits,
            "last_beat_padding_bits": tail_bits,
            "signed_min": -(1 << (width - 1)),
            "signed_max": (1 << (width - 1)) - 1,
        }

    contract_path = ROOT / (
        "contracts/m133_dualrow512_elastic_pwp_stream_vcs_contract_r1_20260824.json"
    )
    contract = json.loads(contract_path.read_text())
    contract_beats = contract["architecture"]["accepted_beats_by_width"]
    require(
        {key: value["accepted_beats"] for key, value in geometry.items()}
        == contract_beats,
        "contract beat geometry differs from independent ceiling division",
    )

    # Positive traffic is derivable directly from the TB loops, independent of
    # its counters: 64 round-robin vectors, 40 mixed vectors (four escapes),
    # then one width-11 long-stall vector.
    phase1_widths = [8 + (index % 4) for index in range(64)]
    phase2_items = []
    for index in range(64, 104):
        phase2_items.append("escape" if index % 10 == 0 else 8 + (index % 4))
    phase3_widths = [11]
    positive_numeric_widths = phase1_widths + [
        item for item in phase2_items if item != "escape"
    ] + phase3_widths
    positive_escapes = sum(item == "escape" for item in phase2_items)
    positive_beats = sum(
        geometry[str(width)]["accepted_beats"] for width in positive_numeric_widths
    ) + positive_escapes
    attack_setup_beats = 1 + 2
    accepted_beats = positive_beats + attack_setup_beats
    require(len(positive_numeric_widths) + positive_escapes == 105,
            "positive vector count mismatch")
    require(positive_beats == 233, "positive beat count mismatch")
    require(accepted_beats == 236, "total accepted beat count mismatch")
    require(len(positive_numeric_widths) * lanes == 9696,
            "numeric lane check count mismatch")

    sealed = ROOT / (
        "dc_handoff/runs/"
        "m133_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824"
    )
    require((sealed / "compile.rc").read_text().strip() == "0", "sealed compile failed")
    require((sealed / "sim.rc").read_text().strip() == "0", "sealed sim failed")
    sim_log = (sealed / "sim.raw.log").read_text()
    require(contract["expected_pass_line"] in sim_log, "sealed pass line missing")
    require("failed at" not in (sealed / "assert.report").read_text(),
            "sealed assertion failure")
    sealed_inputs = (sealed / "input_sha256.txt").read_text()
    require(
        "f5280aac2212c9688fc3a74aa1a87d4bc2e5c2ad256ca9826ba343d2f8f5b435  "
        "tb_m133/tb_m133_dualrow512_elastic_pwp_stream.sv" in sealed_inputs,
        "sealed r1 TB identity missing",
    )
    require(
        "ab45fe7d15dd5a57a55461dd92ebd67a2a8a482a57c0e383f35c1a9c0b62a9b4  "
        "verif_m133/m133_dualrow512_elastic_pwp_stream_assertions.sv" in sealed_inputs,
        "sealed r1 SVA identity missing",
    )

    dc_run = ROOT / (
        "dc_handoff/runs/"
        "m133_dualrow512_elastic_pwp_stream_logic_only_dc_3p000ns_r1_sealed_20260824"
    )
    require((dc_run / "dc.rc").read_text().strip() == "0", "DC backend failed")
    dc_receipt = dict(
        line.split("=", 1) for line in (dc_run / "RUN_COMPLETE.txt").read_text().splitlines()
    )
    require(dc_receipt["cell_area_um2"] == "10853.766052", "DC area drift")
    require(dc_receipt["setup_worst_slack_ns"] == "1.1005", "DC setup drift")
    require(dc_receipt["hold_worst_slack_ns"] == "0.0001", "DC hold drift")
    require(dc_receipt["macro_count"] == "0", "unexpected DC macro")
    area_report = (dc_run / "reports/area.rpt").read_text()
    require("Number of cells:                        12667" in area_report,
            "DC cell count drift")
    require("Number of sequential cells:              2767" in area_report,
            "DC sequential count drift")
    require("Net Interconnect area:      undefined" in area_report,
            "DC wireload boundary drift")
    constraints = (dc_run / "reports/constraint_violators.rpt").read_text()
    require(constraints.count("This design has no violated constraints.") == 5,
            "DC constraint report incomplete")

    cross_report = REVIEW / "frozen_r1_stall_fault_cross_property/assert.report"
    require(cross_report.exists(), "independent cross-property report missing")
    cross_text = cross_report.read_text()
    require("ap_output_stable_under_stall" in cross_text and "failed at" in cross_text,
            "expected frozen-SVA cross-property counterexample missing")

    r2_sealed = ROOT / (
        "dc_handoff/runs/"
        "m133r2_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824"
    )
    require((r2_sealed / "compile.rc").read_text().strip() == "0",
            "sealed r2 compile failed")
    require((r2_sealed / "sim.rc").read_text().strip() == "0",
            "sealed r2 sim failed")
    r2_contract = json.loads((ROOT / (
        "contracts/m133r2_dualrow512_elastic_pwp_stream_vcs_contract_r1_20260824.json"
    )).read_text())
    require(r2_contract["expected_pass_line"] in (r2_sealed / "sim.raw.log").read_text(),
            "sealed r2 pass line missing")
    r2_assert = (r2_sealed / "assert.report").read_text()
    require("failed at" not in r2_assert and "Offending" not in r2_assert,
            "sealed r2 assertion failure")
    require("cp_stall_to_fault_quarantine" in r2_assert and " 1 match" in r2_assert,
            "sealed r2 overlap cover missing")

    independent_r2_receipt = dict(
        line.split("=", 1) for line in
        (REVIEW / "R2_VCS_REVIEW_COMPLETE.txt").read_text().splitlines()
    )
    require(independent_r2_receipt["r1_counterexample_closed"] == "true",
            "independent r2 did not close r1 counterexample")
    require(independent_r2_receipt["independent_assertion_failures"] == "false",
            "independent r2 assertion failure")

    output = {
        "schema": "m133_independent_geometry_and_receipt_audit_v1",
        "status": "PASS_M133R2_WITH_PHYSICAL_SCOPE_BOUNDARY",
        "geometry": geometry,
        "traffic_reconstruction": {
            "positive_vectors": 105,
            "positive_numeric_vectors": len(positive_numeric_widths),
            "positive_escapes": positive_escapes,
            "positive_accepted_beats": positive_beats,
            "attack_setup_accepted_beats": attack_setup_beats,
            "total_accepted_beats": accepted_beats,
            "numeric_lane_checks": len(positive_numeric_widths) * lanes,
            "meaning_of_236": (
                "accepted input handshakes in this directed simulation, including "
                "three legal setup beats before two negative attacks"
            ),
        },
        "source_sha256": hashes,
        "sealed_run": {
            "compile_rc": 0,
            "sim_rc": 0,
            "expected_pass_line_found": True,
            "assertion_failures_found": False,
            "r1_testbench_sha256":
                "f5280aac2212c9688fc3a74aa1a87d4bc2e5c2ad256ca9826ba343d2f8f5b435",
            "r1_sva_sha256":
                "ab45fe7d15dd5a57a55461dd92ebd67a2a8a482a57c0e383f35c1a9c0b62a9b4",
        },
        "independent_cross_property": {
            "stimulus": "sample legal output stall, then present invalid input",
            "rtl_same_cycle_quarantine_observed": True,
            "frozen_ap_output_stable_under_stall_failed": True,
            "classification": (
                "P1 verification/specification conflict: the unconditional stall "
                "consequent does not permit the contract's intentional quarantine"
            ),
            "m133r2_status": "CLOSED_BY_SVA_TB_ONLY_REPAIR_AND_INDEPENDENT_VCS",
            "rtl_changed": False,
            "r2_production_overlap_cover_matches": 1,
            "r2_independent_sticky_fault_cycles": 3,
            "r2_assertion_failures": 0,
        },
        "logic_only_dc": {
            "tool": "Synopsys DC V-2023.12-SP3",
            "corner": "TSMC28 ssg0p9v125c",
            "clock_period_ns": 3.0,
            "clock_model": "ideal_unpropagated",
            "wireload": "ZeroWireload",
            "cell_area_um2": 10853.766052,
            "cells": 12667,
            "sequential_cells": 2767,
            "combinational_area_um2": 5275.493962,
            "sequential_area_um2": 5578.272090,
            "setup_worst_slack_ns": 1.1005,
            "hold_worst_slack_ns": 0.0001,
            "macro_count": 0,
            "net_interconnect_area": None,
            "constraint_violation_classes_clean": 5,
            "paper_ppa_ready": False,
        },
        "identity_hygiene": {
            "r1_correction_counterexample_tb_sha_still_exact": True,
            "r1_correction_interim_audit_sha_still_present": False,
            "impact": (
                "P2 evidence-lineage hygiene only; r2 exact source/run identity and "
                "the frozen r1 SVA counterexample are independently preserved"
            ),
            "final_manifest_supersession_overlay_requested": True,
        },
        "score": {
            "active_m133r2": 89,
            "p0": 0,
            "p1": 1,
            "p2": 3,
        },
        "scope_boundary": {
            "implemented": "standalone 512-bit input stream assembler",
            "bank_mapper": False,
            "two_physical_sram_rows_per_cycle": False,
            "foundry_macro": False,
            "bank_conflicts": False,
            "macro_inclusive_ppa": False,
            "physical_speedup": False,
            "system_speedup": False,
        },
    }
    (REVIEW / "m133_independent_audit.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
