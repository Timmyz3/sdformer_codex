#!/usr/bin/env python3
"""Fail-closed machine audit for the independent M131 hammer review."""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
M131_DC = ROOT / "dc_handoff/runs/m131_synthesis_safe_compact_canonical_k4_fold_logic_only_dc_3p000ns_exploratory_r1_20260824"
M128_DC = ROOT / "dc_handoff/runs/m128_descriptor_streamed_k4_fold_logic_only_dc_3p000ns_exploratory_r1_20260824"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_sha(path: Path, expected: str) -> str:
    require(path.is_file(), f"missing frozen input: {path}")
    observed = sha256(path)
    require(observed == expected, f"SHA mismatch for {path}: {observed} != {expected}")
    return observed


def verify_manifest(path: Path, base: Path) -> Dict[str, str]:
    observed = {}  # type: Dict[str, str]
    for line in read(path).splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"malformed SHA manifest line in {path}: {line!r}")
        expected, relative = match.groups()
        target = base / relative
        require(target.is_file(), f"manifest target missing: {target}")
        actual = sha256(target)
        require(actual == expected, f"manifest SHA mismatch: {target}")
        observed[relative] = actual
    require(bool(observed), f"empty manifest: {path}")
    return observed


def parse_pass_line(path: Path, prefix: str) -> Dict[str, object]:
    lines = [line for line in read(path).splitlines() if line.startswith(prefix)]
    require(len(lines) == 1, f"expected exactly one PASS line with prefix {prefix!r}")
    result = {}  # type: Dict[str, object]
    for token in lines[0][len(prefix):].strip().split():
        require("=" in token, f"malformed PASS token: {token}")
        key, value = token.split("=", 1)
        if value in {"true", "false"}:
            result[key] = value == "true"
        elif re.fullmatch(r"-?[0-9]+", value):
            result[key] = int(value)
        else:
            result[key] = value
    return result


def parse_covers(path: Path) -> Dict[str, int]:
    covers = {}  # type: Dict[str, int]
    for line in read(path).splitlines():
        match = re.search(r"\.sva\.(cp_[A-Za-z0-9_]+), .*?, ([0-9]+) match$", line)
        if match:
            covers[match.group(1)] = int(match.group(2))
    return covers


def parse_area(path: Path) -> float:
    match = re.search(r"^Total cell area:\s+([0-9.]+)$", read(path), re.MULTILINE)
    require(match is not None, f"cell area missing from {path}")
    return float(match.group(1))


def parse_qor(path: Path) -> Dict[str, int]:
    text = read(path)
    fields = {
        "logic_levels": r"Levels of Logic:\s+([0-9.]+)",
        "leaf_cells": r"Leaf Cell Count:\s+([0-9]+)",
        "sequential_cells": r"Sequential Cell Count:\s+([0-9]+)",
        "macro_count": r"Macro Count:\s+([0-9]+)",
    }
    result = {}  # type: Dict[str, int]
    for name, pattern in fields.items():
        match = re.search(pattern, text)
        require(match is not None, f"{name} missing from {path}")
        result[name] = int(float(match.group(1)))
    return result


def parse_worst_slack(path: Path) -> float:
    match = re.search(r"slack \(MET\)\s+(-?[0-9.]+)", read(path))
    require(match is not None, f"MET slack missing from {path}")
    return float(match.group(1))


FROZEN = {
    "contracts/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_contract_r1_20260824.json": "0e657b5916e428fe09df82588479654185055ab734b74a2782fc9b1ec9bae8ba",
    "contracts/m130_r1_dc_elaboration_failure_correction_r1_20260824.json": "9164e6b79846cd6017b03592847d54453d3e2cbfa65549e2cbb9ce281b7fc2ef",
    "contracts/m130_compact_canonical_k4_row_fold_vcs_contract_r1_20260824.json": "0a67fb7c1466257edc7c6d2cad960565c050916d8456addb3e0330025b8b911b",
    "rtl_m130/m130_compact_canonical_k4_row_fold.sv": "ff6d10d2fa341a4ef855f8df196542b990fd71fca34b1b3b81b04c5cb7588e96",
    "rtl_m131/m131_synthesis_safe_compact_canonical_k4_row_fold.sv": "82987dd367892213c3f57f0b17b5df4e92603653be9d8a093c9d9b2229cda4ea",
    "rtl_m128/m128_descriptor_streamed_k4_row_fold.sv": "b7c5c4c329bc4f1a7011398c5d3c20933dd8badfc4b2bbf3b213b15efe01e54d",
    "verif_m131/m131_synthesis_safe_compact_canonical_k4_row_fold_assertions.sv": "17b6493046088f28c6f824e18b3563d703d7c89b4d8d90b6e760135523c79cd4",
    "tb_m131/tb_m131_synthesis_safe_compact_canonical_k4_row_fold.sv": "c81d0cd1a12a5860d1712a71bd04d31960008ce3e21a3914618a30c89488c434",
    "dc_handoff/filelists/date_m131_synthesis_safe_compact_canonical_k4_row_fold_directed_vcs.f": "f65d8f05819ade452b06a4e8442c47e79ff74a52331afd43c08e63e597fd7013",
    "dc_handoff/filelists/date_m131_synthesis_safe_compact_canonical_k4_row_fold_logic_only_dc.f": "6015b365af52a5469e6d4e48661f2916e21c540a3f775e4124d2b2ffec1dced0",
    "dc_handoff/filelists/date_m128_descriptor_streamed_k4_row_fold_logic_only_dc.f": "b074cd225e1c17dfefffe5df97a334bd521ac0dbdee5a0c6a7eef162d2595fac",
    "dc_handoff/scripts/run_vcs_m131_synthesis_safe_compact_canonical_k4_row_fold.sh": "86c51f9d5246ddf86572e231fd09284055823bc35c654868135fac58d99f9887",
    "dc_handoff/scripts/run_dc_m97_m85_logic_only.tcl": "8d30dfd2a6b2480c538b751640aa17d52549162c35905de3bf384798ce3dfdde",
    "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc": "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    "docs/359_DATE终局冻结_20260813.md": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def main() -> None:
    frozen_hashes = {relative: verify_sha(ROOT / relative, expected) for relative, expected in FROZEN.items()}
    review = json.loads(read(REVIEW / "m131_synthesis_safe_compact_canonical_k4_row_fold_independent_hammer_review.json"))
    contract = json.loads(read(ROOT / "contracts/m131_synthesis_safe_compact_canonical_k4_row_fold_vcs_contract_r1_20260824.json"))

    require(review["score"]["overall"] == 92, "review score drift")
    require(review["severity_counts"] == {"P0": 0, "P1": 1, "P2": 4}, "severity count drift")
    require(contract["architecture"]["descriptor_payload_bits"] == 35, "contract payload width drift")
    require(contract["admission"]["complete_row_partition_losslessness"] is False, "losslessness improperly admitted")
    require(contract["admission"]["external_descriptor_producer_implemented"] is False, "producer improperly admitted")
    for key in ("dc_frequency_improvement", "physical_speedup", "system_speedup", "headline"):
        require(contract["admission"][key] is False, f"contract improperly admits {key}")

    preflight = read(REVIEW / "preflight_sha_checks.txt")
    preflight_lines = [line for line in preflight.splitlines() if line.strip()]
    require(len(preflight_lines) == 24, "preflight input count drift")
    require(all(re.search(r"expected=([0-9a-f]{64}) observed=\1$", line) for line in preflight_lines), "preflight SHA mismatch")
    vcs_manifest = verify_manifest(REVIEW / "vcs_output.sha256", REVIEW)
    dc_manifest = verify_manifest(REVIEW / "dc_output.sha256", REVIEW)

    production = parse_pass_line(
        REVIEW / "sealed_vcs_replay/sim.raw.log",
        "PASS M131 compact canonical K4 row fold VCS",
    )
    independent = parse_pass_line(
        REVIEW / "independent_vcs/sim.raw.log",
        "PASS M131 independent hammer",
    )
    require(read(REVIEW / "sealed_vcs_replay/compile.rc").strip() == "0", "production replay compile failed")
    require(read(REVIEW / "sealed_vcs_replay/sim.rc").strip() == "0", "production replay simulation failed")
    require(read(REVIEW / "independent_vcs/compile.rc").strip() == "0", "independent compile failed")
    require(read(REVIEW / "independent_vcs/sim.rc").strip() == "0", "independent simulation failed")
    require(production == {
        "groups": 237, "updates": 237, "sources": 691, "lanes": 22752,
        "done": 193, "done_overlap": 190, "stalls": 60, "long_stall": 17,
        "cross_row_updates": 64, "cross_row_ii1": 63, "plus512": 1,
        "protocol_attacks": 4, "reset_attacks": 1, "idle_payload": 1,
        "descriptor_bits": 35, "producer_implemented": False,
        "physical_speedup": False, "system_speedup": False, "headline": False,
    }, "production replay PASS metrics drift")

    required_independent = {
        "descriptor_bits": 35, "groups": 110, "updates": 109,
        "reset_aborted_descriptors": 1, "sources": 420, "lanes": 10464,
        "k1": 1, "k2": 7, "k3": 3, "k4": 99,
        "cross_group_ii1_intervals": 95, "cross_update_ii1_intervals": 95,
        "done": 104, "done_tags": 104, "done_overlap_next_row": 100,
        "output_stall_cycles": 73, "max_output_stall": 73,
        "group_stall_cycles": 73, "long_stall_replace": 1,
        "plus512": 7, "minus512": 7,
        "idle_payload_ready_checks": 16, "open_row_idle_payload_ready_checks": 1,
        "within_duplicate_attacks": 1, "within_descending_attacks": 1,
        "cross_repeat_attacks": 1, "cross_backtrack_attacks": 1,
        "row_identity_attacks": 1, "dirty_source_attacks": 1,
        "dirty_negate_attacks": 1, "nonlast_source15_attacks": 1,
        "cache_miss_attacks": 1, "block_attacks": 1, "reset_checks": 1,
        "gapped_partition_descriptors_accepted": 3,
        "internal_ready_valid_loop_observed": False,
        "predecessor_negative_index_present": False,
        "complete_row_partition_losslessness": False,
        "descriptor_producer_implemented": False,
        "descriptor_payload_bits_only": True,
        "dc_frequency_improvement": False, "physical_speedup": False,
        "system_speedup": False, "headline": False,
    }
    require(independent == required_independent, "independent PASS metrics drift")

    expected_covers = {
        "cp_k1_descriptor": 1,
        "cp_k4_descriptor": 99,
        "cp_tagged_done_overlaps_next_group": 100,
        "cp_reset_quiesce": 36,
        "cp_cross_row_replace": 95,
        "cp_multidescriptor_row": 1,
        "cp_update_stall_release": 1,
    }
    require(parse_covers(REVIEW / "independent_vcs/assert.report") == expected_covers, "independent SVA cover drift")
    for path in (
        REVIEW / "sealed_vcs_replay/sim.raw.log",
        REVIEW / "sealed_vcs_replay/assert.report",
        REVIEW / "independent_vcs/sim.raw.log",
        REVIEW / "independent_vcs/assert.report",
    ):
        require(re.search(r"failed at|Offending|^Error|^Fatal|watchdog timeout", read(path), re.I | re.M) is None,
                f"simulation failure marker found in {path}")

    m130_rtl = read(ROOT / "rtl_m130/m130_compact_canonical_k4_row_fold.sv")
    m131_rtl = read(ROOT / "rtl_m131/m131_synthesis_safe_compact_canonical_k4_row_fold.sv")
    require(m130_rtl.count("group_source[pick-1]") == 1, "M130 predecessor expression count drift")
    require("group_source[pick-1]" not in m131_rtl, "negative predecessor expression remains in M131")
    for fixed in (
        "group_source[1] <= group_source[0]",
        "group_source[2] <= group_source[1]",
        "group_source[3] <= group_source[2]",
        "assign group_ready = group_capacity",
        "&& (!group_valid || group_semantically_valid);",
        "assign done_valid = update_accept && update_last;",
        "assign done_block = update_block;",
        "assign done_row = update_row;",
    ):
        require(fixed in m131_rtl, f"M131 static contract missing: {fixed}")
    require(3 + 9 + 2 + 16 + 4 + 1 == 35, "descriptor payload arithmetic failed")

    dc_text = read(REVIEW / "dc_elaboration/dc.raw.log")
    check_design = read(REVIEW / "dc_elaboration/reports/check_design.rpt")
    require(read(REVIEW / "dc_elaboration/dc.rc").strip() == "0", "independent DC elaboration failed")
    require("PASS M131 independent DC analyze_elaborate_check_design no_elab312=true negative_index=false compile_run=false physical_speedup=false" in dc_text,
            "independent DC PASS marker missing")
    require(re.search(r"ELAB-312|group_source\[-1\]|out[- ]of[- ]bounds|^Error:", dc_text + check_design, re.I | re.M) is None,
            "negative-index/elaboration failure marker found")
    lint1 = int(re.search(r"Cells do not drive \(LINT-1\)\s+([0-9]+)", check_design).group(1))
    lint31 = int(re.search(r"Shorted outputs \(LINT-31\)\s+([0-9]+)", check_design).group(1))
    require((lint1, lint31) == (386, 780), "independent check_design warning class drift")
    require(check_design.count("Warning:") == 1166, "independent check_design warning total drift")

    m131 = {
        "cell_area_um2": parse_area(M131_DC / "reports/area.rpt"),
        **parse_qor(M131_DC / "reports/qor.rpt"),
        "worst_setup_slack_ns": parse_worst_slack(M131_DC / "reports/timing_setup.rpt"),
        "worst_hold_slack_ns": parse_worst_slack(M131_DC / "reports/timing_hold.rpt"),
    }
    m128 = {
        "cell_area_um2": parse_area(M128_DC / "reports/area.rpt"),
        **parse_qor(M128_DC / "reports/qor.rpt"),
        "worst_setup_slack_ns": parse_worst_slack(M128_DC / "reports/timing_setup.rpt"),
        "worst_hold_slack_ns": parse_worst_slack(M128_DC / "reports/timing_hold.rpt"),
    }
    require(m131 == {
        "cell_area_um2": 89467.055598, "logic_levels": 32, "leaf_cells": 109277,
        "sequential_cells": 14798, "macro_count": 0,
        "worst_setup_slack_ns": 0.6733, "worst_hold_slack_ns": 0.0001,
    }, "M131 full DC metric drift")
    require(m128 == {
        "cell_area_um2": 89045.585598, "logic_levels": 41, "leaf_cells": 107287,
        "sequential_cells": 14782, "macro_count": 0,
        "worst_setup_slack_ns": 0.3387, "worst_hold_slack_ns": 0.0005,
    }, "M128 full DC metric drift")
    delta = {
        "cell_area_um2": round(m131["cell_area_um2"] - m128["cell_area_um2"], 6),
        "cell_area_percent": (m131["cell_area_um2"] / m128["cell_area_um2"] - 1.0) * 100.0,
        "logic_levels": m131["logic_levels"] - m128["logic_levels"],
        "leaf_cells": m131["leaf_cells"] - m128["leaf_cells"],
        "sequential_cells": m131["sequential_cells"] - m128["sequential_cells"],
        "setup_slack_ns": round(m131["worst_setup_slack_ns"] - m128["worst_setup_slack_ns"], 4),
        "hold_slack_ns": round(m131["worst_hold_slack_ns"] - m128["worst_hold_slack_ns"], 4),
    }
    require(math.isclose(delta["cell_area_percent"], 0.47331936464851676, rel_tol=0.0, abs_tol=1e-12), "area delta drift")
    require(delta == {
        "cell_area_um2": 421.47,
        "cell_area_percent": delta["cell_area_percent"],
        "logic_levels": -9,
        "leaf_cells": 1990,
        "sequential_cells": 16,
        "setup_slack_ns": 0.3346,
        "hold_slack_ns": -0.0004,
    }, "M128/M131 comparison drift")

    dc_report_hashes = {}  # type: Dict[str, str]
    for run_name, run_path in (("m131", M131_DC), ("m128", M128_DC)):
        for relative in (
            "dc.log", "reports/qor.rpt", "reports/area.rpt",
            "reports/timing_setup.rpt", "reports/timing_hold.rpt",
            "reports/constraint_violators.rpt", "reports/check_design_precompile.rpt",
            "reports/check_design_postcompile.rpt", "reports/check_timing_postcompile.rpt",
            "reports/resources_postcompile.rpt",
        ):
            dc_report_hashes[f"{run_name}/{relative}"] = sha256(run_path / relative)

    output = {
        "schema": "m131_independent_machine_audit_v1",
        "status": "PASS_FAIL_CLOSED_CONDITIONAL_M131_SYNTHESIS_INDEX_REPAIR",
        "review_score": review["score"],
        "severity_counts": review["severity_counts"],
        "production_exact_sha_vcs_replay": production,
        "independent_adversarial_vcs": independent,
        "independent_sva_covers": expected_covers,
        "ready_valid_and_tagged_done": {
            "group_valid_zero_ready_payload_independent_checks": 17,
            "internal_combinational_ready_valid_cycle_observed": False,
            "cross_row_descriptor_ii1_intervals": 95,
            "cross_row_update_ii1_intervals": 95,
            "tagged_done_overlap_next_row_checks": 100,
            "wrong_done_tag_checks": 0,
        },
        "synthesis_repair": {
            "m130_predecessor_expression_occurrences": 1,
            "m131_predecessor_expression_occurrences": 0,
            "independent_dc_analyze_elaborate_check_design": True,
            "elab312_matches": 0,
            "negative_index_matches": 0,
            "compile_performed_by_independent_dc_audit": False,
            "precompile_check_design_warnings": {"LINT-1": lint1, "LINT-31": lint31, "total": lint1 + lint31},
        },
        "exploratory_logic_only_dc_3ns": {
            "m128": m128,
            "m131": m131,
            "m131_minus_m128": delta,
            "same_tool_tcl_sdc_period_corners": True,
            "exact_sha_launch_manifest_available": False,
            "maximum_frequency_sweep_performed": False,
            "macro_inclusive": False,
            "physical_speedup_admitted": False,
        },
        "claim_boundary": {
            "descriptor_payload_bits": 35,
            "descriptor_payload_only": True,
            "external_descriptor_producer_implemented": False,
            "complete_row_partition_losslessness": False,
            "frequency_improvement_admitted": False,
            "physical_speedup_admitted": False,
            "system_speedup_admitted": False,
            "headline_ready": False,
        },
        "frozen_input_sha256": frozen_hashes,
        "review_vcs_output_sha256": vcs_manifest,
        "review_dc_output_sha256": dc_manifest,
        "exploratory_dc_report_sha256": dc_report_hashes,
    }
    target = REVIEW / "m131_independent_machine_audit.json"
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS M131 machine audit score=92 P0=0 P1=1 P2=4 producer=false lossless=false physical_speedup=false headline=false")


if __name__ == "__main__":
    main()
