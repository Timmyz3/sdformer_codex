#!/usr/bin/python3.12
"""Independent, source-only M2194 hammer.  Never invokes VCS/EDA/license/GPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent / "mechanical_checks.json"

RTL = HW / "rtl_m2193/m2193_c2_tsbg_b4_selective_bank_fill_frontend.sv"
SVA = HW / "verif_m2193/m2193_c2_tsbg_selective_bank_fill_assertions.sv"
TB = HW / "tb_m2193/tb_m2193_c2_tsbg_selective_bank_fill_directed.sv"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2193_c2_tsbg_selective_bank_fill_directed_vcs.f"
PARSER = HW / "system_simulator/scripts/parse_m2195_m2193_c2_tsbg_selective_bank_fill_directed_vcs.py"
RUNNER = HW / "dc_handoff/scripts/run_m2195_m2194_m2193_selective_bank_fill_directed_vcs_one_shot.sh"
TEST = HW / "tests/test_m2193_c2_tsbg_selective_bank_fill_source.py"
CONTRACT = HW / "contracts/m2193_c2_tsbg_selective_bank_fill_source_contract_r1_20260904.json"
M2018 = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
AUTHOR = HW / "reviews/m2193_m2184_c2_tsbg_selective_bank_fill_source_author_receipt_r1_20260904"
M2184 = HW / "reviews/m2184_m2175_ep34_tsbg_union_bank_fair_cpu_quickkill_result_hammer_r1_20260904"


EXPECTED = {
    RTL: "f651ea3a3b4dfab04d021a1e44797e7ab72c244cb7edf7496e18ac1ac033339e",
    SVA: "98ffb5200398dfff158f314dbbcd8d494cc362f90a000ef4b9dc59323c669459",
    TB: "39e0c54d341fa360e23815aa94675d694606fe2c94c5f8a28efc40308bd15be7",
    FILELIST: "068d689e7946e6aa69dd9a8f6e9abef33b3135e0bf0d4ae75c2e5bf6a8cd5230",
    PARSER: "d3388102f9eee653821f803f10112a15e54fbc61d9ce6f8b85afc19babc31a54",
    RUNNER: "183afb85b20e84cda660bf3f56e1fac91a63fb9beb7c5f9e7465438cf46e9759",
    TEST: "3898553e357d21d805eedf1ac3481c407fb43df4dd78c4c0056bc64e5f3a2ae1",
    CONTRACT: "578cc732b9d245e562d2797d5027984296f1ff959e47e967d706a3b559d7baa3",
    M2018: "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    M803: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exhaustive_seal(directory: Path) -> None:
    assert directory.is_dir() and not directory.is_symlink()
    assert not any(path.is_symlink() for path in directory.rglob("*"))
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    listed = []
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        listed.append(name)
        assert sha(directory / name) == digest
    actual = sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path.name not in {manifest.name, outer.name}
    )
    assert sorted(listed) == actual
    digest, name = outer.read_text().split(maxsplit=1)
    assert name.strip().lstrip("*") == manifest.name and sha(manifest) == digest


def main() -> int:
    for path, expected in EXPECTED.items():
        assert path.is_file() and not path.is_symlink()
        assert sha(path) == expected, path
    exhaustive_seal(AUTHOR)
    exhaustive_seal(M2184)
    assert json.loads(M2184.joinpath("review.json").read_text())["status"] == (
        "PASS_M2184_M2175_CPU_QUICKKILL_HAMMER__GO_RTL_CONSIDERATION_ONLY"
    )

    official = subprocess.run(
        ["/opt/anaconda3/bin/python3.12", str(TEST)], cwd=ROOT,
        text=True, capture_output=True, check=True,
    )
    assert "semantic_mutations=12 parser_control=1 parser_mutations=5" in official.stdout
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    compile(PARSER.read_text(), str(PARSER), "exec")

    rtl = RTL.read_text()
    sva = SVA.read_text()
    tb = TB.read_text()
    m803 = M803.read_text()
    runner = RUNNER.read_text()

    semantic_tokens = {
        "b4_union_low": "find_needed_mask[0] |= active_row_q[ctx][find_group][7:0]",
        "b4_union_high": "find_needed_mask[1] |= active_row_q[ctx][find_group][15:8]",
        "coverage_hit_low": "find_needed_mask[0] & ~cache_bank_valid_q[entry][0]",
        "coverage_hit_high": "find_needed_mask[1]\n                             & ~cache_bank_valid_q[entry][1]",
        "missing_only_request": "assign core_req_bank_valid = fill_missing_mask_q[fill_half_q]",
        "popcount_charge": "assign core_req_source_count = popcount8(core_req_bank_valid)",
        "response_mask_identity": "core_rsp_bank_valid == fill_missing_mask_q[fill_half_q]",
        "merge_returned_only": "if (core_rsp_bank_valid[bank]) begin",
        "valid_after_slice5": "if (fill_slice_q == OUTPUT_SLICES - 1) begin",
        "per_half_bank_valid": "logic [7:0] cache_bank_valid_q [0:CACHE_ROWS-1][0:1]",
        "private_acc24": "logic signed [23:0] acc_q [0:BUNDLE-1]",
        "private_sign": "logic [SOURCES_PER_GROUP-1:0] sign_row_q\n        [0:BUNDLE-1]",
        "private_tag": "logic [TAG_BITS-1:0] context_tag_q [0:BUNDLE-1]",
        "private_terminal": "commit_terminal = commit_slice_q == OUTPUT_SLICES - 1",
    }
    assert all(token in rtl for token in semantic_tokens.values())
    assert rtl.count("m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter") == 1
    assert "core_req_source_count == req_mask_count" in m803
    assert "core_req_bank_valid != 0" in m803
    assert "pending_mask_q[bank]" in m803
    assert "incoming_mask[slot]" in m803
    assert "bank_rsp_slot[bank] == slot[2:0]" in m803
    assert "bank_rsp_valid[bank] && rsp_shape_legal[bank]" in m803

    # Both instances are the same wrapper and differ only by SCHEDULE_MODE.
    assert tb.count("m2193_directed_side #(.SCHEDULE_MODE(") == 2
    assert "SCHEDULE_MODE(0)) ordinary" in tb
    assert "SCHEDULE_MODE(1)) tsbg" in tb
    assert tb.count(".SOURCE_GROUPS(SOURCE_GROUPS)") == 1

    required_covers = (
        "cp_partial_refill", "cp_eviction", "cp_selective_request",
        "cp_independent_bank_backpressure", "cp_response_reorder",
        "cp_bridge_backpressure", "cp_commit_backpressure",
        "cp_positive_source", "cp_negative_source", "cp_terminal",
        "cp_zero_descriptor_skip",
    )
    assert all(token in sva for token in required_covers)
    assert all(token in tb for token in (
        "weight_value = -128", "ordinary Acc24 mismatch", "TSBG Acc24 mismatch",
        "phase0 ordinary selective mask mismatch",
        "phase1 ordinary missing-only mask mismatch",
    ))

    # Contracted identity verification is incomplete: tag is wired but never
    # checked against a golden value, and is absent from the stall-stability SVA.
    commit_tag_mentions = len(re.findall(r"commit_tag_[ot]", tb))
    tag_comparison = bool(re.search(r"commit_tag_[ot]\s*(?:!==|!=|==|===)", tb))
    tag_in_sva_interface = bool(re.search(r"input\s+logic\s+\[TAG_BITS-1:0\]\s+commit_tag", sva))
    tag_in_hold = bool(re.search(r"\$stable\s*\([^)]*commit_tag", sva, re.S))
    assert commit_tag_mentions > 0
    assert not tag_comparison and not tag_in_sva_interface and not tag_in_hold

    # Future M2195 budget and state remain closed.
    assert runner.count('"${LMUTIL}" lmstat ') == 1
    assert runner.count('"${VCS}" -full64') == 1
    assert runner.count('300s "${WORK}/simv"') == 1
    assert "automatic_retry':False" in runner
    census = []
    for pattern in (
        "results/m2195_m2193_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904*",
        "results/.m2195_m2193_selective_bank_fill_vcs_attempt_consumed*",
        "results/.m2195_m2193_selective_bank_fill_vcs_launch_lock*",
        "results/.m2195_m2193_selective_bank_fill_vcs_work*",
    ):
        census.extend(str(path.relative_to(HW)) for path in HW.glob(pattern))
    assert census == [], census

    result = {
        "schema": "m2194_m2193_c2_tsbg_selective_bank_fill_source_mechanical_checks_r1_v1",
        "status": "PASS_MECHANICAL_CHECKS_WITH_ONE_CONTRACT_GAP",
        "identity_files_exact": len(EXPECTED),
        "m2184_and_author_double_seals_exhaustive": True,
        "official_source_suite": official.stdout.strip(),
        "semantic_mutations_rejected": 12,
        "parser_control_passed": 1,
        "parser_mutations_rejected": 5,
        "union_coverage_missing_popcount_merge_static_checks": len(semantic_tokens),
        "m803_arbitrary_nonzero_mask_and_reorder_static_checks": 6,
        "ordinary_tsbg_same_wrapper_and_ports": True,
        "directed_acc24_masks_backpressure_and_zero_checks_present": True,
        "commit_tag_mentions_in_tb": commit_tag_mentions,
        "commit_tag_golden_comparison_present": tag_comparison,
        "commit_tag_sva_port_present": tag_in_sva_interface,
        "commit_tag_stall_stability_present": tag_in_hold,
        "m2195_result_attempt_work_lock_census": census,
        "docs359_sha256": sha(DOCS359),
        "execution": {"vcs": 0, "simv": 0, "eda": 0, "license": 0, "gpu": 0},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS_M2194_MECHANICAL_WITH_P1 commit_tag_golden=0 commit_tag_stall_sva=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
