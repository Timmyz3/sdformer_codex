#!/opt/anaconda3/bin/python3.12
"""Static, no-EDA tests for the M2213 causal-ablation source package."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path


HW = Path(__file__).resolve().parents[1]
RTL = HW / "rtl_m2213/m2213_c2_tsbg_b4_postread_causal_frontend.sv"
M2018 = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
SVA = HW / "verif_m2213/m2213_c2_tsbg_postread_causal_assertions.sv"
TB = HW / "tb_m2213/tb_m2213_c2_tsbg_preread_postread_causal_directed.sv"
PARSER = HW / "system_simulator/scripts/parse_m2215_m2213_preread_postread_causal_directed_vcs.py"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2213_preread_postread_causal_directed_vcs.f"
RUNNER = HW / "dc_handoff/scripts/run_m2215_m2214_m2213_preread_postread_causal_directed_vcs_one_shot.sh"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def audit(rtl: str, sva: str, tb: str, parser: str, runner: str) -> None:
    need("module m2213_c2_tsbg_b4_postread_causal_frontend" in rtl,
         "post-read module")
    need("parameter int SCHEDULE_MODE" not in rtl, "post-read must be group-major only")
    need("localparam int ORDER_INDEX = map_group * BUNDLE + map_ctx" in rtl,
         "group-major order")
    need("postread_hit_q <= 1;" in rtl and "state_q <= ST_FETCH_REQ;" in rtl,
         "hit must enter real fetch")
    need("if (!postread_hit_q) begin" in rtl and "cache_weight_q[fill_cache_q]" in rtl,
         "late response payload discard")
    need("core_rsp_identity_legal" in rtl and
         "postread_identity_accept_count_q" in rtl,
         "return identity observation")
    for counter in ("debug_postread_row_count", "debug_postread_bundle_request_count",
                    "debug_postread_bundle_response_count",
                    "debug_postread_bank_request_count",
                    "debug_postread_bank_response_count",
                    "debug_postread_identity_accept_count"):
        need(counter in rtl and counter in tb, f"observable counter {counter}")
    need("debug_postread_row_count * 96" in sva and
         "debug_postread_row_count * 12" in sva,
         "SVA causal conservation")
    need("ordinary_reads=%0d postread_reads=%0d preread_reads=%0d" in tb,
         "three-axis output")
    need("`CONNECT_FROZEN(dut_ordinary, ordinary, 0" in tb and
         "`CONNECT_FROZEN(dut_preread, preread, 1" in tb and
         "m2213_c2_tsbg_b4_postread_causal_frontend" in tb,
         "three exact axes")
    need("ordinary.scalar_bank_request_count != ROWS * BANKS_PER_ROW" in tb and
         "postread.scalar_bank_request_count != ROWS * BANKS_PER_ROW" in tb and
         "preread.scalar_bank_request_count != GROUPS * BANKS_PER_ROW" in tb,
         "fair request ledger")
    need("postread.product_count != EXPECTED_PRODUCTS" in tb and
         "preread.product_count != EXPECTED_PRODUCTS" in tb,
         "product equality")
    need("mismatch_o + mismatch_l + mismatch_p" in tb,
         "three-axis golden mismatch ledger")
    need(not re.search(r"(?m)^\\`", tb), "escaped preprocessor token")
    need(parser.count("PASS_RE") >= 2 and parser.count("COVER_RE") >= 2,
         "strict parser")
    need("(2304, 2304, 576, 1728)" in parser and
         "216, 216, 1728, 1728, 216" in parser,
         "parser exact causal ledgers")
    need("claim_boundary" in parser and '"energy": False' in parser and
         '"paper_citable": False' in parser,
         "parser claim boundary")
    need("M2215_EXPECTED_RUNNER_SHA256" in runner and
         "M2215_EXPECTED_M2214_REVIEW_SHA256" in runner,
         "launch identity pins")
    need("PASS_M2214_M2213_SOURCE_HAMMER__M2215_ONE_SHOT_VCS_AUTHORIZED" in runner,
         "review gate")
    need("all_other_eda_runs':0" in runner and "automatic_retry':False" in runner,
         "execution budget")
    need("rm -rf -- \"${WORK}/csrc\" \"${WORK}/simv.daidir\" \"${WORK}/simv.vdb\"" in runner,
         "build-only cleanup")


def main() -> None:
    need(sha(M2018) == "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
         "frozen M2018 drift")
    need(sha(M803) == "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
         "frozen M803 drift")
    need(sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs359 drift")
    expected_filelist = [
        "hw_autoresearch_nts07/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "hw_autoresearch_nts07/rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv",
        "hw_autoresearch_nts07/rtl_m2213/m2213_c2_tsbg_b4_postread_causal_frontend.sv",
        "hw_autoresearch_nts07/verif_m2213/m2213_c2_tsbg_postread_causal_assertions.sv",
        "hw_autoresearch_nts07/tb_m2213/tb_m2213_c2_tsbg_preread_postread_causal_directed.sv",
    ]
    need(FILELIST.read_text().splitlines() == expected_filelist, "filelist order")
    rtl, sva, tb, parser, runner = (
        path.read_text() for path in (RTL, SVA, TB, PARSER, RUNNER))
    audit(rtl, sva, tb, parser, runner)
    subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)], check=True,
                   capture_output=True, text=True, timeout=30)

    mutations = [
        (rtl.replace("postread_hit_q <= 1;", "postread_hit_q <= 0;", 1), sva, tb, parser, runner),
        (rtl.replace("state_q <= ST_FETCH_REQ;", "state_q <= ST_BRIDGE;"), sva, tb, parser, runner),
        (rtl.replace("postread_identity_accept_count_q", "removed_identity_counter"), sva, tb, parser, runner),
        (rtl, sva.replace("debug_postread_row_count * 96", "debug_postread_row_count * 8"), tb, parser, runner),
        (rtl, sva, tb.replace("`CONNECT_FROZEN(dut_preread", "// missing preread", 1), parser, runner),
        (rtl, sva, tb.replace("postread.product_count != EXPECTED_PRODUCTS", "1'b0", 1), parser, runner),
        (rtl, sva, "\\`timescale" + tb.split("`timescale", 1)[1], parser, runner),
        (rtl, sva, tb, parser.replace("(2304, 2304, 576, 1728)", "(2304, 2304, 2304, 0)", 1), runner),
        (rtl, sva, tb, parser.replace('"energy": False', '"energy": True', 1), runner),
        (rtl, sva, tb, parser, runner.replace("automatic_retry':False", "automatic_retry':True", 1)),
    ]
    rejected = 0
    for variant in mutations:
        try:
            audit(*variant)
        except AssertionError:
            rejected += 1
    need(rejected == len(mutations) == 10, "mutation rejection count")
    print("PASS_M2213_SOURCE_TESTS frozen_m2018=1 three_axes=1 "
          "postread_real_request_response=1 exact_commit_product=1 "
          "runner_bash=1 mutations=10 vcs_runs=0 license_queries=0 "
          "eda_runs=0 gpu_runs=0")


if __name__ == "__main__":
    main()
