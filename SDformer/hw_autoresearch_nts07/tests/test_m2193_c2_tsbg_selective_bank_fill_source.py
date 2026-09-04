#!/usr/bin/python3.12
"""Source-only M2193 tests. Never invoke VCS, EDA, lmutil, or a GPU."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
RTL = HW / "rtl_m2193/m2193_c2_tsbg_b4_selective_bank_fill_frontend.sv"
M2018 = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
SVA = HW / "verif_m2193/m2193_c2_tsbg_selective_bank_fill_assertions.sv"
TB = HW / "tb_m2193/tb_m2193_c2_tsbg_selective_bank_fill_directed.sv"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2193_c2_tsbg_selective_bank_fill_directed_vcs.f"
PARSER = HW / "system_simulator/scripts/parse_m2195_m2193_c2_tsbg_selective_bank_fill_directed_vcs.py"
RUNNER = HW / "dc_handoff/scripts/run_m2195_m2194_m2193_selective_bank_fill_directed_vcs_one_shot.sh"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stripped(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


def balanced_sv(path: Path) -> None:
    text = stripped(path.read_text())
    for opening, closing in (("module", "endmodule"), ("function", "endfunction"),
                             ("task", "endtask"), ("case", "endcase"),
                             ("generate", "endgenerate"), ("begin", "end")):
        assert len(re.findall(rf"\b{opening}\b", text)) == len(
            re.findall(rf"\b{closing}\b", text)), (path, opening, closing)
    pairs = {')': '(', ']': '[', '}': '{'}
    stack: list[str] = []
    for char in text:
        if char in "([{": stack.append(char)
        elif char in ")]}":
            assert stack and stack.pop() == pairs[char], (path, char)
    assert not stack, (path, stack)


def semantic_errors(text: str) -> list[str]:
    checks = {
        "new_module": "module m2193_c2_tsbg_b4_selective_bank_fill_frontend" in text,
        "bank_valid_array": "logic [7:0] cache_bank_valid_q [0:CACHE_ROWS-1][0:1];" in text,
        "b4_union": "for (int ctx = 0; ctx < BUNDLE; ctx++) begin\n            find_needed_mask[0] |= active_row_q[ctx][find_group][7:0];" in text,
        "coverage_hit_low": "find_needed_mask[0] & ~cache_bank_valid_q[entry][0]" in text,
        "coverage_hit_high": "find_needed_mask[1]\n                             & ~cache_bank_valid_q[entry][1]" in text,
        "missing_only_low": "find_missing_mask[0] = find_needed_mask[0]\n                    & ~cache_bank_valid_q[entry][0];" in text,
        "request_mask": "assign core_req_bank_valid = fill_missing_mask_q[fill_half_q];" in text,
        "request_popcount": "assign core_req_source_count = popcount8(core_req_bank_valid);" in text,
        "response_mask": "core_rsp_bank_valid == fill_missing_mask_q[fill_half_q]" in text,
        "merge_only_returned": "if (core_rsp_bank_valid[bank]) begin" in text,
        "valid_after_last_slice": "cache_bank_valid_q[fill_cache_q][fill_half_q]\n                        <= cache_bank_valid_q[fill_cache_q][fill_half_q]\n                           | fill_missing_mask_q[fill_half_q];" in text,
        "partial_counter": "partial_hit_count_q <= partial_hit_count_q + 1'b1;" in text,
        "select_nonzero_half": "fill_half_q <= find_missing_mask[0] != 0 ? 1'b0 : 1'b1;" in text,
        "m803_reused": "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter" in text,
        "private_acc24": "logic signed [23:0] acc_q" in text,
        "private_sign": "current_sign_row_q" in text,
        "private_terminal": "commit_terminal = commit_slice_q == OUTPUT_SLICES - 1;" in text,
    }
    return [name for name, value in checks.items() if not value]


def load_parser():
    spec = importlib.util.spec_from_file_location("m2195_parser", PARSER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_parser_reject(module, sim: str, build: str = "clean\n", rc: str = "0\n") -> None:
    with tempfile.TemporaryDirectory(prefix="m2193_parser_mut_") as td:
        root = Path(td)
        (root / "sim.log").write_text(sim)
        (root / "compile.log").write_text(build)
        (root / "rc").write_text(rc)
        try:
            module.parse(root / "sim.log", root / "compile.log", root / "rc")
        except module.Failure:
            return
        raise AssertionError("parser mutation accepted")


def main() -> int:
    assert sha(M2018) == "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21"
    assert sha(M803) == "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156"
    assert sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    compile(PARSER.read_text(), str(PARSER), "exec")
    for path in (RTL, SVA, TB): balanced_sv(path)
    source = RTL.read_text()
    assert not semantic_errors(source), semantic_errors(source)

    mutations = [
        ("logic [7:0] cache_bank_valid_q [0:CACHE_ROWS-1][0:1];", "logic cache_row_only_q;"),
        ("find_needed_mask[0] |= active_row_q[ctx][find_group][7:0];", "find_needed_mask[0] = active_row_q[0][find_group][7:0];"),
        ("find_needed_mask[1]\n                             & ~cache_bank_valid_q[entry][1]", "find_needed_mask[1]"),
        ("find_missing_mask[0] = find_needed_mask[0]\n                    & ~cache_bank_valid_q[entry][0];", "find_missing_mask[0] = find_needed_mask[0];"),
        ("assign core_req_bank_valid = fill_missing_mask_q[fill_half_q];", "assign core_req_bank_valid = 8'hff;"),
        ("assign core_req_source_count = popcount8(core_req_bank_valid);", "assign core_req_source_count = 4'd8;"),
        ("core_rsp_bank_valid == fill_missing_mask_q[fill_half_q]", "core_rsp_bank_valid == 8'hff"),
        ("if (core_rsp_bank_valid[bank]) begin", "if (1'b1) begin"),
        ("| fill_missing_mask_q[fill_half_q];", "| 8'hff;"),
        ("partial_hit_count_q <= partial_hit_count_q + 1'b1;", "partial_hit_count_q <= partial_hit_count_q;"),
        ("fill_half_q <= find_missing_mask[0] != 0 ? 1'b0 : 1'b1;", "fill_half_q <= 1'b0;"),
        ("m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter", "missing_adapter adapter"),
    ]
    for old, new in mutations:
        assert old in source
        assert semantic_errors(source.replace(old, new, 1)), (old, new)

    expected_filelist = [
        "hw_autoresearch_nts07/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "hw_autoresearch_nts07/rtl_m2193/m2193_c2_tsbg_b4_selective_bank_fill_frontend.sv",
        "hw_autoresearch_nts07/verif_m2193/m2193_c2_tsbg_selective_bank_fill_assertions.sv",
        "hw_autoresearch_nts07/tb_m2193/tb_m2193_c2_tsbg_selective_bank_fill_directed.sv",
    ]
    assert FILELIST.read_text().splitlines() == expected_filelist
    m803 = M803.read_text()
    assert "core_req_source_count == req_mask_count" in m803
    assert "core_req_bank_valid != 0" in m803
    assert "pending_mask_q[bank]" in m803
    assert "incoming_mask[slot]" in m803

    sva = SVA.read_text()
    for token in ("cp_partial_refill", "cp_eviction", "cp_selective_request",
                  "cp_independent_bank_backpressure", "cp_response_reorder",
                  "cp_bridge_backpressure", "cp_commit_backpressure",
                  "cp_positive_source", "cp_negative_source", "cp_terminal",
                  "cp_zero_descriptor_skip"):
        assert token in sva
    tb = TB.read_text()
    for token in ("weight_value = -128", "ordinary Acc24 mismatch",
                  "TSBG Acc24 mismatch", "phase0 ordinary selective mask mismatch",
                  "phase1 ordinary missing-only mask mismatch", "send_bundle(0)",
                  "send_bundle(1)", "send_bundle(2)"):
        assert token in tb

    parser = load_parser()
    cover = ("M2193_COVER partial_o=1 partial_t=1 eviction_o=2 eviction_t=2 "
             "reorder_o=3 reorder_t=3 reqstall_o=4 reqstall_t=4 bridgestall_o=5 "
             "bridgestall_t=5 commitstall_o=6 commitstall_t=6 zero_o=8 zero_t=8\n")
    passed = ("PASS_M2193_C2_TSBG_SELECTIVE_BANK_FILL_DIRECTED bundles=3 commits_o=72 "
              "commits_t=72 partial_o=1 partial_t=1 refills_o=120 refills_t=120 "
              "scalar_o=120 scalar_t=120 products_o=4096 products_t=4096\n")
    with tempfile.TemporaryDirectory(prefix="m2193_parser_control_") as td:
        root = Path(td)
        (root / "sim.log").write_text(cover + passed)
        (root / "compile.log").write_text("clean\n")
        (root / "rc").write_text("0\n")
        result = parser.parse(root / "sim.log", root / "compile.log", root / "rc")
        assert result["status"] == "RAW_PASS_M2195_M2193_DIRECTED_VCS_PENDING_M2196_RESULT_HAMMER"
    expect_parser_reject(parser, cover.replace("partial_o=1", "partial_o=0") + passed)
    expect_parser_reject(parser, cover + passed.replace("scalar_o=120", "scalar_o=119"))
    expect_parser_reject(parser, cover + passed + "$fatal\n")
    expect_parser_reject(parser, cover + passed + passed)
    expect_parser_reject(parser, cover + passed, rc="1\n")

    runner = RUNNER.read_text()
    assert runner.count('"${VCS}" -full64') == 1
    assert runner.count('"${LMUTIL}" lmstat ') == 1
    assert "all_other_eda_runs':0" in runner and "automatic_retry':False" in runner
    result = HW / "results/m2195_m2193_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904"
    attempt = HW / "results/.m2195_m2193_selective_bank_fill_vcs_attempt_consumed"
    lock = HW / "results/.m2195_m2193_selective_bank_fill_vcs_launch_lock"
    assert not result.exists() and not attempt.exists() and not lock.exists()
    print("PASS_M2193_SOURCE_TESTS semantic_mutations=12 parser_control=1 parser_mutations=5 sv_balance=3 vcs_runs=0 eda_runs=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
