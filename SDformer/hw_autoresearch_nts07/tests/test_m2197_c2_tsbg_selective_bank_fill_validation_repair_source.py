#!/usr/bin/python3.12
"""M2197 validation-repair source tests; never run VCS, EDA, a license, or GPU."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
BASE_TEST = HW / "tests/test_m2193_c2_tsbg_selective_bank_fill_source.py"
RTL = HW / "rtl_m2193/m2193_c2_tsbg_b4_selective_bank_fill_frontend.sv"
M2018 = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
OLD_SVA = HW / "verif_m2193/m2193_c2_tsbg_selective_bank_fill_assertions.sv"
OLD_TB = HW / "tb_m2193/tb_m2193_c2_tsbg_selective_bank_fill_directed.sv"
SVA = HW / "verif_m2197/m2197_c2_tsbg_selective_bank_fill_assertions.sv"
TB = HW / "tb_m2197/tb_m2197_c2_tsbg_selective_bank_fill_directed.sv"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2197_c2_tsbg_selective_bank_fill_directed_vcs.f"
PARSER = HW / "system_simulator/scripts/parse_m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs.py"
RUNNER = HW / "dc_handoff/scripts/run_m2199_m2198_m2197_selective_bank_fill_directed_vcs_one_shot.sh"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2194 = HW / "reviews/m2194_m2193_c2_tsbg_selective_bank_fill_source_hammer_r1_20260904/review.json"


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
        if char in "([{":
            stack.append(char)
        elif char in ")]} ".replace(" ", ""):
            assert stack and stack.pop() == pairs[char], (path, char)
    assert not stack, (path, stack)


def validation_errors(tb: str, sva: str) -> list[str]:
    checks = {
        "separate_golden_tag_arrays":
            "logic [23:0] golden_tag_o [0:3];" in tb
            and "logic [23:0] golden_tag_t [0:3];" in tb,
        "ordinary_context_tag_mapping":
            "golden_tag_o[ctx] = 24'h530000 + which * 16 + ctx;" in tb,
        "tsbg_context_tag_mapping":
            "golden_tag_t[ctx] = 24'h530000 + which * 16 + ctx;" in tb,
        "ordinary_golden_tag_check":
            "commit_tag_o !== golden_tag_o[commit_context_o]" in tb,
        "tsbg_golden_tag_check":
            "commit_tag_t !== golden_tag_t[commit_context_t]" in tb,
        "ordinary_context_check":
            "commit_context_o !== 3'(commit_seq_o / 6)" in tb,
        "tsbg_context_check":
            "commit_context_t !== 3'(commit_seq_t / 6)" in tb,
        "ordinary_slice_check":
            "commit_slice_o !== 3'(commit_seq_o % 6)" in tb,
        "tsbg_slice_check":
            "commit_slice_t !== 3'(commit_seq_t % 6)" in tb,
        "ordinary_terminal_check":
            "commit_terminal_o !== (commit_slice_o == 5)" in tb,
        "tsbg_terminal_check":
            "commit_terminal_t !== (commit_slice_t == 5)" in tb,
        "ordinary_acc24_check": "M2197 ordinary Acc24 mismatch" in tb,
        "tsbg_acc24_check": "M2197 TSBG Acc24 mismatch" in tb,
        "identity_ledger_token": "identity_o=%0d identity_t=%0d" in tb,
        "different_tags_loaded":
            "load_tag = 24'h530000 + bundle_count * 16 + ctx;" in tb,
        "sva_commit_tag_port":
            "input logic [TAG_BITS-1:0] commit_tag," in sva,
        "sva_commit_tag_connection":
            ".commit_context(commit_context), .commit_tag(commit_tag)," in tb,
        "sva_tag_stall_hold":
            "$stable({commit_context, commit_tag, commit_slice, commit_terminal})" in sva,
    }
    return [name for name, passed in checks.items() if not passed]


def load_parser():
    spec = importlib.util.spec_from_file_location("m2199_parser", PARSER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_parser_reject(module, sim: str, build: str = "clean\n",
                         rc: str = "0\n") -> None:
    with tempfile.TemporaryDirectory(prefix="m2197_parser_mut_") as td:
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
    assert sha(RTL) == "f651ea3a3b4dfab04d021a1e44797e7ab72c244cb7edf7496e18ac1ac033339e"
    assert sha(M2018) == "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21"
    assert sha(M803) == "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156"
    assert sha(OLD_SVA) == "98ffb5200398dfff158f314dbbcd8d494cc362f90a000ef4b9dc59323c669459"
    assert sha(OLD_TB) == "39e0c54d341fa360e23815aa94675d694606fe2c94c5f8a28efc40308bd15be7"
    assert sha(M2194) == "27aea19106ec6085bf00d9a2fa5d67ab78c71712b6a16591e1188ae4686c30bf"
    assert sha(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
    base = subprocess.run(["python3.12", str(BASE_TEST)], check=True,
                          text=True, capture_output=True)
    assert "PASS_M2193_SOURCE_TESTS" in base.stdout
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    compile(PARSER.read_text(), str(PARSER), "exec")
    for path in (SVA, TB):
        balanced_sv(path)

    tb = TB.read_text()
    sva = SVA.read_text()
    assert not validation_errors(tb, sva), validation_errors(tb, sva)
    mutations = [
        ("commit_tag_o !== golden_tag_o[commit_context_o]", "1'b0"),
        ("commit_tag_t !== golden_tag_t[commit_context_t]", "1'b0"),
        ("$stable({commit_context, commit_tag, commit_slice, commit_terminal})",
         "$stable({commit_context, commit_slice, commit_terminal})"),
        ("golden_tag_o[ctx] = 24'h530000 + which * 16 + ctx;",
         "golden_tag_o[ctx] = 24'h530000 + which * 16;"),
        ("golden_tag_t[ctx] = 24'h530000 + which * 16 + ctx;",
         "golden_tag_t[ctx] = 24'h530000 + which * 16;"),
        ("input logic [TAG_BITS-1:0] commit_tag,",
         "input logic [TAG_BITS-1:0] unverified_commit_tag,"),
        (".commit_context(commit_context), .commit_tag(commit_tag),",
         ".commit_context(commit_context),"),
        ("commit_context_o !== 3'(commit_seq_o / 6)", "1'b0"),
        ("commit_slice_t !== 3'(commit_seq_t % 6)", "1'b0"),
        ("commit_terminal_o !== (commit_slice_o == 5)", "1'b0"),
    ]
    for old, new in mutations:
        targets_sva = old in sva
        source = sva if targets_sva else tb
        assert old in source, old
        mut_tb, mut_sva = (tb, source.replace(old, new, 1)) if targets_sva \
            else (source.replace(old, new, 1), sva)
        assert validation_errors(mut_tb, mut_sva), (old, new)

    expected_filelist = [
        "hw_autoresearch_nts07/rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "hw_autoresearch_nts07/rtl_m2193/m2193_c2_tsbg_b4_selective_bank_fill_frontend.sv",
        "hw_autoresearch_nts07/verif_m2197/m2197_c2_tsbg_selective_bank_fill_assertions.sv",
        "hw_autoresearch_nts07/tb_m2197/tb_m2197_c2_tsbg_selective_bank_fill_directed.sv",
    ]
    assert FILELIST.read_text().splitlines() == expected_filelist

    parser = load_parser()
    cover = ("M2197_COVER partial_o=1 partial_t=1 eviction_o=2 eviction_t=2 "
             "reorder_o=3 reorder_t=3 reqstall_o=4 reqstall_t=4 bridgestall_o=5 "
             "bridgestall_t=5 commitstall_o=6 commitstall_t=6 zero_o=8 zero_t=8\n")
    passed = ("PASS_M2197_C2_TSBG_SELECTIVE_BANK_FILL_DIRECTED bundles=3 commits_o=72 "
              "commits_t=72 identity_o=72 identity_t=72 partial_o=1 partial_t=1 "
              "refills_o=120 refills_t=120 scalar_o=120 scalar_t=120 "
              "products_o=4096 products_t=4096\n")
    with tempfile.TemporaryDirectory(prefix="m2197_parser_control_") as td:
        root = Path(td)
        (root / "sim.log").write_text(cover + passed)
        (root / "compile.log").write_text("clean\n")
        (root / "rc").write_text("0\n")
        result = parser.parse(root / "sim.log", root / "compile.log", root / "rc")
        assert result["status"] == \
            "RAW_PASS_M2199_M2197_DIRECTED_VCS_PENDING_M2200_RESULT_HAMMER"
        assert result["ledger"]["identity_checks_ordinary"] == 72
    expect_parser_reject(parser, cover + passed.replace("identity_o=72", "identity_o=71"))
    expect_parser_reject(parser, cover + passed.replace("identity_t=72", "identity_t=71"))
    expect_parser_reject(parser, cover.replace("commitstall_o=6", "commitstall_o=0") + passed)
    expect_parser_reject(parser, cover + passed + "$fatal\n")
    expect_parser_reject(parser, cover + passed + passed)
    expect_parser_reject(parser, cover + passed, rc="1\n")

    runner = RUNNER.read_text()
    assert runner.count('"${VCS}" -full64') == 1
    assert runner.count('"${LMUTIL}" lmstat ') == 1
    assert "all_other_eda_runs':0" in runner
    assert "automatic_retry':False" in runner
    assert "PASS_M2198_M2197_SOURCE_HAMMER__M2199_ONE_SHOT_VCS_AUTHORIZED" in runner
    result = HW / "results/m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904"
    attempt = HW / "results/.m2199_m2197_selective_bank_fill_vcs_attempt_consumed"
    lock = HW / "results/.m2199_m2197_selective_bank_fill_vcs_launch_lock"
    assert not result.exists() and not attempt.exists() and not lock.exists()
    print("PASS_M2197_SOURCE_TESTS validation_mutations=10 parser_control=1 "
          "parser_mutations=6 sv_balance=2 inherited_m2193_static=PASS "
          "vcs_runs=0 eda_runs=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
