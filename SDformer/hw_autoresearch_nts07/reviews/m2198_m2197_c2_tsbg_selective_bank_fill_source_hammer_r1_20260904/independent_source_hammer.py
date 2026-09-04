#!/usr/bin/python3.12
"""Independent M2198 source hammer; no VCS, EDA, license, GPU, or Git."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent / "mechanical_checks.json"

RTL = HW / "rtl_m2193/m2193_c2_tsbg_b4_selective_bank_fill_frontend.sv"
SVA = HW / "verif_m2197/m2197_c2_tsbg_selective_bank_fill_assertions.sv"
TB = HW / "tb_m2197/tb_m2197_c2_tsbg_selective_bank_fill_directed.sv"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2197_c2_tsbg_selective_bank_fill_directed_vcs.f"
PARSER = HW / "system_simulator/scripts/parse_m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs.py"
RUNNER = HW / "dc_handoff/scripts/run_m2199_m2198_m2197_selective_bank_fill_directed_vcs_one_shot.sh"
TEST = HW / "tests/test_m2197_c2_tsbg_selective_bank_fill_validation_repair_source.py"
CONTRACT = HW / "contracts/m2197_c2_tsbg_selective_bank_fill_source_contract_r1_20260904.json"
OLD_SVA = HW / "verif_m2193/m2193_c2_tsbg_selective_bank_fill_assertions.sv"
OLD_TB = HW / "tb_m2193/tb_m2193_c2_tsbg_selective_bank_fill_directed.sv"
M2018 = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M2194 = HW / "reviews/m2194_m2193_c2_tsbg_selective_bank_fill_source_hammer_r1_20260904"
AUTHOR = HW / "reviews/m2197_m2194_c2_tsbg_commit_tag_validation_repair_source_author_receipt_r1_20260904"


EXPECTED = {
    RTL: "f651ea3a3b4dfab04d021a1e44797e7ab72c244cb7edf7496e18ac1ac033339e",
    SVA: "8003115edb919e9c5c6c9c36ce4ba75dfb37d9ec9f23e7c4cf59e2aed3b461b4",
    TB: "a8a954826324aa20443e7b2acbbc6a0b1b2a92f83ebdd84bfdbb0879920526e3",
    FILELIST: "5beddf477b6938b599cfab962eba60f6d79dceeb825380f2e5cdc6f22b49dc13",
    PARSER: "fde65c8372c9eab82ae49caea03137cdd93d0bd996fe65e9549220869a743571",
    RUNNER: "745da777421e5601776f1caf158f4905fdbe8c82f6c0095c118d7b2d98ceb3fb",
    TEST: "81d4cb93e7534e5ebb6cf68c02ded17db862479ab646deccc9ef9eb60e50dd5d",
    CONTRACT: "01aa9873330dddbc837929032bee18b89320a601a0ac491680d64339454577ed",
    OLD_SVA: "98ffb5200398dfff158f314dbbcd8d494cc362f90a000ef4b9dc59323c669459",
    OLD_TB: "39e0c54d341fa360e23815aa94675d694606fe2c94c5f8a28efc40308bd15be7",
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
    listed: list[str] = []
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
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


def load_test_module():
    spec = importlib.util.spec_from_file_location("m2197_source_test", TEST)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    for path, expected in EXPECTED.items():
        assert path.is_file() and not path.is_symlink()
        assert sha(path) == expected, path
    exhaustive_seal(M2194)
    exhaustive_seal(AUTHOR)
    fail = json.loads(M2194.joinpath("review.json").read_text())
    assert fail["status"] == (
        "FAIL_M2194_M2193_SOURCE_HAMMER__M2195_NOT_AUTHORIZED__COMMIT_TAG_CHECK_REQUIRED"
    )
    assert fail["severity_counts"] == {"p0": 0, "p1": 1, "p2": 0}

    official = subprocess.run(
        ["/opt/anaconda3/bin/python3.12", str(TEST)], cwd=ROOT,
        text=True, capture_output=True, check=True,
    )
    assert "validation_mutations=10 parser_control=1 parser_mutations=6" in official.stdout
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    compile(PARSER.read_text(), str(PARSER), "exec")

    tb = TB.read_text()
    sva = SVA.read_text()
    rtl = RTL.read_text()
    m803 = M803.read_text()

    # Directly close every part of the sole M2194 P1.
    repair_tokens = {
        "sva_commit_tag_port": "input logic [TAG_BITS-1:0] commit_tag," in sva,
        "sva_commit_tag_connected": ".commit_context(commit_context), .commit_tag(commit_tag)," in tb,
        "sva_header_holds_tag": "$stable({commit_context, commit_tag, commit_slice, commit_terminal})" in sva,
        "ordinary_four_golden_tags": "logic [23:0] golden_tag_o [0:3];" in tb,
        "tsbg_four_golden_tags": "logic [23:0] golden_tag_t [0:3];" in tb,
        "ordinary_context_formula": "golden_tag_o[ctx] = 24'h530000 + which * 16 + ctx;" in tb,
        "tsbg_context_formula": "golden_tag_t[ctx] = 24'h530000 + which * 16 + ctx;" in tb,
        "loaded_bundle_context_formula": "load_tag = 24'h530000 + bundle_count * 16 + ctx;" in tb,
        "ordinary_next_context": "commit_context_o !== 3'(commit_seq_o / 6)" in tb,
        "tsbg_next_context": "commit_context_t !== 3'(commit_seq_t / 6)" in tb,
        "ordinary_next_slice": "commit_slice_o !== 3'(commit_seq_o % 6)" in tb,
        "tsbg_next_slice": "commit_slice_t !== 3'(commit_seq_t % 6)" in tb,
        "ordinary_tag_compare": "commit_tag_o !== golden_tag_o[commit_context_o]" in tb,
        "tsbg_tag_compare": "commit_tag_t !== golden_tag_t[commit_context_t]" in tb,
        "ordinary_terminal_compare": "commit_terminal_o !== (commit_slice_o == 5)" in tb,
        "tsbg_terminal_compare": "commit_terminal_t !== (commit_slice_t == 5)" in tb,
        "ordinary_acc24_lanes": "M2197 ordinary Acc24 mismatch" in tb and "lane < LANES" in tb,
        "tsbg_acc24_lanes": "M2197 TSBG Acc24 mismatch" in tb and "lane < LANES" in tb,
        "per_bundle_24_each": "commit_seq_o != 24 || commit_seq_t != 24" in tb,
        "total_72_each": "commits_o != 72 || commits_t != 72" in tb,
        "cumulative_identity_ledgers": "identity_checks_o <= identity_checks_o + 1" in tb
            and "identity_checks_t <= identity_checks_t + 1" in tb,
    }
    assert all(repair_tokens.values()), [k for k, v in repair_tokens.items() if not v]

    # Independently recompute the tag space: distinct in context and bundle.
    tags = [[0x530000 + bundle * 16 + context for context in range(4)]
            for bundle in range(3)]
    assert len({value for row in tags for value in row}) == 12
    assert all(len(set(row)) == 4 for row in tags)
    assert all(max(tags[index]) < min(tags[index + 1]) for index in range(2))

    # Re-run each of the ten frozen validation mutations through the actual
    # validator and require an independently observed rejection.
    mod = load_test_module()
    assert mod.validation_errors(tb, sva) == []
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
    mutation_rejections = 0
    for old, new in mutations:
        if old in sva:
            mut_tb, mut_sva = tb, sva.replace(old, new, 1)
        else:
            assert old in tb, old
            mut_tb, mut_sva = tb.replace(old, new, 1), sva
        assert mod.validation_errors(mut_tb, mut_sva), old
        mutation_rejections += 1
    assert mutation_rejections == 10

    # Inherited selective-bank semantics and M803 protocol remain exact.
    inherited_tokens = (
        "find_needed_mask[0] |= active_row_q[ctx][find_group][7:0]",
        "find_needed_mask[1] |= active_row_q[ctx][find_group][15:8]",
        "find_missing_mask[0] = find_needed_mask[0]",
        "assign core_req_bank_valid = fill_missing_mask_q[fill_half_q]",
        "assign core_req_source_count = popcount8(core_req_bank_valid)",
        "core_rsp_bank_valid == fill_missing_mask_q[fill_half_q]",
        "cache_bank_valid_q[fill_cache_q][fill_half_q]",
        "logic signed [23:0] acc_q [0:BUNDLE-1]",
    )
    assert all(token in rtl for token in inherited_tokens)
    assert all(token in m803 for token in (
        "core_req_source_count == req_mask_count", "core_req_bank_valid != 0",
        "pending_mask_q[bank]", "incoming_mask[slot]",
        "bank_rsp_slot[bank] == slot[2:0]",
    ))
    assert tb.count("m2197_directed_side #(.SCHEDULE_MODE(") == 2
    assert "SCHEDULE_MODE(0)) ordinary" in tb and "SCHEDULE_MODE(1)) tsbg" in tb

    # Parser and one-shot runner must not allow the identity ledger to vanish.
    parser_text = PARSER.read_text()
    assert "values[3] == values[4] == 72" in parser_text
    assert '"identity_checks_ordinary": values[3]' in parser_text
    runner = RUNNER.read_text()
    assert runner.count('"${LMUTIL}" lmstat ') == 1
    assert runner.count('"${VCS}" -full64') == 1
    assert runner.count('300s "${WORK}/simv"') == 1
    assert "all_other_eda_runs':0" in runner and "automatic_retry':False" in runner
    assert "PASS_M2198_M2197_SOURCE_HAMMER__M2199_ONE_SHOT_VCS_AUTHORIZED" in runner

    census: list[str] = []
    for pattern in (
        "results/m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904*",
        "results/.m2199_m2197_selective_bank_fill_vcs_attempt_consumed*",
        "results/.m2199_m2197_selective_bank_fill_vcs_launch_lock*",
        "results/.m2199_m2197_selective_bank_fill_vcs_work*",
    ):
        census.extend(str(path.relative_to(HW)) for path in HW.glob(pattern))
    assert census == [], census

    result = {
        "schema": "m2198_m2197_c2_tsbg_selective_bank_fill_source_mechanical_checks_r1_v1",
        "status": "PASS_M2198_MECHANICAL_SOURCE_CHECKS",
        "identity_files_exact": len(EXPECTED),
        "m2194_and_m2197_author_double_seals_exhaustive": True,
        "official_source_suite": official.stdout.strip(),
        "m2194_p1_repair_checks": len(repair_tokens),
        "distinct_golden_tags": 12,
        "validation_mutations_independently_rejected": mutation_rejections,
        "parser_mutations_rejected": 6,
        "inherited_selective_bank_static_tokens": len(inherited_tokens),
        "m803_protocol_static_tokens": 5,
        "ordinary_tsbg_same_wrapper_and_ports": True,
        "m2199_result_attempt_work_lock_census": census,
        "docs359_sha256": sha(DOCS359),
        "execution": {"license": 0, "vcs": 0, "simv": 0, "eda": 0, "gpu": 0},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS_M2198_MECHANICAL p1_closed=1 mutations=10 m2199_census=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
