#!/usr/bin/env python3
"""Read-only independent result hammer for M1998; launches no EDA."""

from __future__ import print_function

import hashlib
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
RESULT = HW / "results/m1998_m1995_c2_tsbg_b4_keyword_legal_directed_vcs_r1_20260902"
ATTEMPT = HW / "results/.m1998_m1995_c2_tsbg_b4_keyword_legal_vcs_attempt_consumed"
M1990_RESULT = HW / "results/m1986_m1880_c2_tsbg_b4_bounded_directed_vcs_r1_20260902"
M1990_REVIEW = HW / "reviews/m1990_m1986_c2_tsbg_b4_parseable_vcs_result_hammer_r1_20260902"
M1995_REVIEW = HW / "reviews/m1995_m1992_tsbg_dc_keyword_failure_hammer_r1_20260902"
M1997_REVIEW = HW / "reviews/m1997_m1996_m1995_c2_tsbg_keyword_legal_vcs_source_hammer_r1_20260902"
RUNNER = HW / "dc_handoff/scripts/run_m1998_m1997_m1995_c2_tsbg_b4_keyword_legal_vcs_one_shot.sh"
FILELIST = HW / "dc_handoff/filelists/iscas_m1996_m1995_c2_tsbg_b4_keyword_legal_directed_vcs.f"
M1995 = HW / "rtl_m1995/m1995_m1880_c2_tsbg_b4_dc_keyword_legal_frontend.sv"
ADAPTER = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
SVA = HW / "verif_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend_assertions.sv"
TB = HW / "tb_m1984/tb_m1984_c2_tsbg_b4_parseable_pass.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_PASS = (
    "PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED "
    "rows=48 issues=576 products=9216 commits=24 bundles_base=576 "
    "bundles_tsbg=144 scalar_base=4608 scalar_tsbg=1152 stale=1 "
    "retired_replay=1 replay_accept=0 reset=2 recovery=1"
)
EXPECTED_FIELDS = {
    "rows": 48, "issues": 576, "products": 9216, "commits": 24,
    "bundles_base": 576, "bundles_tsbg": 144, "scalar_base": 4608,
    "scalar_tsbg": 1152, "stale": 1, "retired_replay": 1,
    "replay_accept": 0, "reset": 2, "recovery": 1,
}
FIELD_ORDER = list(EXPECTED_FIELDS)
PHASES = [
    "reset", "full_load", "full_execute", "retired_replay",
    "replay_reset_recovery", "stale_attack", "stale_reset_recovery",
    "recovery_load", "recovery_execute", "final_checks",
]
EXPECTED_TSBG_COVERS = {
    "cp_independent_bank_backpressure": 9,
    "cp_bank_response_reorder": 156,
    "cp_bridge_positive": 42,
    "cp_bridge_negative": 36,
    "cp_bridge_stall": 63,
    "cp_commit_stall": 4,
    "cp_terminal": 8,
    "cp_cache_eviction": 6022,
    "cp_weight_bundle": 6939,
    "cp_stale_attack": 4,
    "cp_reset_recovery_minimum_one_cycle": 6,
}
EXPECTED_BASE_COVERS = {
    "cp_independent_bank_backpressure": 9,
    "cp_bank_response_reorder": 588,
    "cp_bridge_positive": 42,
    "cp_bridge_negative": 36,
    "cp_bridge_stall": 62,
    "cp_commit_stall": 4,
    "cp_terminal": 8,
    "cp_cache_eviction": 6190,
    "cp_weight_bundle": 6939,
    "cp_stale_attack": 0,
    "cp_reset_recovery_minimum_one_cycle": 0,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_seal(directory):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert directory.is_dir() and not directory.is_symlink()
    expected_outer = outer.read_text().strip().split()
    assert len(expected_outer) == 2 and expected_outer[1] == "SHA256SUMS"
    assert expected_outer[0] == sha(manifest)
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.strip()
        assert digest == sha(directory / rel), (directory, rel)
    return True


def parse_pass_text(text):
    lines = [x for x in text.splitlines() if x.startswith("PASS_M1880_C2_TSBG_B4_REAL_M803_TYPED_SIGNED_DIRECTED")]
    assert len(lines) == 1
    assert lines[0] == EXPECTED_PASS
    toks = lines[0].split()
    assert len(toks) == 1 + len(EXPECTED_FIELDS)
    keys = []
    values = {}
    for token in toks[1:]:
        assert re.fullmatch(r"[a-z0-9_]+=[0-9]+", token)
        key, value = token.split("=", 1)
        assert key not in values
        keys.append(key)
        values[key] = int(value)
    assert keys == FIELD_ORDER
    assert values == EXPECTED_FIELDS
    return values


def extract_covers(text):
    found = {"tsbg": {}, "base": {}}
    pattern = re.compile(r"\.sva_(tsbg|base)\.(cp_[^,]+), [0-9]+ attempts, ([0-9]+) match$")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            axis, cover, count = match.groups()
            assert cover not in found[axis]
            found[axis][cover] = int(count)
    return found


def phase_lines(text):
    return [x for x in text.splitlines() if x.startswith("M1970_PHASE ")]


def load_lines(text):
    return [x for x in text.splitlines() if x.startswith("M1970_LOAD_")]


def mutation_hammer():
    mutations = {
        "wrong_rows": EXPECTED_PASS.replace("rows=48", "rows=47"),
        "truncated": EXPECTED_PASS.rsplit(" ", 1)[0],
        "extra_suffix": EXPECTED_PASS + " suffix=1",
        "duplicate": EXPECTED_PASS + "\n" + EXPECTED_PASS,
        "reordered": EXPECTED_PASS.replace("rows=48 issues=576", "issues=576 rows=48"),
        "missing_field": EXPECTED_PASS.replace(" stale=1", ""),
        "wrong_prefix": EXPECTED_PASS.replace("PASS_M1880", "PASS_M1995", 1),
        "wrong_stale": EXPECTED_PASS.replace("stale=1", "stale=0"),
        "duplicate_field": EXPECTED_PASS + " rows=48",
        "nonnumeric": EXPECTED_PASS.replace("recovery=1", "recovery=yes"),
    }
    rejected = {}
    for name, mutant in mutations.items():
        try:
            parse_pass_text(mutant)
            rejected[name] = False
        except (AssertionError, ValueError):
            rejected[name] = True
    assert all(rejected.values())
    return rejected


def main():
    seals = {
        "result": verify_seal(RESULT),
        "attempt": verify_seal(ATTEMPT),
        "m1990_review": verify_seal(M1990_REVIEW),
        "m1995_review": verify_seal(M1995_REVIEW),
        "m1997_review": verify_seal(M1997_REVIEW),
    }
    identities = {
        "result_manifest_sha256": sha(RESULT / "SHA256SUMS"),
        "result_outer_seal_sha256": sha(RESULT / "SHA256SUMS.seal.sha256"),
        "attempt_manifest_sha256": sha(ATTEMPT / "SHA256SUMS"),
        "attempt_outer_seal_sha256": sha(ATTEMPT / "SHA256SUMS.seal.sha256"),
        "receipt_sha256": sha(RESULT / "receipt.txt"),
        "compile_log_sha256": sha(RESULT / "vcs_compile.log"),
        "simv_log_sha256": sha(RESULT / "simv.log"),
        "runner_sha256": sha(RUNNER),
        "filelist_sha256": sha(FILELIST),
        "m1995_rtl_sha256": sha(M1995),
        "adapter_sha256": sha(ADAPTER),
        "sva_sha256": sha(SVA),
        "testbench_sha256": sha(TB),
        "m1990_review_sha256": sha(M1990_REVIEW / "review.json"),
        "m1995_review_sha256": sha(M1995_REVIEW / "review.json"),
        "m1997_review_sha256": sha(M1997_REVIEW / "review.json"),
        "docs359_sha256": sha(DOCS359),
    }
    expected_identity = {
        "runner_sha256": "de872c1b7f323b483a008108fc48d793c8176667eb19f4edfc543e3035b22e96",
        "filelist_sha256": "a89c09074abbde86fc3f5b2a748418bef601f5bf9ec6f53736992e155e29414c",
        "m1995_rtl_sha256": "2c1a8a7644b359a153decdc3106a8718992d37d54809007b61e184121fcc14fd",
        "adapter_sha256": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
        "sva_sha256": "e5519a75c14d68dfc273c3a7e9930560fa8a3c7779ab5ed7f22f294a14be58c2",
        "testbench_sha256": "d46a47dada89e16cdc3f2593020a89e3513060a8a1a03ae3a1963d0483b96081",
        "m1990_review_sha256": "e2935ed23f2e2b24798ea6b6ab1f098fcd356e1969e31279793a063c9b07b80c",
        "m1995_review_sha256": "37adc83f6b6f70457d06e8ba215dba64d345fd03e2c2b8f3ea5ed363f11a5c01",
        "m1997_review_sha256": "b2545bd3b3c0d819e6c8bf8a506286f5f725dde204eb251e6b84b3e6307909f5",
        "docs359_sha256": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    }
    for key, expected in expected_identity.items():
        assert identities[key] == expected, (key, identities[key])

    source_review = json.loads((M1997_REVIEW / "review.json").read_text())
    assert source_review["status"] == "PASS_M1997_C2_TSBG_KEYWORD_LEGAL_VCS_SOURCE_HAMMER"
    assert source_review["identity"]["m1995_rtl_sha256"] == identities["m1995_rtl_sha256"]
    assert source_review["identity"]["runner_sha256"] == identities["runner_sha256"]
    assert source_review["identity"]["filelist_sha256"] == identities["filelist_sha256"]
    assert source_review["identity"]["m1995_failure_review_sha256"] == identities["m1995_review_sha256"]

    receipt = RESULT.joinpath("receipt.txt").read_text()
    expected_receipt = (
        "status=RAW_PASS_M1998_KEYWORD_LEGAL_VCS_PENDING_INDEPENDENT_RESULT_REVIEW\n"
        "license_queries=1\nvcs_compiles=1\nsimv_runs=1\nretry=false\n"
        "identifier_rename_only=true\nbehavioral_rtl_directed_only=true\n"
        "same_area=false\nexact_cycle_speedup=false\nsystem_speedup=false\n"
    )
    assert receipt == expected_receipt
    assert ATTEMPT.joinpath("ATTEMPT_CONSUMED.txt").read_text() == (
        "status=M1998_ATTEMPT_CONSUMED\nlicense_queries=1\n"
        "vcs_compiles=1\nsimv_runs=1\nretry=false\n"
    )
    names = sorted(x.name for x in (HW / "results").iterdir()
                   if "m1998" in x.name or "m1995_c2_tsbg_b4_keyword_legal" in x.name)
    assert names == [
        ".m1998_m1995_c2_tsbg_b4_keyword_legal_vcs_attempt_consumed",
        "m1998_m1995_c2_tsbg_b4_keyword_legal_directed_vcs_r1_20260902",
    ], names

    runner = RUNNER.read_text()
    assert runner.count('"${LMUTIL}" lmstat') == 1
    assert runner.count('"${VCS}" -full64') == 1
    assert runner.count('180s "${WORK}/simv"') == 1
    assert "retry=false" in runner
    assert "-assert svaext" in runner
    assert "-assert global_finish_maxfail=1" in runner

    compile_log = RESULT.joinpath("vcs_compile.log").read_text(errors="replace")
    simv_log = RESULT.joinpath("simv.log").read_text(errors="replace")
    for compiled in [str(ADAPTER.relative_to(REPO)), str(M1995.relative_to(REPO)),
                     str(SVA.relative_to(REPO)), str(TB.relative_to(REPO))]:
        assert compile_log.count("Parsing design file '{}'".format(compiled)) == 1, compiled
    assert "Top Level Modules:\n       tb_m1880_c2_tsbg_b4_real_channel_signed_frontend" in compile_log
    forbidden_compile = [r"Error-", r"Warning-\[SVAA-RNF\]", r"Ignoring.*global_finish_maxfail",
                         r"global_finish_maxfail.*(?:ignored|unknown)"]
    forbidden_sim = [r"Warning-\[SVAA-RNF\]", r"Ignoring.*global_finish_maxfail",
                     r"global_finish_maxfail.*(?:ignored|unknown)", r": started at .* failed at",
                     r"Assertion[^\n]*failed", r"Error-\[SVA", r"\$(?:error|fatal)",
                     r"Fatal:", r"whole-test watchdog expired", r"directed timeout",
                     r"post-reset legal-service timeout", r"M1970_LOAD_TIMEOUT"]
    assert all(re.search(p, compile_log, re.I) is None for p in forbidden_compile)
    assert all(re.search(p, simv_log, re.I) is None for p in forbidden_sim)

    parsed_fields = parse_pass_text(simv_log)
    assert simv_log.count("$finish called") == 1
    phase_counts = {}
    for phase in PHASES:
        begin = simv_log.count("M1970_PHASE {}_begin".format(phase))
        complete = simv_log.count("M1970_PHASE {}_complete".format(phase))
        assert begin == complete == 1
        phase_counts[phase] = {"begin": begin, "complete": complete}
    assert simv_log.count("M1970_LOAD_BEGIN") == 52
    assert simv_log.count("M1970_LOAD_COMPLETE") == 52
    assert simv_log.count("M1970_LOAD_TIMEOUT") == 0
    covers = extract_covers(simv_log)
    assert covers["tsbg"] == EXPECTED_TSBG_COVERS
    assert covers["base"] == EXPECTED_BASE_COVERS

    old_log = M1990_RESULT.joinpath("simv.log").read_text(errors="replace")
    old_covers = extract_covers(old_log)
    behavior_equivalence = {
        "exact_pass_line_equal": [x for x in old_log.splitlines() if x.startswith("PASS_")] ==
                                 [x for x in simv_log.splitlines() if x.startswith("PASS_")],
        "phase_ledger_equal": phase_lines(old_log) == phase_lines(simv_log),
        "load_ledger_equal": load_lines(old_log) == load_lines(simv_log),
        "cover_ledger_equal": old_covers == covers,
    }
    assert all(behavior_equivalence.values())
    mutations = mutation_hammer()

    mechanical = {
        "status": "PASS_M1999_M1998_M1995_KEYWORD_LEGAL_VCS_RESULT_MECHANICAL_HAMMER",
        "seals": seals,
        "identity": identities,
        "namespace_entries": names,
        "execution_census": {"license_queries": 1, "vcs_compiles": 1,
                             "simv_runs": 1, "automatic_retry": False},
        "parsed_pass_fields": parsed_fields,
        "phase_counts": phase_counts,
        "load_counts": {"begin": 52, "complete": 52, "timeout": 0},
        "sva_covers": covers,
        "m1990_behavior_ledger_equivalence": behavior_equivalence,
        "mutation_rejections": mutations,
        "mutation_count": len(mutations),
        "forbidden_log_patterns_absent": True,
        "eda_launched": False,
    }
    OUT.joinpath("mechanical_checks.json").write_text(
        json.dumps(mechanical, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": mechanical["status"], "mutation_count": len(mutations)}, sort_keys=True))


if __name__ == "__main__":
    main()
