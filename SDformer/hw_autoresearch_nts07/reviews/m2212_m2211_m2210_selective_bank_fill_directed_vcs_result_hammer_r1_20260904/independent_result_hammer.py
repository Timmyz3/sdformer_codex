#!/usr/bin/python3.12
"""Independent CPU-only M2212 result hammer; runs no VCS/EDA/license/GPU/Git."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
HERE = Path(__file__).resolve().parent
RESULTS = HW / "results"
ATTEMPT = RESULTS / ".m2211_m2209_selective_bank_fill_vcs_attempt_consumed"
RESULT = RESULTS / "m2211_m2209_c2_tsbg_selective_bank_fill_directed_vcs_r1_20260904"
M2210 = HW / "reviews/m2210_m2209_m2200_selective_bank_fill_vcs_runner_repair_source_hammer_r1_20260904"
M2209_AUTHOR = HW / "reviews/m2209_m2200_selective_bank_fill_vcs_runner_repair_source_author_receipt_r1_20260904"
M2198 = HW / "reviews/m2198_m2197_c2_tsbg_selective_bank_fill_source_hammer_r1_20260904"
M2197_AUTHOR = HW / "reviews/m2197_m2194_c2_tsbg_commit_tag_validation_repair_source_author_receipt_r1_20260904"
CONTRACT = HW / "contracts/m2209_m2200_selective_bank_fill_vcs_runner_repair_source_contract_r1_20260904.json"
M2197_CONTRACT = HW / "contracts/m2197_c2_tsbg_selective_bank_fill_source_contract_r1_20260904.json"
RUNNER = HW / "dc_handoff/scripts/run_m2211_m2210_m2209_selective_bank_fill_directed_vcs_one_shot.sh"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2197_c2_tsbg_selective_bank_fill_directed_vcs.f"
RTL = HW / "rtl_m2193/m2193_c2_tsbg_b4_selective_bank_fill_frontend.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
M2018 = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
SVA = HW / "verif_m2197/m2197_c2_tsbg_selective_bank_fill_assertions.sv"
TB = HW / "tb_m2197/tb_m2197_c2_tsbg_selective_bank_fill_directed.sv"
PARSER = HW / "system_simulator/scripts/parse_m2199_m2197_c2_tsbg_selective_bank_fill_directed_vcs.py"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    RUNNER: "19958f56f230ee1ced6c42c54201b79379db803d17f1629b7ecbaa549a477a45",
    FILELIST: "5beddf477b6938b599cfab962eba60f6d79dceeb825380f2e5cdc6f22b49dc13",
    RTL: "f651ea3a3b4dfab04d021a1e44797e7ab72c244cb7edf7496e18ac1ac033339e",
    M803: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    M2018: "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    SVA: "8003115edb919e9c5c6c9c36ce4ba75dfb37d9ec9f23e7c4cf59e2aed3b461b4",
    TB: "a8a954826324aa20443e7b2acbbc6a0b1b2a92f83ebdd84bfdbb0879920526e3",
    PARSER: "fde65c8372c9eab82ae49caea03137cdd93d0bd996fe65e9549220869a743571",
    CONTRACT: "4f44a95b2e22d31afc520a0b62d194a7fbbd175101caa816162773cb6a1247bb",
    M2197_CONTRACT: "01aa9873330dddbc837929032bee18b89320a601a0ac491680d64339454577ed",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    need(isinstance(value, dict), f"JSON object required {path}")
    return value


def verify_seal(directory: Path) -> dict[str, object]:
    need(directory.is_dir() and not directory.is_symlink(), f"invalid dir {directory}")
    need(not any(path.is_symlink() for path in directory.rglob("*")), "symlink in seal")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"], "outer seal")
    listed = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.strip().lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe member")
        need(sha(directory / rel) == digest, f"member drift {rel}")
        listed.add(rel.as_posix())
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(actual == listed, "non-exhaustive seal")
    return {"manifest_sha256": sha(manifest), "members": len(listed),
            "exhaustive": True, "symbolic_links": 0}


def main() -> int:
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             f"source identity drift {path}")
    seals = {"attempt": verify_seal(ATTEMPT), "result": verify_seal(RESULT),
             "m2210": verify_seal(M2210), "m2209_author": verify_seal(M2209_AUTHOR),
             "m2198": verify_seal(M2198), "m2197_author": verify_seal(M2197_AUTHOR)}
    need(sha(M2210 / "review.json") ==
         "a4d6917d6a7453390ed7b9a8742adba95a70b3949479ce74f6e5bff1a311e78c",
         "M2210 identity")
    auth = read_json(M2210 / "review.json")
    need(auth.get("status") == "PASS_M2210_M2209_SOURCE_HAMMER__M2211_ONE_SHOT_VCS_AUTHORIZED" and
         auth.get("authorization") == {"m2211": True, "license_queries": 1,
         "vcs_compiles": 1, "simv_runs": 1, "parser_runs": 1,
         "all_other_eda_runs": 0, "automatic_retry": False, "reuse_old_artifacts": False},
         "M2210 authorization")
    matching = sorted(path.name for path in RESULTS.iterdir() if "m2211" in path.name)
    need(matching == sorted([ATTEMPT.name, RESULT.name]), f"M2211 multiplicity {matching}")
    need(ATTEMPT.joinpath("ATTEMPT_CONSUMED.txt").read_text() ==
         "status=M2211_ATTEMPT_CONSUMED\nlicense_queries=1\nvcs_compiles=1\nsimv_runs=1\nparser_runs=1\nretry=false\nreuse_old_artifacts=false\n",
         "attempt marker")
    need(RESULT.joinpath("RUN_COMPLETE.txt").read_text() ==
         "RAW_PASS_M2211_M2209_DIRECTED_VCS_PENDING_M2212_RESULT_HAMMER\n", "run complete")
    need(RESULT.joinpath("simv.rc").read_text() == "0\n", "sim return code")
    need(RESULT.joinpath("parser.log").read_text() ==
         "RAW_PASS_M2199_M2197_DIRECTED_VCS_PENDING_M2200_RESULT_HAMMER\n", "parser token")
    need(seals["result"]["members"] == 7, "result member count")
    forbidden = {"simv", "vc_hdrs.h", "csrc", "simv.daidir", "simv.vdb"}
    need(not any(path.name in forbidden for path in RESULT.rglob("*")), "build object retained")

    runner = RUNNER.read_text()
    need(runner.count('"${VCS}" -full64 -sverilog') == 1 and
         runner.count('300s "${WORK}/simv"') == 1 and
         runner.count('"${PYTHON}" -B "${PARSER}"') == 1, "runner invocation count")
    build = RESULT.joinpath("vcs_compile.log").read_text(errors="replace")
    sim = RESULT.joinpath("simv.log").read_text(errors="replace")
    fatal_terms = ("Error-", "Syntax error", "Compiler directive error", "$fatal",
                   "Assertion failed", "Offending", "UVM_FATAL")
    need(not any(term in build for term in fatal_terms) and
         not any(term in sim for term in fatal_terms), "compile/sim fatal diagnostic")
    need(build.count("Starting vcs inline pass...") == 1 and
         len(re.findall(r"^CPU time: .* to compile", build, re.MULTILINE)) == 1,
         "compile occurrence")
    cover_re = re.compile(
        r"^M2197_COVER partial_o=(\d+) partial_t=(\d+) eviction_o=(\d+) "
        r"eviction_t=(\d+) reorder_o=(\d+) reorder_t=(\d+) reqstall_o=(\d+) "
        r"reqstall_t=(\d+) bridgestall_o=(\d+) bridgestall_t=(\d+) "
        r"commitstall_o=(\d+) commitstall_t=(\d+) zero_o=(\d+) zero_t=(\d+)$", re.MULTILINE)
    pass_re = re.compile(
        r"^PASS_M2197_C2_TSBG_SELECTIVE_BANK_FILL_DIRECTED bundles=(\d+) "
        r"commits_o=(\d+) commits_t=(\d+) identity_o=(\d+) identity_t=(\d+) "
        r"partial_o=(\d+) partial_t=(\d+) refills_o=(\d+) refills_t=(\d+) "
        r"scalar_o=(\d+) scalar_t=(\d+) products_o=(\d+) products_t=(\d+)$", re.MULTILINE)
    covers = cover_re.findall(sim)
    passes = pass_re.findall(sim)
    need(len(covers) == len(passes) == 1, "unique cover/pass")
    cover = list(map(int, covers[0]))
    values = list(map(int, passes[0]))
    need(all(value > 0 for value in cover), "zero coverage counter")
    need(values == [3, 72, 72, 72, 72, 2, 2, 588, 156, 588, 156, 3264, 3264],
         "directed ledger values")
    receipt = read_json(RESULT / "receipt.json")
    need(receipt.get("status") == "RAW_PASS_M2199_M2197_DIRECTED_VCS_PENDING_M2200_RESULT_HAMMER",
         "immutable parser receipt status")
    need(receipt.get("ledger") == {"bundles": 3, "commits_ordinary": 72,
         "commits_tsbg": 72, "identity_checks_ordinary": 72,
         "identity_checks_tsbg": 72, "products_each": 3264,
         "refill_banks_ordinary": 588, "refill_banks_tsbg": 156,
         "scalar_requests_ordinary": 588, "scalar_requests_tsbg": 156}, "receipt ledger")
    need(receipt.get("claim_boundary") == {"directed_vcs_only": True,
         "rtl_performance": False, "same_area": False, "timing": False,
         "energy": False, "power": False, "paper_result": False,
         "component_speedup_admitted": False, "system_speedup": False,
         "headline": False}, "claim boundary")
    tb = TB.read_text()
    for token in ("ordinary context mismatch", "ordinary slice mismatch",
                  "ordinary golden-tag mismatch", "ordinary terminal mismatch",
                  "ordinary Acc24 mismatch", "TSBG context mismatch", "TSBG slice mismatch",
                  "TSBG golden-tag mismatch", "TSBG terminal mismatch", "TSBG Acc24 mismatch"):
        need(token in tb, f"missing scoreboard check {token}")
    need("protocol_error" in tb and "stale_response_seen" in tb and
         "$fatal(1, \"M2197 protocol/numeric failure\")" in tb, "protocol checks")

    read_ratio = values[7] / values[8]
    read_reduction = 1.0 - values[8] / values[7]
    result = {
        "schema": "m2212_m2211_m2210_selective_bank_fill_directed_vcs_result_mechanical_checks_r1_v1",
        "status": "PASS_M2212_M2211_RESULT_HAMMER__SELECTIVE_BANK_DIRECTED_RTL_ONLY",
        "identity": {"attempt_manifest_sha256": sha(ATTEMPT / "SHA256SUMS"),
                     "result_manifest_sha256": sha(RESULT / "SHA256SUMS"),
                     "sim_log_sha256": sha(RESULT / "simv.log"),
                     "compile_log_sha256": sha(RESULT / "vcs_compile.log"),
                     "receipt_sha256": sha(RESULT / "receipt.json"),
                     "m2210_review_sha256": sha(M2210 / "review.json"),
                     "docs359_sha256": sha(DOC359)},
        "seals": seals,
        "execution_census": {"license_queries": 1, "vcs_compile_invocations": 1,
                             "top_level_simv_invocations": 1, "parser_invocations": 1,
                             "automatic_retry": False, "old_artifact_reuse": False,
                             "aslr_internal_simv_reexec_disclosed": True},
        "functional": {"bundles": 3, "ordinary_commits": 72, "tsbg_commits": 72,
                       "ordinary_identity_checks": 72, "tsbg_identity_checks": 72,
                       "exact_acc24_values": True, "exact_context_slice_tag_terminal": True,
                       "positive_and_negative_sources": True,
                       "assertion_or_fatal_failures": 0},
        "coverage": {"partial_hit": {"ordinary": cover[0], "tsbg": cover[1]},
                     "eviction": {"ordinary": cover[2], "tsbg": cover[3]},
                     "response_reorder": {"ordinary": cover[4], "tsbg": cover[5]},
                     "request_backpressure": {"ordinary": cover[6], "tsbg": cover[7]},
                     "bridge_backpressure": {"ordinary": cover[8], "tsbg": cover[9]},
                     "commit_backpressure": {"ordinary": cover[10], "tsbg": cover[11]},
                     "zero_descriptor_skip": {"ordinary": cover[12], "tsbg": cover[13]},
                     "legal_protocol_stress_covered": True,
                     "explicit_illegal_or_stale_protocol_attack_injected": False},
        "ledger": {"ordinary_refill_reads": values[7], "ordinary_scalar_requests": values[9],
                   "tsbg_refill_reads": values[8], "tsbg_scalar_requests": values[10],
                   "read_count_conserved_each_mode": values[7] == values[9] and values[8] == values[10],
                   "directed_read_ratio_ordinary_over_tsbg": read_ratio,
                   "directed_read_reduction_tsbg_vs_ordinary": read_reduction,
                   "ordinary_products": values[11], "tsbg_products": values[12],
                   "product_count_conserved": values[11] == values[12],
                   "per_mode_cycles_emitted": False,
                   "cycle_conservation_or_speedup_verifiable": False,
                   "concurrent_simulation_finish_time_ps": 6766500},
        "admission": {"selective_bank_directed_rtl_function": True,
                      "selective_refill_protocol": True, "commit_identity": True,
                      "cpu_premodel_speedup": False, "rtl_performance": False,
                      "same_area": False, "timing": False, "hold": False,
                      "power": False, "energy": False, "system_speedup": False,
                      "paper_ppa_ready": False},
        "open_evidence": {"severity_counts": {"p0": 0, "p1": 0, "p2": 2},
                          "p2": ["per-mode RTL cycle counters were not emitted, so cycle conservation/speedup is unverified",
                                 "coverage stresses legal reorder/backpressure/partial-hit paths but injects no explicit illegal or stale protocol attack"]},
        "review_execution": {"vcs_runs": 0, "simv_runs": 0, "parser_runs": 0,
                             "eda_runs": 0, "license_queries": 0, "gpu_runs": 0,
                             "git_mutation": False, "source_modified": False,
                             "m2211_modified": False, "docs359_modified": False},
    }
    output = HERE / "mechanical_checks.json"
    need(not output.exists(), "fresh output required")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
