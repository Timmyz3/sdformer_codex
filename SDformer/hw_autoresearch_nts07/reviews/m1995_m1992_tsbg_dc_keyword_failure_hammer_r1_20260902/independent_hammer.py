#!/usr/bin/env python3
"""Read-only failure hammer for M1992 and exact additive M1995 repair model."""

from __future__ import print_function

import hashlib
import json
import re
import sys
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
RUNS = HW / "dc_handoff/runs"
Q = RUNS / "m1992_m1990_c2_tsbg_b4_matched_two_axis_logic_only_dc_r1_20260902.failed_or_incomplete.338853.quarantine"
A = RUNS / ".m1992_m1990_c2_tsbg_b4_matched_dc_attempt_consumed"
RTL = HW / "rtl_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "rtl": "8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "q_manifest": "d8f4f4dd5ee2ab2f4cd0e4a52de9c156454c8b8cb3777a6834bc345a6fca3041",
    "q_outer": "c181587f642a55310bb97a3e427bf6c9d31986e77433b7b61d4e8af2f591b0a1",
    "attempt_manifest": "07e1ee99173abb73efbf87e856fb4a096b03174ebf2b0e498a34d20fa49849b7",
    "attempt_outer": "81883ef662799aa1e85f2ce28f1be7e79886fb5a40caef259aa1a95a25e5727e",
    "dc_log": "49dade262637d758e8e74fe8e7770b21b5d5a811b96c789bb27fa9818f6d30f5",
    "m1995_model": "2c1a8a7644b359a153decdc3106a8718992d37d54809007b61e184121fcc14fd",
}

TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_$])context(?![A-Za-z0-9_$])")
EXPECTED_POSITIONS = [
    (207, 27), (214, 40),
    (469, 22), (469, 35), (469, 53), (470, 31),
    (473, 34), (474, 32), (480, 31),
    (628, 26), (628, 39), (628, 57), (629, 35),
    (633, 38), (634, 36), (640, 35),
]


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def need(value, message):
    if not value:
        raise AssertionError(message)


def verify_seal(directory):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and outer.is_file(), "missing seal: " + str(directory))
    for line in manifest.read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip(" *")
        need(sha(directory / name) == digest, "manifest mismatch: " + name)
    outer_digest, outer_name = outer.read_text().strip().split()
    need(outer_name.lstrip("*") == "SHA256SUMS", "bad outer seal target")
    need(sha(manifest) == outer_digest, "outer seal mismatch")


def positions(text):
    result = []
    for match in TOKEN_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        column = match.start() - text.rfind("\n", 0, match.start())
        result.append((line, column))
    return result


def valid_additive_successor(old, candidate):
    """Accept only the exact 16-token context->ctx alpha-renaming."""
    need(positions(old) == EXPECTED_POSITIONS, "old token map differs")
    model = TOKEN_RE.sub("ctx", old)
    need(candidate == model, "candidate has a non-alpha-renaming byte delta")
    need(not TOKEN_RE.search(candidate), "reserved standalone context remains")
    need(sha_bytes(candidate.encode()) == EXPECTED["m1995_model"],
         "candidate model SHA differs")


def main():
    need(sha(RTL) == EXPECTED["rtl"], "M1880 was modified")
    need(sha(DOCS359) == EXPECTED["docs359"], "docs/359 was modified")
    need(sha(Q / "SHA256SUMS") == EXPECTED["q_manifest"],
         "quarantine manifest identity mismatch")
    need(sha(Q / "SHA256SUMS.seal.sha256") == EXPECTED["q_outer"],
         "quarantine outer seal identity mismatch")
    need(sha(A / "SHA256SUMS") == EXPECTED["attempt_manifest"],
         "attempt manifest identity mismatch")
    need(sha(A / "SHA256SUMS.seal.sha256") == EXPECTED["attempt_outer"],
         "attempt outer seal identity mismatch")
    verify_seal(Q)
    verify_seal(A)

    namespaces = sorted(p.name for p in RUNS.iterdir()
                        if "m1992_m1990_c2_tsbg_b4_matched" in p.name)
    need(namespaces == [
        ".m1992_m1990_c2_tsbg_b4_matched_dc_attempt_consumed",
        "m1992_m1990_c2_tsbg_b4_matched_two_axis_logic_only_dc_r1_20260902.failed_or_incomplete.338853.quarantine",
    ], "M1992 namespace is not a unique consumed failure")
    need((A / "ATTEMPT_CONSUMED.txt").read_text() ==
         "status=M1992_ATTEMPT_CONSUMED\n"
         "dc_shell_runs=2\naxes=ordinary_lru4,tsbg_b4\n"
         "retry=false\n", "attempt marker differs")
    need((Q / "RUN_FAILED_OR_INCOMPLETE.txt").read_text() ==
         "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\n"
         "exit_code=35\nretry=false\n", "failure marker differs")
    need((Q / "ordinary_lru4").is_dir() and
         not (Q / "tsbg_b4").exists(),
         "axis execution census differs")

    log_path = Q / "ordinary_lru4/dc.log"
    need(sha(log_path) == EXPECTED["dc_log"], "DC log identity mismatch")
    log = log_path.read_text(errors="replace")
    primary = re.findall(
        r"^Error:\s+.*m1880_c2_tsbg_b4_real_channel_signed_frontend\.sv:(\d+): "
        r"Syntax error at or near token 'context'\. \(VER-294\)$",
        log, flags=re.MULTILINE)
    need(primary == ["207", "214", "469", "473", "474", "480"],
         "observed VER-294 line sequence differs")
    need(log.index("sv:207: Syntax error") < log.index("(VER-40)"),
         "VER-294 is not before error-limit cascade")
    need(log.index("(VER-40)") < log.index("(LBR-0)") < log.index("(UID-4)"),
         "cascade ordering differs")
    need("Presto compilation terminated with 7 errors" in log,
         "Presto error-limit receipt missing")
    need("status=FAIL_ELABORATION_NO_CURRENT_DESIGN" in
         (Q / "ordinary_lru4/TCL_EXPLICIT_FAILURE.txt").read_text(),
         "Tcl fail-closed marker missing")
    for forbidden in ["out of memory", "cannot allocate", "checkout failed",
                      "timed out", "segmentation fault", "core dump"]:
        need(forbidden not in log.lower(), "resource failure found: " + forbidden)
    need("Elapsed time for this session 11 seconds" in log and
         "Memory usage for this session including child processes 677 Mbytes" in log,
         "bounded runtime/resource receipt differs")

    source = RTL.read_text()
    need(positions(source) == EXPECTED_POSITIONS,
         "complete standalone context token map differs")
    candidate = TOKEN_RE.sub("ctx", source)
    valid_additive_successor(source, candidate)
    need(len(source) - len(candidate) == 64,
         "expected exact 16*(7-3)=64 byte reduction")
    need("module m1880_c2_tsbg_b4_real_channel_signed_frontend #(\n" in candidate,
         "module/top changed")

    mutations = [
        ("miss_one_rename", source.replace("context", "ctx", 1)),
        ("rename_load_context_substring", candidate.replace("load_context", "load_ctx", 1)),
        ("rename_module", candidate.replace(
            "module m1880_c2_tsbg_b4_real_channel_signed_frontend",
            "module m1995_c2_tsbg_b4_real_channel_signed_frontend", 1)),
        ("change_schedule_default", candidate.replace(
            "parameter int SCHEDULE_MODE = 1",
            "parameter int SCHEDULE_MODE = 0", 1)),
        ("change_comment", candidate.replace("M1880 is the additive successor",
                                              "M1995 is the additive successor", 1)),
        ("rename_ctx_again", candidate.replace("input logic [2:0] ctx",
                                                "input logic [2:0] c", 1)),
    ]
    mutation_results = []
    for name, mutated in mutations:
        rejected = False
        reason = ""
        try:
            valid_additive_successor(source, mutated)
        except AssertionError as exc:
            rejected = True
            reason = str(exc)
        need(rejected, "repair mutation escaped: " + name)
        mutation_results.append({"name": name, "rejected": True,
                                 "reason": reason})

    evidence = {
        "schema": "m1995_m1992_keyword_failure_independent_hammer_r1_v1",
        "status": "PASS_M1995_M1992_KEYWORD_FAILURE_AND_REPAIR_MODEL_HAMMER",
        "identity": {
            "m1880_rtl_sha256": sha(RTL),
            "docs359_sha256": sha(DOCS359),
            "quarantine_manifest_sha256": sha(Q / "SHA256SUMS"),
            "quarantine_outer_seal_sha256": sha(Q / "SHA256SUMS.seal.sha256"),
            "attempt_manifest_sha256": sha(A / "SHA256SUMS"),
            "attempt_outer_seal_sha256": sha(A / "SHA256SUMS.seal.sha256"),
            "dc_log_sha256": sha(log_path),
            "modeled_m1995_source_sha256": sha_bytes(candidate.encode()),
        },
        "execution_census": {
            "attempt_consumed": True,
            "automatic_retry": False,
            "ordinary_axis_started": True,
            "tsbg_axis_started": False,
            "published_result": False,
            "quarantined_failure": True,
        },
        "failure": {
            "observed_unique_first_cause_class": "Presto VER-294 reserved standalone SystemVerilog keyword context",
            "first_source_line": 207,
            "presto_reported_source_lines_before_error_limit": [207, 214, 469, 473, 474, 480],
            "all_source_token_positions_line_column": EXPECTED_POSITIONS,
            "all_source_token_count": 16,
            "ver40_is_error_limit_cascade": True,
            "lbr0_uid4_are_no_design_cascade": True,
            "adapter_ver104_is_noncausal_warning": True,
            "functional_failure_observed": False,
            "structural_synthesis_failure_observed": False,
            "resource_or_license_failure_observed": False,
            "latent_post_parse_errors_excluded": False,
        },
        "repair_model": {
            "new_additive_source_required": True,
            "old_m1880_must_remain_unchanged": True,
            "only_permitted_delta": "16 standalone identifier tokens context -> ctx",
            "model_sha256": EXPECTED["m1995_model"],
            "module_name_must_remain_m1880": True,
            "non_context_substrings_must_remain_unchanged": True,
            "byte_length_reduction": 64,
        },
        "mutations": mutation_results,
        "mutation_count": len(mutation_results),
        "mutations_rejected": len(mutation_results),
        "eda_launched": False,
        "license_query_launched": False,
    }
    Path(__file__).with_name("mechanical_checks.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(evidence["status"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAIL_M1995: %s" % exc, file=sys.stderr)
        raise
