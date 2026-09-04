#!/usr/bin/env python3
"""Read-only M2152 hammer for the consumed M2151 native-SAIF preflight.

This checker invokes no simulator, EDA tool, license query, or GPU workload.  It
only authenticates the sealed M2151 failure artifacts and derives the causal
failure fingerprint used by the accompanying review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
Q = HW / "results/m2151_m2149_m2018_ordinary_single_axis_native_saif_preflight_r1_20260904.failed.2551961.quarantine"
ATTEMPT = HW / "results/.m2151_m2149_ordinary_single_axis_native_saif_preflight_attempt_consumed"
CANONICAL = HW / "results/m2151_m2149_m2018_ordinary_single_axis_native_saif_preflight_r1_20260904"
LOCK = HW / "results/.m2151_m2149_ordinary_single_axis_native_saif_preflight_launch_lock"
M2150 = HW / "reviews/m2150_m2149_m2018_ordinary_single_axis_native_saif_preflight_source_hammer_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "quarantine_manifest_sha256": "c550ea84a5ff984c426131a8ae95ac5293a6abc173241d83269a09ce3858a474",
    "quarantine_outer_sha256": "90f9413d86c4666c2b3d10c798e1e62fc8b8558187126527a8a7ff84bfb5d3ed",
    "attempt_manifest_sha256": "cae90b374e249cf01312fb1a52b6643b76f82393eaac75e869d90b3a76be8dfe",
    "attempt_outer_sha256": "49b1ddbb88aa2593a0231f730453828711a6ca586ffde37ce7c33d40b68cb48e",
    "m2150_review_sha256": "133c1b5fbeadfd74b0080a3512b54aee1523dddb7a0b5ec69d2ea74ff58649ff",
    "m2150_manifest_sha256": "571f610986478c9dbc0658fc20628902b9412e274b4a945ba4a85b7660809087",
    "m2150_outer_sha256": "c5546a9954e4b545ca6923ac5266dcc682d93d47ab357f4a4f62c2120f1d72aa",
    "docs359_sha256": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(root: Path, manifest_sha: str, outer_sha: str) -> int:
    need(root.is_dir() and not root.is_symlink(), f"bad sealed root: {root}")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(sha256(manifest) == manifest_sha, f"manifest drift: {root}")
    need(sha256(outer) == outer_sha, f"outer drift: {root}")
    need(outer.read_text().split() == [sha256(manifest), "SHA256SUMS"],
         f"outer binding: {root}")
    listed: set[str] = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, f"bad manifest row: {row}")
        digest, rel = fields
        member = root / rel
        need(member.is_file() and not member.is_symlink(), f"bad member: {rel}")
        need(sha256(member) == digest, f"member drift: {rel}")
        listed.add(rel)
    actual = {str(path.relative_to(root)) for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    need(listed == actual, f"non-exhaustive seal: {root}")
    return len(actual)


def one(pattern: str, text: str, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    need(len(matches) == 1, f"{label} count={len(matches)}")
    return matches[0]


def main() -> int:
    q_members = verify_seal(
        Q, EXPECTED["quarantine_manifest_sha256"],
        EXPECTED["quarantine_outer_sha256"])
    attempt_members = verify_seal(
        ATTEMPT, EXPECTED["attempt_manifest_sha256"],
        EXPECTED["attempt_outer_sha256"])
    m2150_members = verify_seal(
        M2150, EXPECTED["m2150_manifest_sha256"],
        EXPECTED["m2150_outer_sha256"])
    need(sha256(M2150 / "review.json") == EXPECTED["m2150_review_sha256"],
         "M2150 review drift")
    need(sha256(DOC359) == EXPECTED["docs359_sha256"], "docs359 drift")
    need(not CANONICAL.exists() and not LOCK.exists(), "canonical/lock exists")

    attempt = json.loads((ATTEMPT / "attempt.json").read_text())
    need(attempt["status"] == "M2151_ATTEMPT_CONSUMED"
         and attempt["automatic_retry"] is False, "attempt disposition")
    counts = json.loads((Q / "execution_counts.json").read_text())
    need(counts == {
        "license_queries": 1, "vcs_compiles": 1, "simv_runs": 1,
        "raw_saif_files_written": 1, "admitted_saif_files": 0,
        "dc_runs": 0, "ptpx_runs": 0, "icc2_runs": 0, "gpu_runs": 0,
    }, "execution counts")

    compile_log = (Q / "vcs_compile.log").read_text(errors="replace")
    need("Top Level Modules:\n       tb_m2149_m2018_ordinary_single_axis_native_saif_preflight" in compile_log,
         "compile top")
    need("6 modules and 0 UDP read." in compile_log
         and "simv up to date" in compile_log, "compile completion")
    need(not re.search(r"(^|\n)(Error|Fatal)-|\bError:\s|\bFatal:\s",
                       compile_log), "compile error token")

    runtime = (Q / "rtl_sim.log").read_text(errors="replace")
    census = one(r"^M2149_INTERNAL_KNOWNNESS_CENSUS .*total=228/228 .*$",
                 runtime, "census")
    begin = one(r"^M2149_RTL_SAIF_WINDOW_BEGIN .*time_ns=1167\.01 .*$",
                runtime, "begin")
    phase2 = one(r"^M2149_UCLI_PHASE order=2 action=run_reset_and_preload .*$",
                 runtime, "phase2")
    phase3 = one(r"^M2149_UCLI_PHASE order=3 action=first_stop_reached .*$",
                 runtime, "phase3")
    warning = one(r"^Warning-\[SAIF_REPORT_BEFORE_RESET\] Toggle reporting not done$",
                  runtime, "reset warning")
    ignored = one(r"^  This request to reset power information will be ignored\.$",
                  runtime, "reset ignored")
    phase4 = one(r"^M2149_UCLI_PHASE order=4 action=power_reset .*$",
                 runtime, "phase4")
    end = one(
        r"^M2149_RTL_SAIF_WINDOW_END axis=ordinary_lru4 .*"
        r"measurement_cycles=20292 rows=149 issues=1278 products=29472 "
        r"commits=24 bundles=1788 scalar_weight_reads=14304 "
        r"duration_ns=60876\.00$", runtime, "end")
    passed = one(
        r"^PASS_M2149_ORDINARY_SINGLE_AXIS_NATIVE_SAIF_PREFLIGHT "
        r"ledger_exact=1 arithmetic_scoreboard_exact=1 .*frontends=1 "
        r"schedule_mode=0 second_axis=0 .*paper_citable=0$",
        runtime, "functional pass")
    phase5 = one(r"^M2149_UCLI_PHASE order=5 action=second_stop_reached .*$",
                 runtime, "phase5")
    need(census.start() < begin.start() < phase2.start() < phase3.start()
         < warning.start() < ignored.start() < phase4.start()
         < end.start() < passed.start() < phase5.start(), "actual chronology")
    need(not re.search(r"Assertion failed|\$fatal|\bFatal:\s|\bError:\s",
                       runtime), "runtime functional failure")
    need("Time: 62043010 ps" in runtime, "simulation final time")

    runtime_parse = (Q / "runtime_parse.log").read_text().strip()
    need(runtime_parse == "M2149_PARSE_FAIL_CLOSED: causal marker order",
         "runtime parser fingerprint")
    failed = (Q / "FAILED_DO_NOT_CITE.txt").read_text()
    need("command failed rc=2" in failed and "automatic_retry=false" in failed,
         "failure marker")

    saif_path = Q / "rtl_execute.saif"
    saif = saif_path.read_text(errors="replace")
    duration = one(r"\(DURATION\s+([0-9.]+)\)", saif, "SAIF duration")
    need(float(duration.group(1)) == 62043.01, "SAIF duration fingerprint")
    need(saif.count("(SAIFILE") == 1 and saif.count("(INSTANCE") == 0,
         "header-only SAIF topology")
    need(len(re.findall(r"\(T0\s+", saif)) == 0
         and saif_path.stat().st_size == 356, "header-only SAIF payload")
    need(re.search(r"\(DESIGN\s*\)", saif) is not None, "empty design field")

    result = {
        "status": "PASS_M2152_READ_ONLY_FAILURE_HAMMER",
        "review_execution": {
            "license_queries": 0, "vcs_compiles": 0, "simv_runs": 0,
            "saif_files": 0, "dc_runs": 0, "ptpx_runs": 0,
            "icc2_runs": 0, "gpu_runs": 0,
        },
        "sealed_members": {
            "m2151_quarantine": q_members,
            "m2151_attempt": attempt_members,
            "m2150_review": m2150_members,
        },
        "execution_counts": counts,
        "compile_and_function": {
            "vcs_compile_passed": True,
            "single_top_compiled": True,
            "functional_ledger_exact": True,
            "arithmetic_scoreboard_exact": True,
            "single_frontend_schedule_mode_zero": True,
            "second_axis_absent": True,
            "assertion_failures": 0,
        },
        "acquisition_failure": {
            "power_reset_explicitly_ignored": True,
            "prehistory_ns": 1167.01,
            "measurement_ns": 60876.0,
            "raw_saif_duration_ns": 62043.01,
            "raw_saif_bytes": 356,
            "raw_saif_instances": 0,
            "raw_saif_activity_records": 0,
            "runtime_parser_impossible_order_rejected": True,
            "admitted_saif_files": 0,
        },
        "disposition": {
            "attempt_consumed": True, "automatic_retry": False,
            "m2151_retry_authorized": False, "canonical_result_exists": False,
            "launch_lock_exists": False, "paper_citable": False,
        },
        "docs359_sha256": sha256(DOC359),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
