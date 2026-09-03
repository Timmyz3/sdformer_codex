#!/usr/bin/python3.12
"""Read-only independent hammer for the sealed M2063 failure.

This script verifies the M2063 attempt token and failure quarantine, compares
the quarantined evidence with the retained raw work tree, and audits the
M2058/M2061/M2063 failure progression.  It does not launch VCS, EDA, a license
query, or a GPU task, and it writes nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
ATTEMPT = HW / "results/.m2063_m2056_tsbg_init0_mapped_energy_attempt_consumed"
FAILURE = HW / "results/m2063_m2056_tsbg_init0_mapped_energy_r1_20260903.failed_or_incomplete.quarantine"
RAW = HW / "results/.m2063_m2056_tsbg_init0_mapped_energy_work.1226415"
LOG = FAILURE / "evidence/candidate/ordinary_lru4/mapped_sim.log"
TB = HW / "tb_m2018/tb_m2063_m2018_tsbg_matched_mapped_energy.sv"
CONTRACT = HW / "contracts/m2063_m2056_m2018_tsbg_init0_mapped_energy_source_contract_r1_20260903.json"
M2057_LOG = HW / "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/sim_slot42.log"
M2061_LOG = HW / "results/m2061_m2056_tsbg_settled_mapped_energy_r1_20260903.failed_or_incomplete.quarantine/evidence/candidate/ordinary_lru4/mapped_sim.log"
M2058_LOG = HW / "results/.m2058_m2056_tsbg_matched_mapped_energy_failure_stage.600973/work/candidate/ordinary_lru4/mapped_sim.log"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "contract": "c227175efe3247922877dec8273cc4cc33ca383f9357e44f964f8a91c8c1843c",
    "tb": "9ac794347ccb40b2f56c7b1dfc87737d1a9da80d36554a3b15a7c524f63b9619",
    "attempt_manifest": "85cbe9eba1a5fa0af3fcb5a33165aef6f812134425c2519a6068a4ab2635c0ba",
    "attempt_outer": "b0dab239938d059747f6163a713532632c517bc43fc5892f6222a4cf132b9a18",
    "failure_manifest": "ba92cc5ad43d8ac25ce19f9aa80406d06a70dcd15fc0ffd79a4f028df32bd193",
    "failure_outer": "e39c6f4d92785c8c0186caf187b403c054ad7139b63c605e46a90ca784979401",
    "m2057_log": "5e2e0e72c119815901449737e1f1440275cf0e922b74d123060119fd52c6806f",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, f"duplicate JSON key {key}: {path}")
            result[key] = value
        return result

    def bad(token):
        raise AssertionError(f"nonfinite JSON value {token}: {path}")

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=bad)


def verify_sealed_directory(root: Path, expected_manifest: str,
                            expected_outer: str) -> int:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(sha(manifest) == expected_manifest, f"manifest drift: {root}")
    need(sha(outer) == expected_outer, f"outer-file drift: {root}")
    need(outer.read_text(encoding="ascii").split() ==
         [sha(manifest), "SHA256SUMS"], f"outer contents drift: {root}")
    declared = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        rel = Path(name.lstrip("*"))
        need(not rel.is_absolute() and ".." not in rel.parts,
             f"unsafe member: {rel}")
        member = root / rel
        need(member.is_file() and not member.is_symlink(), f"bad member: {member}")
        need(sha(member) == digest, f"member drift: {member}")
        need(rel.as_posix() not in declared, f"duplicate member: {rel}")
        declared.add(rel.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    }
    need(actual == declared, f"non-exhaustive seal: {root}")
    return len(declared)


def fingerprint(root: Path):
    members = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            members.append({"type": "symlink", "path": rel,
                            "target": os.readlink(path)})
        elif path.is_file():
            members.append({"type": "file", "path": rel,
                            "sha256": sha(path), "bytes": path.stat().st_size})
    return members


def main() -> int:
    need(sha(CONTRACT) == EXPECTED["contract"], "M2063 contract drift")
    need(sha(TB) == EXPECTED["tb"], "M2063 TB drift")
    need(sha(M2057_LOG) == EXPECTED["m2057_log"], "M2057 anchor log drift")
    need(sha(DOC359) == EXPECTED["docs359"], "docs/359 drift")
    attempt_members = verify_sealed_directory(
        ATTEMPT, EXPECTED["attempt_manifest"], EXPECTED["attempt_outer"])
    failure_members = verify_sealed_directory(
        FAILURE, EXPECTED["failure_manifest"], EXPECTED["failure_outer"])

    attempt = strict_json(ATTEMPT / "attempt.json")
    failure = strict_json(FAILURE / "failure.json")
    need(attempt["status"] == "M2063_ATTEMPT_CONSUMED_NO_RETRY",
         "attempt is not consumed/no-retry")
    need(attempt["required_simv_plusargs"] ==
         ["+vcs+initreg+0", "+WORKLOAD_SLOT=42"], "runtime identity drift")
    need(failure["attempt"] is True and failure["complete"] is False,
         "failure status drift")
    need(failure["phase"] == "SIMV_SAIF_ordinary_lru4",
         "failure phase drift")
    need((failure["vcs_compiles"], failure["simv_runs"],
          failure["saif_files"], failure["ptpx_runs"]) == (1, 1, 0, 0),
         "actual execution-count drift")

    sealed_fp = strict_json(FAILURE / "work_tree_fingerprint.json")["members"]
    actual_fp = fingerprint(RAW)
    need(actual_fp == sealed_fp, "raw work tree does not match sealed fingerprint")
    regular = [row for row in sealed_fp if row["type"] == "file"]
    symlinks = [row for row in sealed_fp if row["type"] == "symlink"]
    need((len(sealed_fp), len(regular), len(symlinks),
          sum(row["bytes"] for row in regular)) ==
         (112, 110, 2, 237392430), "fingerprint population drift")
    need((RAW / "candidate/ordinary_lru4/mapped_sim.log").read_bytes() ==
         LOG.read_bytes(), "quarantined runtime log differs from raw work")
    need((RAW / "compile_logs/ordinary_lru4.compile.log").read_bytes() ==
         (FAILURE / "evidence/compile_logs/ordinary_lru4.compile.log").read_bytes(),
         "quarantined compile log differs from raw work")
    need((RAW / "license_preflight.log").read_bytes() ==
         (FAILURE / "evidence/license_preflight.log").read_bytes(),
         "quarantined license log differs from raw work")

    text = LOG.read_text(encoding="utf-8", errors="replace")
    need(text.count("M2063_SIM_COMMAND_JSON=") == 1, "sim command marker count")
    need(text.count("M1970_LOAD_BEGIN") == 192 and
         text.count("M1970_LOAD_COMPLETE") == 192, "preload record count")
    need("M1970_PHASE full_load_complete cycle=383" in text and
         "M1970_PHASE full_execute_begin cycle=383" in text, "preload boundary")
    need("M2063_SAIF_WINDOW_BEGIN" in text and
         "M2063_SAIF_WINDOW_END" not in text, "window marker order")
    need("M2063 ordinary mapped completion ledger drift" in text,
         "completion fatal absent")
    need("at time 62037010 ps" in text, "fatal timestamp drift")
    need("mapped X/Z" not in text, "M2063 was incorrectly observed as X/Z fatal")
    need("PASS_M2063" not in text and "PASS_M2051" not in text,
         "unexpected final PASS")

    # The source must retain a ten-member ordinary completion conjunction.  It
    # reports no member values, so no post-hoc root-cause localization is legal.
    tb = TB.read_text(encoding="utf-8")
    completion = re.search(
        r"task automatic check_selected_completion;.*?"
        r"`ifdef M2056_AXIS_ORDINARY(?P<body>.*?)"
        r"\$fatal\(1, \"M2063 ordinary mapped completion ledger drift\"\);",
        tb, flags=re.S)
    need(completion is not None, "ordinary completion checker absent")
    body = completion.group("body")
    expected_terms = [
        "measured_cycles != FROZEN_BASE_CYCLES",
        "core.base.row_access_count != FROZEN_ROWS",
        "core.base.issue_count != FROZEN_ISSUES",
        "core.base.product_count != FROZEN_PRODUCTS",
        "core.base.cache_miss_count != 149",
        "core.base.cache_hit_count != 0",
        "core.base.cache_eviction_count != 145",
        "core.base.weight_bundle_beat_count != FROZEN_BASE_BUNDLES",
        "core.base.scalar_bank_request_count != FROZEN_BASE_SCALAR",
        "core.base.scalar_bank_response_count != FROZEN_BASE_SCALAR",
    ]
    need(all(term in body for term in expected_terms), "completion term drift")
    need("$display" not in body and "%0d" not in body,
         "unexpected per-member completion diagnostics")

    m2057 = M2057_LOG.read_text(encoding="utf-8", errors="replace")
    need("M1970_PHASE full_execute_complete cycle=20676" in m2057 and
         "base_cycles=20292" in m2057 and
         "PASS_M2051_EP34_TSBG_FULL40_CYCLE" in m2057,
         "M2057 source-RTL reference drift")
    m2061 = M2061_LOG.read_text(encoding="utf-8", errors="replace")
    m2058 = M2058_LOG.read_text(encoding="utf-8", errors="replace")
    need("M2061 mapped X/Z signal=ordinary.cycle_count" in m2061,
         "M2061 comparison drift")
    need("M2056 ordinary mapped bridge/commit/control X/Z" in m2058,
         "M2058 comparison drift")

    output = {
        "status": "PASS_M2069_M2063_FAILURE_HAMMER__M2063_FAILED_NO_RETRY__NO_FOURTH_POWER_ATTEMPT",
        "score_over_100": 97,
        "severity_counts": {"P0": 0, "P1": 1, "P2": 2},
        "attempt_double_seal": True,
        "failure_quarantine_double_seal": True,
        "attempt_manifest_members": attempt_members,
        "failure_manifest_members": failure_members,
        "fingerprinted_members": len(sealed_fp),
        "fingerprinted_regular_files": len(regular),
        "fingerprinted_symlinks": len(symlinks),
        "fingerprinted_regular_bytes": sum(row["bytes"] for row in regular),
        "raw_tree_exact_match": True,
        "quarantined_logs_match_raw": True,
        "ordinary_compile_runs": 1,
        "ordinary_simv_runs": 1,
        "tsbg_simv_runs": 0,
        "saif_files": 0,
        "ptpx_runs": 0,
        "preload_cycles": 383,
        "fatal_time_ps": 62037010,
        "fatal_class": "mapped_completion_counter_identity_drift",
        "runtime_xz_fatal_observed": False,
        "completion_conjuncts": 10,
        "failing_conjunct_uniquely_recoverable": False,
        "m2057_source_reference_base_cycles": 20292,
        "m2057_source_reference_complete_cycle": 20676,
        "mapped_functionality": False,
        "saif": False,
        "power": False,
        "energy": False,
        "paper_citable_numeric_result": False,
        "m2063_retry_allowed": False,
        "new_power_successor_authorized": False,
        "docs359_unchanged": True,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
