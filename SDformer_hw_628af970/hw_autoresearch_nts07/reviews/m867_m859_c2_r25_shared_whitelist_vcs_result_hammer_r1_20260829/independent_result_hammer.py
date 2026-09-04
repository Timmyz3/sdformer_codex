#!/usr/bin/env python3
"""Receipt-blind M859/C2 R25 canonical VCS result hammer.

This is intentionally standard-library-only and does not import the M859
publication guard.  It never runs VCS/simv or any other EDA tool.  Mutation
attacks are performed only on copies under a TemporaryDirectory.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Optional


HW = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESULT = HW / "results/m859_c2_r25_shared_whitelist_vcs_r1_20260829"
ATTEMPT = HW / "results/.m859_c2_r25_shared_whitelist_vcs_attempt_consumed"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m859_c2_r25_shared_whitelist_exact_sha.sh"
CONTRACT = HW / "contracts/m859_c2_r25_shared_whitelist_source_only_contract_r1_20260829.json"
CANDIDATE = HW / "contracts/m859_c2_r25_shared_whitelist_vcs_launch_candidate_source_only_r1_20260829.json"
RELEASE = HW / "contracts/m861_m859_c2_r25_shared_whitelist_vcs_launch_admission_r1_20260829.json"
FINAL = HW / "reviews/m862_m861_m859_c2_r25_shared_whitelist_final_launch_hammer_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

RUNNER_SHA = "da423b17f6245b0e9af9cc6df05a846e221175da45bfbce9408fe91930a9f8d6"
CONTRACT_SHA = "a7458798f11b0ba02d83072d93cf6185508de0e882eb9bf4c02a0b7380e66c5f"
CANDIDATE_SHA = "bf8599efc0ebce9b7e11b6d2ca38061b869c6555bddd620acb93a0ae3332696e"
RELEASE_SHA = "427c09a2da0f41911dcc3ee8c407f7f2ee5717318152ce74d9bb58d6ece3194e"
RELEASE_OUTER = "98ddfe00f2538093c033e4bb8db0e685a5c8ec0830abf9f0b9445a633782449d"
FINAL_OUTER = "b115c1f8dd8b88286de065f67a642c23496024ba6c7eec715a1b79f756b4bcd4"
DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
RECEIPT_NAME = "m859_c2_r25_shared_whitelist_vcs_receipt_r1.json"
RECEIPT_SCHEMA = "m859_c2_r25_shared_whitelist_vcs_receipt_v1"
RECEIPT_STATUS = "PASS_M859_R25_EXACT_VCS_PENDING_INDEPENDENT_RECEIPT_HAMMER"
FINAL_STATUS = "PASS100_M859_R25_SHARED_WHITELIST_FINAL_LAUNCH__ONE_VCS_ATTEMPT_AUTHORIZED"
RELEASE_STATUS = "AUTHORIZED_ONE_M859_R25_SHARED_WHITELIST_CHANNEL_SPLIT_VCS_ATTEMPT"

PAYLOAD = {
    "RUN_COMPLETE.txt", "launch_identity.txt", RECEIPT_NAME,
    "attack/compile.log", "attack/compile.rc", "attack/sim.log",
    "attack/sim.rc", "attack/assert.report",
    "attack/assert.report.disablelog", "equalbw/compile.log",
    "equalbw/compile.rc", "equalbw/sim.log", "equalbw/sim.rc",
    "equalbw/assert.report", "equalbw/assert.report.disablelog",
}
ALL_FILES = PAYLOAD | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}


class AuditFailure(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AuditFailure(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise AuditFailure("duplicate JSON key: " + key)
        value[key] = item
    return value


def strict_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=reject_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          AuditFailure("nonfinite JSON token: " + value)))


def parse_manifest(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/].*)", line)
        require(match is not None, "malformed manifest line")
        name = match.group(2)
        require(name not in result and not name.startswith("../") and
                "/../" not in name and not name.endswith("/.."),
                "duplicate or escaping manifest member")
        result[name] = match.group(1)
    return result


def verify_regular_tree(root: Path, expected_files, expected_dirs):
    require(root.is_dir() and not root.is_symlink(), "root not regular directory")
    files, directories = set(), set()
    for member in root.rglob("*"):
        relative = member.relative_to(root).as_posix()
        require(not member.is_symlink(), "symlink present: " + relative)
        if member.is_file():
            files.add(relative)
        elif member.is_dir():
            directories.add(relative)
        else:
            raise AuditFailure("nonregular member: " + relative)
    require(files == set(expected_files), "exact file population mismatch")
    require(directories == set(expected_dirs), "exact directory population mismatch")


def verify_recursive_result(root: Path):
    verify_regular_tree(root, ALL_FILES, {"attack", "equalbw"})
    manifest = parse_manifest(root / "SHA256SUMS")
    require(set(manifest) == PAYLOAD and len(manifest) == 15,
            "manifest must name exact 15 payload files")
    for name, expected in manifest.items():
        require(sha(root / name) == expected, "payload SHA drift: " + name)
    manifest_sha = sha(root / "SHA256SUMS")
    outer = (root / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8")
    require(outer == manifest_sha + "  SHA256SUMS\n", "outer seal drift")
    return {
        "manifest_sha256": manifest_sha,
        "outer_seal_file_sha256": sha(root / "SHA256SUMS.seal.sha256"),
        "payload_regular_files": 15,
        "files_including_seals": 17,
        "directories": 2,
        "symlinks": 0,
    }


def verify_flat_sealed(root: Path, expected_files=None):
    require(root.is_dir() and not root.is_symlink(), "flat seal root invalid")
    members = list(root.iterdir())
    require(all(p.is_file() and not p.is_symlink() for p in members),
            "flat sealed directory contains symlink/directory/nonregular member")
    files = {p.name for p in members}
    if expected_files is not None:
        require(files == set(expected_files), "flat sealed population mismatch")
    manifest = parse_manifest(root / "SHA256SUMS")
    require(set(manifest) == files - {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
            "flat manifest population mismatch")
    for name, expected in manifest.items():
        require(sha(root / name) == expected, "flat member SHA drift")
    manifest_sha = sha(root / "SHA256SUMS")
    require((root / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8") ==
            manifest_sha + "  SHA256SUMS\n", "flat outer seal drift")
    return {"manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": sha(root / "SHA256SUMS.seal.sha256")}


def verify_double_sealed_file(path: Path, expected_sha: str,
                              expected_outer: Optional[str] = None):
    require(path.is_file() and not path.is_symlink(), "double-sealed payload invalid")
    require(sha(path) == expected_sha, "double-sealed payload SHA drift")
    inner = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(inner.is_file() and not inner.is_symlink() and
            outer.is_file() and not outer.is_symlink(), "sidecars invalid")
    require(inner.read_text(encoding="utf-8") == expected_sha + "  " + path.name + "\n",
            "inner sidecar drift")
    inner_sha = sha(inner)
    require(outer.read_text(encoding="utf-8") == inner_sha + "  " + inner.name + "\n",
            "outer sidecar content drift")
    if expected_outer is not None:
        require(sha(outer) == expected_outer, "outer sidecar file SHA drift")


def exact_mapping_text(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        require(line.count("=") == 1, "launch identity malformed")
        key, value = line.split("=", 1)
        require(key not in result, "duplicate launch identity key")
        result[key] = value
    return result


NEGATIVE = re.compile(r"failed at|Offending|Assertion\s+fail|^Error(?:-|\[|:)|^Fatal|Fatal:|watchdog", re.I | re.M)


def match_count(report: str, suffix: str) -> int:
    matches = re.findall(r"\." + re.escape(suffix) + r",\s+\d+ attempts,\s+(\d+) match", report)
    require(matches, "missing assertion cover: " + suffix)
    return max(int(value) for value in matches)


def parse_attack(root: Path):
    compile_log = (root / "attack/compile.log").read_text(encoding="utf-8")
    sim_log = (root / "attack/sim.log").read_text(encoding="utf-8")
    report = (root / "attack/assert.report").read_text(encoding="utf-8")
    require((root / "attack/compile.rc").read_text() == "0\n" and
            (root / "attack/sim.rc").read_text() == "0\n", "attack rc nonzero")
    require("Version V-2023.12-SP1_Full64" in compile_log and
            "All of 3 modules done" in compile_log, "attack compile completion missing")
    require(not NEGATIVE.search(compile_log + "\n" + sim_log + "\n" + report),
            "attack error/assertion failure token")
    pattern = re.compile(
        r"^PASS M803 R16 channel-split cutthrough adapter VCS "
        r"attack_classes=(\d+) reset_cases=(\d+) legal_response_on_request_fault=(\d+) "
        r"same_cycle_reuse_cases=(\d+) sticky_quiescent_checks=(\d+) "
        r"normal_requests=(\d+) normal_responses=(\d+) "
        r"request_side_effect_violations=(\d+) response_side_effect_violations=(\d+)$", re.M)
    matches = pattern.findall(sim_log)
    require(matches == [("12", "12", "2", "1", "10", "6", "5", "0", "0")],
            "attack PASS tuple drift")
    required_covers = [
        "cp_legal_response_illegal_request_same_cycle",
        "cp_illegal_response_legal_request_same_cycle",
        "cp_pending_drain_request_attack",
        "cp_response_backpressure_then_attack",
        "cp_held_response_request_attack_retire",
        "cp_cutthrough_request_attack_retire",
        "cp_same_cycle_slot_reuse", "cp_sticky_fault_quiescent",
        "cp_protocol_attack",
    ]
    covers = {name: match_count(report, name) for name in required_covers}
    require(all(value > 0 for value in covers.values()), "zero mandatory attack cover")
    return {
        "compile_rc": 0, "sim_rc": 0, "attack_classes": 12,
        "reset_cases": 12, "same_cycle_reuse_cases": 1,
        "legal_response_on_request_fault": 2,
        "side_effect_violations": 0, "assertion_covers": covers,
    }


def parse_equalbw(root: Path):
    compile_log = (root / "equalbw/compile.log").read_text(encoding="utf-8")
    sim_log = (root / "equalbw/sim.log").read_text(encoding="utf-8")
    report = (root / "equalbw/assert.report").read_text(encoding="utf-8")
    require((root / "equalbw/compile.rc").read_text() == "0\n" and
            (root / "equalbw/sim.rc").read_text() == "0\n", "equalbw rc nonzero")
    require("Version V-2023.12-SP1_Full64" in compile_log and
            "All of 17 modules done" in compile_log, "equalbw compile completion missing")
    require(not NEGATIVE.search(compile_log + "\n" + sim_log + "\n" + report),
            "equalbw error/assertion failure token")
    row_pattern = re.compile(
        r"^M803EQ cutthrough equalbw B=(\d+) events=(\d+) "
        r"k8_cycles=(\d+) k1x8_cycles=(\d+) speedup=([0-9.]+) "
        r"tuple_mismatches=(\d+) weight_mismatches=(\d+)$", re.M)
    rows = row_pattern.findall(sim_log)
    expected = [
        ("1", "20", "51", "53"), ("2", "41", "131", "133"),
        ("4", "90", "486", "499"), ("8", "110", "1231", "1246"),
        ("1", "0", "14", "14"),
    ]
    require(len(rows) == 5, "equalbw row count drift")
    require([row[:4] for row in rows] == expected, "exact cycle rows drift")
    require(all(row[5:] == ("0", "0") for row in rows), "row mismatch nonzero")
    pass_pattern = re.compile(
        r"^PASS M803EQ channel-split cutthrough-8bank equal-bandwidth FC2 VCS "
        r"clean_cases=(\d+) exact_cycle_cases=(\d+) cycles=([^ ]+) reset_cases=(\d+) "
        r"protocol_attacks=(\d+) numeric_mismatches=(\d+) tuple_mismatches=(\d+) "
        r"weight_mismatches=(\d+) service_sva_bound=(true|false) "
        r"adapter_sva_bound=(true|false) racefree_cycle_monitor=(true|false) "
        r"request_stalls=(\d+) result_stalls=(\d+) raw_stalls=(\d+) "
        r"full8_requests=(\d+) k1x8_full_issue=(\d+) "
        r"candidate_younger_before_older=(\d+) baseline_younger_before_older=(\d+)$", re.M)
    matches = pass_pattern.findall(sim_log)
    require(len(matches) == 1, "equalbw PASS token count drift")
    value = matches[0]
    require(value[:2] == ("10", "5") and
            value[2] == "51/53,131/133,486/499,1231/1246,14/14" and
            value[3:5] == ("2", "4") and value[5:8] == ("0", "0", "0") and
            value[8:11] == ("true", "true", "true") and
            all(int(item) > 0 for item in value[11:]), "equalbw PASS fields drift")
    covers = {
        name: match_count(report, name) for name in [
            "cp_full_eight_bank_request", "cp_pending_request_stall",
            "cp_same_cycle_slot_reuse", "cp_protocol_attack",
            "cp_out_of_order_bundle_response", "cp_k8_request",
            "cp_result_stall", "cp_all_eight_lane_group",
            "cp_eight_requests_same_cycle",
        ]
    }
    require(all(value > 0 for value in covers.values()), "zero mandatory equalbw cover")
    return {
        "compile_rc": 0, "sim_rc": 0, "clean_cases": 10,
        "exact_cycle_cases": 5, "protocol_attacks": 4,
        "numeric_mismatches": 0, "tuple_mismatches": 0,
        "weight_mismatches": 0,
        "exact_cycles": {"k8": [51, 131, 486, 1231, 14],
                         "k1x8": [53, 133, 499, 1246, 14]},
        "nonzero_runtime_covers": {name: int(value) for name, value in zip(
            ["request_stalls", "result_stalls", "raw_stalls", "full8_requests",
             "k1x8_full_issue", "candidate_younger_before_older",
             "baseline_younger_before_older"], value[11:])},
        "assertion_covers": covers,
    }


def expected_receipt():
    return {
        "schema": RECEIPT_SCHEMA, "status": RECEIPT_STATUS,
        "runner_sha256": RUNNER_SHA, "contract_sha256": CONTRACT_SHA,
        "candidate_sha256": CANDIDATE_SHA, "release_sha256": RELEASE_SHA,
        "final_hammer_outer_seal_sha256": FINAL_OUTER,
        "tool": "Synopsys VCS V-2023.12-SP1",
        "publication": {
            "source": "PRIVATE_VCS_WORK_WITH_TOOL_SYMLINKS_ALLOWED",
            "canonical": "EXACT_15_REGULAR_FILE_SHARED_WHITELIST_DOUBLE_SEALED",
            "source_dev_inode_size_sha_pre_post_stable": True,
            "shared_whitelist_authority": "verif_m859.m859_c2_r25_shared_whitelist_guard.WHITELIST",
        },
        "attack_contract": {
            "same_cycle_slot_reuse": 1, "ledger_conservation": True,
            "illegal_response_closes_both": True,
            "legal_response_survives_request_fault": True,
        },
        "exact_cycles": {"k8": [51, 131, 486, 1231, 14],
                         "k1x8": [53, 133, 499, 1246, 14]},
        "frozen_k1_vs_k1x8": "SOURCE_SHA_BOUND_ONLY__NOT_RERUN_OR_CHANGED",
        "claim_boundary": {
            "vcs_validated": True, "dc": False, "ppa": False,
            "system_speedup": False, "headline": False, "paper_citable": False,
        },
    }


def verify_semantics(root: Path):
    receipt = strict_json(root / RECEIPT_NAME)
    require(receipt == expected_receipt(), "receipt exact identity/status/schema drift")
    require((root / "RUN_COMPLETE.txt").read_text(encoding="utf-8") == RECEIPT_STATUS + "\n",
            "RUN_COMPLETE drift")
    expected_binding = {
        "runner_sha256": RUNNER_SHA, "contract_sha256": CONTRACT_SHA,
        "candidate_sha256": CANDIDATE_SHA, "release_sha256": RELEASE_SHA,
        "final_hammer_outer_seal_sha256": FINAL_OUTER,
    }
    require(exact_mapping_text(root / "launch_identity.txt") == expected_binding,
            "launch identity binding drift")
    attack = parse_attack(root)
    equalbw = parse_equalbw(root)
    return receipt, attack, equalbw


def verify_attempt():
    identity = verify_flat_sealed(
        ATTEMPT, {"attempt.json", "SHA256SUMS", "SHA256SUMS.seal.sha256"})
    value = strict_json(ATTEMPT / "attempt.json")
    require(value == {
        "schema": "m826_c2_r20_atomic_vcs_attempt_v1",
        "status": "ONE_M826_R20_VCS_ATTEMPT_CONSUMED",
        "runner_sha256": RUNNER_SHA, "contract_sha256": CONTRACT_SHA,
        "candidate_sha256": CANDIDATE_SHA, "release_sha256": RELEASE_SHA,
        "final_hammer_outer_seal_sha256": FINAL_OUTER,
        "claim_boundary": {"vcs_complete": False, "paper_citable": False,
                           "system_speedup": False},
    }, "attempt exact identity drift")
    siblings = sorted(p.name for p in (HW / "results").iterdir()
                      if "m859_c2_r25_shared_whitelist_vcs" in p.name and
                      ("attempt" in p.name or "failed_or_incomplete" in p.name))
    require(siblings == [ATTEMPT.name], "attempt/quarantine population drift")
    return identity


def verify_authorities():
    verify_double_sealed_file(RUNNER, RUNNER_SHA)
    verify_double_sealed_file(CONTRACT, CONTRACT_SHA)
    verify_double_sealed_file(CANDIDATE, CANDIDATE_SHA)
    verify_double_sealed_file(RELEASE, RELEASE_SHA, RELEASE_OUTER)
    require(sha(DOC359) == DOC359_SHA, "docs/359 drift")
    final_identity = verify_flat_sealed(FINAL)
    require(final_identity["outer_seal_file_sha256"] == FINAL_OUTER,
            "final hammer outer binding drift")
    release = strict_json(RELEASE)
    final = strict_json(FINAL / "review.json")
    require(release.get("status") == RELEASE_STATUS and
            release.get("authorization", {}).get("max_attempts") == 1,
            "release semantics drift")
    require(final.get("status") == FINAL_STATUS and final.get("score_out_of_100") == 100 and
            (final.get("p0_count"), final.get("p1_count"), final.get("p2_count")) == (0, 0, 0),
            "final hammer semantics drift")
    require(final.get("review_target") == {
        "release_sha256": RELEASE_SHA, "runner_sha256": RUNNER_SHA,
        "contract_sha256": CONTRACT_SHA, "candidate_sha256": CANDIDATE_SHA,
    }, "final hammer target drift")
    return final_identity


def reseal(root: Path):
    lines = []
    for name in sorted(PAYLOAD):
        lines.append(sha(root / name) + "  " + name + "\n")
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha(root / "SHA256SUMS") + "  SHA256SUMS\n", encoding="utf-8")


def expect_reject(label, mutate, semantic=True):
    with tempfile.TemporaryDirectory(prefix="m867_result_attack_") as temp:
        copy = Path(temp) / "copy"
        shutil.copytree(RESULT, copy, symlinks=True)
        mutate(copy)
        rejected = False
        try:
            verify_recursive_result(copy)
            if semantic:
                verify_semantics(copy)
        except (AuditFailure, OSError, UnicodeError, json.JSONDecodeError):
            rejected = True
        require(rejected, "mutation attack accepted: " + label)
    return label


def json_mutation(path: Path, fn):
    value = strict_json(path)
    fn(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def run_mutation_attacks():
    attacks = []
    attacks.append(expect_reject("payload_byte_flip_unsealed",
        lambda root: (root / "attack/sim.log").write_bytes(
            (root / "attack/sim.log").read_bytes() + b"X")))
    attacks.append(expect_reject("extra_regular_file",
        lambda root: (root / "extra.txt").write_text("x\n")))
    def symlink_attack(root):
        (root / "attack/compile.rc").unlink()
        (root / "attack/compile.rc").symlink_to("sim.rc")
    attacks.append(expect_reject("payload_symlink", symlink_attack))
    attacks.append(expect_reject("manifest_byte_flip",
        lambda root: (root / "SHA256SUMS").write_bytes(
            (root / "SHA256SUMS").read_bytes() + b"X")))
    attacks.append(expect_reject("outer_seal_byte_flip",
        lambda root: (root / "SHA256SUMS.seal.sha256").write_bytes(
            (root / "SHA256SUMS.seal.sha256").read_bytes() + b"X")))
    def mutate_receipt(root, field, value):
        json_mutation(root / RECEIPT_NAME, lambda data: data.__setitem__(field, value))
        reseal(root)
    attacks.append(expect_reject("receipt_status_resealed",
        lambda root: mutate_receipt(root, "status", "PASS_BUT_WRONG")))
    attacks.append(expect_reject("receipt_runner_binding_resealed",
        lambda root: mutate_receipt(root, "runner_sha256", "0" * 64)))
    def rc_attack(root):
        (root / "equalbw/sim.rc").write_text("1\n")
        reseal(root)
    attacks.append(expect_reject("equalbw_nonzero_rc_resealed", rc_attack))
    def cycle_attack(root):
        path = root / "equalbw/sim.log"
        path.write_text(path.read_text().replace("k8_cycles=51", "k8_cycles=52", 1))
        reseal(root)
    attacks.append(expect_reject("exact_cycle_drift_resealed", cycle_attack))
    def mismatch_attack(root):
        path = root / "equalbw/sim.log"
        path.write_text(path.read_text().replace("numeric_mismatches=0", "numeric_mismatches=1", 1))
        reseal(root)
    attacks.append(expect_reject("numeric_mismatch_resealed", mismatch_attack))
    def cover_attack(root):
        path = root / "equalbw/sim.log"
        path.write_text(path.read_text().replace("full8_requests=882", "full8_requests=0", 1))
        reseal(root)
    attacks.append(expect_reject("zero_coverage_resealed", cover_attack))
    def assertion_attack(root):
        path = root / "attack/assert.report"
        path.write_text(path.read_text() + "Assertion failed at 1ns\n")
        reseal(root)
    attacks.append(expect_reject("assertion_failure_resealed", assertion_attack))
    return attacks


def seal_review():
    members = ["RUN_COMPLETE.txt", "independent_result_hammer.py",
               "mechanical_checks.txt", "review.json", "review.md"]
    lines = [sha(OUT / name) + "  " + name + "\n" for name in members]
    (OUT / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        sha(OUT / "SHA256SUMS") + "  SHA256SUMS\n", encoding="utf-8")
    verify_flat_sealed(OUT, set(members) | {"SHA256SUMS", "SHA256SUMS.seal.sha256"})


def main():
    result_identity = verify_recursive_result(RESULT)
    final_identity = verify_authorities()
    attempt_identity = verify_attempt()
    receipt, attack, equalbw = verify_semantics(RESULT)
    attacks = run_mutation_attacks()
    review = {
        "schema": "m867_m859_c2_r25_shared_whitelist_vcs_result_hammer_v1",
        "date": "2026-08-29",
        "status": "PASS100_M859_R25_DIRECTED_COMPONENT_VCS_E3_RESULT_ADMITTED",
        "verdict": "PASS",
        "score_out_of_100": 100,
        "p0_count": 0, "p1_count": 0, "p2_count": 0,
        "p0": [], "p1": [], "p2": [],
        "reviewer_role": "Fresh independent receipt/result hammer; no VCS, simv, license, DC, PT, FM, PTPX, GPU, remote, or other EDA execution.",
        "identity": {
            "canonical_result": "results/m859_c2_r25_shared_whitelist_vcs_r1_20260829",
            "canonical_result_seal": result_identity,
            "receipt_sha256": sha(RESULT / RECEIPT_NAME),
            "runner_sha256": RUNNER_SHA, "contract_sha256": CONTRACT_SHA,
            "candidate_sha256": CANDIDATE_SHA, "release_sha256": RELEASE_SHA,
            "release_outer_seal_file_sha256": RELEASE_OUTER,
            "final_hammer_outer_seal_file_sha256": FINAL_OUTER,
            "final_hammer_manifest_sha256": final_identity["manifest_sha256"],
            "attempt": attempt_identity,
            "docs359_sha256": DOC359_SHA,
        },
        "canonical_population": {
            "regular_payload_files": 15, "files_including_two_seals": 17,
            "directories": ["attack", "equalbw"], "symlinks": 0,
            "shared_whitelist_exact": True, "recursive_double_seal": "PASS",
        },
        "vcs_evidence": {
            "tool": receipt["tool"], "attack": attack, "equal_bandwidth": equalbw,
        },
        "mutation_attacks": {"passed": len(attacks), "failed": 0, "names": attacks,
                             "canonical_modified": False},
        "execution_receipt": {
            "vcs_runs_by_reviewer": 0, "simv_runs_by_reviewer": 0,
            "license_queries_by_reviewer": 0, "eda_runs_by_reviewer": 0,
            "dc_runs_by_reviewer": 0, "pt_runs_by_reviewer": 0,
            "formality_runs_by_reviewer": 0, "ptpx_runs_by_reviewer": 0,
            "gpu_or_remote_jobs_by_reviewer": 0,
            "canonical_result_modified": False, "m803_frozen_source_modified": False,
            "docs359_modified": False,
        },
        "claim_boundary": {
            "directed_component_vcs_e3_citable": True,
            "functional_scope": "M803/C2 signed K8 channel-split adapter plus equal-bandwidth K1x8 component workload",
            "exact_directed_cycle_tuple_citable_only_with_scope": True,
            "exact_cycles": "K8/K1x8=51/53,131/133,486/499,1231/1246,14/14",
            "rtl_random_or_full_network_validation": False,
            "dc_or_physical_ppa": False, "timing": False, "energy": False,
            "component_speedup_headline": False, "system_speedup": False,
            "paper_headline": False,
            "paper_usage": "May cite only as directed component-level Synopsys VCS E3 functional/cycle evidence with exact equal-bandwidth workload labels; not as DC/PPA/energy/system performance.",
        },
        "required_next_gate": "A separate authorized matched K1/K8/K1x8 Synopsys DC result and fresh result hammer are required before any area, timing, throughput-per-area, PPA, or energy statement.",
    }
    mechanical = [
        "PASS independent recursive double seal and exact 15-payload whitelist",
        "PASS receipt schema/status and runner/contract/candidate/release/final bindings",
        "PASS one consumed attempt and zero M859 failure quarantines",
        "PASS attack compile/sim rc=0, no errors/assertion failures, mandatory covers nonzero",
        "PASS equalbw compile/sim rc=0, all mismatches zero, stalls/attacks/covers nonzero",
        "PASS exact cycles K8/K1x8 51/53,131/133,486/499,1231/1246,14/14",
        "PASS sealed-copy mutation attacks %d/%d" % (len(attacks), len(attacks)),
        "PASS P0=0 P1=0 P2=0",
    ]
    (OUT / "mechanical_checks.txt").write_text("\n".join(mechanical) + "\n", encoding="utf-8")
    (OUT / "review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "review.md").write_text(
        "# M867 — M859/C2 R25 canonical VCS result hammer\n\n"
        "**PASS 100/100; P0/P1/P2 = 0/0/0.**\n\n"
        "The exact 15-file canonical payload is regular-file-only, recursively double-sealed, and bound to the reviewed runner, release, final hammer, and consumed attempt. Independent parsing finds both compile/simulation return codes zero, no error/assertion-failure token, nonzero protocol/coverage activity, and zero numeric/tuple/weight mismatches.\n\n"
        "Admitted E3 evidence is limited to the directed component VCS tuple: `K8/K1x8 = 51/53, 131/133, 486/499, 1231/1246, 14/14`. This is not DC, timing, PPA, energy, full-network speedup, or a headline performance claim.\n\n"
        "Twelve mutation attacks on sealed copies were rejected; canonical artifacts and frozen M803/docs/359 were not changed.\n",
        encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text(
        "PASS100_M859_R25_DIRECTED_COMPONENT_VCS_E3_RESULT_ADMITTED\n", encoding="utf-8")
    seal_review()
    print("PASS100 M867 M859/C2 R25 independent result hammer")


if __name__ == "__main__":
    main()
