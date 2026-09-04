#!/usr/bin/env python3
"""Independent, read-only hammer of the exact M1183 M1168R3 VCS release.

This checker deliberately does not invoke the runner, VCS, simv, any EDA
binary, or a license client.  It checks both the release envelope and the
actual pre-attempt Python gate embedded in the exact runner.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import stat
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RELEASE = HW / "contracts/m1183_m1182_m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_launch_release_r3_20260830.json"
CONTRACT = HW / "contracts/m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_contract_r1_20260830.json"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m1168r3_m1162_c1_common_charge_protocol_exact_sha_r3.sh"
AUTHOR = HW / "reviews/m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_author_receipt_r1_20260830"
HAMMER = HW / "reviews/m1182_m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_source_hammer_r1_20260830"
RELEASE_AUTHOR = HW / "reviews/m1183_m1182_m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_release_author_receipt_r1_20260830"
AUTHOR_CHECK = RELEASE_AUTHOR / "static_no_eda_release_check.py"
R2_Q = HW / "results/m1168r2_m1162_c1_common_charge_protocol_unit_delay_vcs_r2_20260830.failed_or_incomplete.3284331.quarantine"
R3_ATTEMPT = HW / "results/.m1168r3_m1162_c1_common_charge_protocol_vcs_r3_attempt_consumed"
R3_RESULT = HW / "results/m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r3_20260830"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = HERE / "CHECK_OUTPUT.json"

EXPECTED_RELEASE = "cc285797c98784548933f86d98f410000f0036ac9dbdfe27f19cdd1f241c3403"
EXPECTED_REVIEW = "9216102c2298966d54ddd478e42734b01c25f1d4c685762fbe579d08b07bf96e"
EXPECTED_OUTER = "b2efc1076de8be88b420d2701f4e0b7dd065dfe449b45cc9bae3bdc84d16ac18"
EXPECTED_DOC359 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise AssertionError(f"duplicate key: {key}")
            out[key] = value
        return out
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def verify_leaf(path: Path) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    assert side.read_text().split() == [sha(path), path.name]
    assert outer.read_text().split() == [sha(side), side.name]


def verify_recursive(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert directory.is_dir() and not directory.is_symlink()
    assert outer.read_text().split() == [sha(manifest), "SHA256SUMS"]
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        p = Path(name)
        assert name not in listed and not p.is_absolute() and ".." not in p.parts
        listed[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        rel = member.relative_to(directory).as_posix()
        if rel in {"SHA256SUMS", "SHA256SUMS.seal.sha256"} or member.is_symlink():
            continue
        if stat.S_ISREG(member.lstat().st_mode):
            actual.add(rel)
    assert set(listed) == actual
    for name, digest in listed.items():
        assert sha(directory / name) == digest


def same_uid_eda_hits():
    blocked = {"vcs", "vcs1", "simv", "dc_shell", "pt_shell", "fm_shell",
               "icc2_shell", "common_shell_exec", "common_shell_exe"}
    ancestry = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestry:
        ancestry.add(pid)
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().split()[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            break
    hits = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in ancestry:
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            comm = (proc / "comm").read_text().strip()
            argv = [x.decode(errors="replace") for x in
                    (proc / "cmdline").read_bytes().split(b"\0") if x]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if comm in blocked or blocked.intersection(Path(arg).name for arg in argv):
            hits.append({"pid": int(proc.name), "comm": comm, "argv": argv[:4]})
    return hits


def memavailable_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise AssertionError("MemAvailable absent")


def validate_structural(d: dict) -> None:
    assert d["schema"] == "m1183_m1182_m1181_m1168r3_m1162_c1_common_charge_protocol_vcs_launch_release_r3_v1"
    assert d["status"] == "AUTHORIZE_EXACTLY_ONE_M1168R3_FUNCTIONAL_VCS_ATTEMPT"
    assert d["release"] is True and d["launch_now"] is False and d["inert_authoring_only"] is True
    ident = d["identity"]
    assert ident["runner_sha256"] == sha(RUNNER)
    assert ident["source_contract_sha256"] == sha(CONTRACT)
    assert ident["hammer_review_sha256"] == EXPECTED_REVIEW
    assert ident["hammer_outer_seal_file_sha256"] == EXPECTED_OUTER
    assert d["authorization"] == {"vcs_compiles": 1, "simv_runs": 1, "all_other_eda_runs": 0}
    assert d["required_environment"]["runner_arguments"] == 0
    assert d["required_environment"]["M1168R3_EXPECTED_HAMMER_REVIEW_SHA256"] == EXPECTED_REVIEW
    assert d["required_environment"]["M1168R3_EXPECTED_HAMMER_OUTER_SHA256"] == EXPECTED_OUTER
    assert d["unique_attempt"]["single_attempt"] is True
    assert d["unique_attempt"]["attempt_path"] == "results/.m1168r3_m1162_c1_common_charge_protocol_vcs_r3_attempt_consumed"
    assert d["unique_attempt"]["result_path"] == "results/m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r3_20260830"
    assert d["operational_gates"]["minimum_memavailable_kib"] == 67_108_864
    assert d["operational_gates"]["failure_quarantine_recursive_seal_required"] is True
    assert d["fresh_release_hammer_gate"]["required"] is True
    assert d["fresh_release_hammer_gate"]["direct_execution_before_fresh_release_hammer"] is False
    for key in ("functional_vcs_verified", "timing_verified", "cycles_measured", "speedup",
                "ppa", "power", "energy", "system_speedup", "paper_citable", "headline"):
        assert d["claim_boundary"][key] is False


def reject_mutation(base: dict, mutator) -> None:
    trial = copy.deepcopy(base)
    mutator(trial)
    try:
        validate_structural(trial)
    except (AssertionError, KeyError, TypeError):
        return
    raise AssertionError("mutation accepted")


def main() -> int:
    release = strict_json(RELEASE)
    runner = RUNNER.read_text()
    assert sha(RELEASE) == EXPECTED_RELEASE
    assert sha(HAMMER / "review.json") == EXPECTED_REVIEW
    assert sha(HAMMER / "SHA256SUMS.seal.sha256") == EXPECTED_OUTER
    assert sha(DOC359) == EXPECTED_DOC359
    verify_leaf(RELEASE)
    verify_leaf(CONTRACT)
    for directory in (AUTHOR, HAMMER, R2_Q, RELEASE_AUTHOR):
        verify_recursive(directory)

    # Reuse the author's frozen 29-file identity map as data, then independently
    # hash every byte.  The semantic/runtime checks below do not call its validator.
    spec = importlib.util.spec_from_file_location("m1183_author_check", AUTHOR_CHECK)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    assert len(module.EXPECTED) == 29
    for path, digest in module.EXPECTED.items():
        assert path.is_file() and not path.is_symlink() and sha(path) == digest

    validate_structural(release)
    mutations = [
        lambda d: d.__setitem__("status", "PASS_FUNCTIONAL_VCS_ONLY"),
        lambda d: d.__setitem__("launch_now", True),
        lambda d: d.__setitem__("inert_authoring_only", False),
        lambda d: d["identity"].__setitem__("runner_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("source_contract_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("hammer_review_sha256", "0" * 64),
        lambda d: d["identity"].__setitem__("hammer_outer_seal_file_sha256", "0" * 64),
        lambda d: d["authorization"].__setitem__("vcs_compiles", 2),
        lambda d: d["authorization"].__setitem__("simv_runs", 2),
        lambda d: d["authorization"].__setitem__("all_other_eda_runs", 1),
        lambda d: d["required_environment"].__setitem__("runner_arguments", 1),
        lambda d: d["required_environment"].__setitem__("M1168R3_EXPECTED_HAMMER_REVIEW_SHA256", "0" * 64),
        lambda d: d["unique_attempt"].__setitem__("single_attempt", False),
        lambda d: d["unique_attempt"].__setitem__("attempt_path", "results/r2"),
        lambda d: d["unique_attempt"].__setitem__("result_path", "results/alias"),
        lambda d: d["operational_gates"].__setitem__("minimum_memavailable_kib", 0),
        lambda d: d["operational_gates"].__setitem__("failure_quarantine_recursive_seal_required", False),
        lambda d: d["fresh_release_hammer_gate"].__setitem__("required", False),
        lambda d: d["fresh_release_hammer_gate"].__setitem__("direct_execution_before_fresh_release_hammer", True),
        lambda d: d["claim_boundary"].__setitem__("functional_vcs_verified", True),
        lambda d: d["claim_boundary"].__setitem__("timing_verified", True),
        lambda d: d["claim_boundary"].__setitem__("speedup", True),
        lambda d: d["claim_boundary"].__setitem__("paper_citable", True),
        lambda d: d["claim_boundary"].__setitem__("headline", True),
    ]
    for mutator in mutations:
        reject_mutation(release, mutator)

    vcs_calls = runner.count('"${VCS_BIN}" -full64')
    simv_calls = runner.count("./simv -no_save")
    unit_delay = "+define+UNIT_DELAY" in runner
    failure_seal = all(token in runner for token in
                       ('trap on_exit EXIT', 'seal_dir "${WORK}"', 'failed_or_incomplete.$$.quarantine'))
    normal_masks = all(token in runner for token in
                       ("legal_masks_clear=29", "protocol_attacks=7", "service_assumption_attacks=2"))
    same_uid_gate = "EDA collision" in runner
    memory_gate = "MemAvailable below 64 GiB" in runner
    assert (vcs_calls, simv_calls, unit_delay, failure_seal, normal_masks,
            same_uid_gate, memory_gate) == (1, 1, True, True, True, True, True)

    # Critical cross-artifact runtime compatibility: the exact runner's
    # pre-attempt Python gate dereferences a key absent from the exact release.
    runner_contract_key = "contract_sha256" if "i['contract_sha256']" in runner else None
    release_keys = sorted(release["identity"])
    deterministic_pre_attempt_keyerror = (
        runner_contract_key is not None and runner_contract_key not in release["identity"])

    # The release says a fresh release hammer is mandatory, but the exact runner
    # binds only M1182's source hammer.  No M1184/M1186 release-hammer digest is
    # accepted or checked at runtime.
    release_hammer_runtime_bound = bool(re.search(
        r"EXPECTED_(?:RELEASE_)?HAMMER.*M1184|M1184.*EXPECTED_.*HAMMER", runner))
    source_hammer_runtime_bound = (
        "M1168R3_EXPECTED_HAMMER_REVIEW_SHA256" in runner and
        "M1168R3_EXPECTED_HAMMER_OUTER_SHA256" in runner)

    mem = memavailable_kib()
    hits = same_uid_eda_hits()
    problems = []
    if deterministic_pre_attempt_keyerror:
        problems.append({
            "severity": "P0",
            "id": "RUNTIME_RELEASE_IDENTITY_KEY_MISMATCH",
            "detail": "Exact runner dereferences release.identity.contract_sha256, but exact release contains source_contract_sha256 and no contract_sha256. The no-argument launch deterministically raises KeyError before attempt creation; no VCS can run."
        })
    if not release_hammer_runtime_bound:
        problems.append({
            "severity": "P0",
            "id": "FRESH_RELEASE_HAMMER_NOT_RUNTIME_BOUND",
            "detail": "Exact runner accepts only the M1182 source-hammer review/outer digests. It does not consume or verify any fresh M1184/M1186 release-hammer digest, so the mandatory fresh release-hammer gate is not cryptographically enforced by the launch path."
        })

    output = {
        "schema": "m1186_m1183_m1168r3_c1_vcs_release_hammer_check_r1_v1",
        "status": "FAIL_CLOSED_M1186_M1183_RELEASE__SUCCESSOR_RUNNER_AND_RELEASE_REQUIRED__NO_VCS_NO_EDA",
        "verdict": "NO_GO",
        "score": 78,
        "issue_counts": {"P0": len([p for p in problems if p["severity"] == "P0"]), "P1": 0, "P2": 0},
        "problems": problems,
        "verified": {
            "exact_release_sha256": EXPECTED_RELEASE,
            "exact_source_hammer_review_sha256": EXPECTED_REVIEW,
            "exact_source_hammer_outer_sha256": EXPECTED_OUTER,
            "exact_files_verified": 29,
            "recursive_seals_verified": 4,
            "release_structural_mutations_rejected": len(mutations),
            "r2_quarantine_bound": True,
            "r3_attempt_absent": not R3_ATTEMPT.exists(),
            "r3_result_absent": not R3_RESULT.exists(),
            "r3_work_absent": not any((HW / "results").glob(".m1168r3_m1162_c1_common_charge_protocol_vcs_r3_work.*")),
            "r3_quarantine_absent": not any((HW / "results").glob("m1168r3_m1162_c1_common_charge_protocol_unit_delay_vcs_r3_20260830.failed_or_incomplete.*")),
            "vcs_compile_cardinality": vcs_calls,
            "simv_cardinality": simv_calls,
            "foundry_unit_delay": unit_delay,
            "same_uid_gate_present": same_uid_gate,
            "memory_gate_present": memory_gate,
            "failure_recursive_sealing_present": failure_seal,
            "normal_attack_masks_and_counts_pinned": normal_masks,
            "source_hammer_runtime_bound": source_hammer_runtime_bound,
            "release_hammer_runtime_bound": release_hammer_runtime_bound,
            "runner_required_identity_key": runner_contract_key,
            "release_identity_keys": release_keys,
            "docs359_sha256": sha(DOC359),
            "memavailable_kib_snapshot": mem,
            "same_uid_eda_hits_snapshot": hits,
        },
        "execution_audit": {"runner_invocations": 0, "vcs_compiles": 0, "simv_runs": 0,
                            "all_eda_runs": 0, "license_queries": 0,
                            "attempts_consumed": 0, "results_created": 0},
        "required_repair": [
            "Create additive successor runner and release; do not overwrite the exact R3 artifacts.",
            "Use one canonical contract identity key consistently in release and runner pre-attempt gate.",
            "Bind the fresh different-author release-hammer review and outer-seal digests through required launch environment and verify them before attempt creation.",
            "Fresh different-author source/release hammer the successor before the single VCS compile and simv attempt."
        ],
        "claim_boundary": {"functional_vcs_verified": False, "timing_verified": False,
                           "cycles_measured": False, "speedup": False, "ppa": False,
                           "power": False, "energy": False, "system_speedup": False,
                           "paper_citable": False, "headline": False}
    }
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
