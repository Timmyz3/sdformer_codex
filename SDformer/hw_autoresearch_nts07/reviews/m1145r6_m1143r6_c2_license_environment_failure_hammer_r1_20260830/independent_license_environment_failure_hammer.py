#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Independent read-only M1143R6 license-environment failure hammer.

This script never launches VCS/DC or reads a license-route value into an
artifact.  Route identity is represented only by presence, byte length, and
SHA-256.  It audits the already sealed failure and authorizes source authoring
only; the consumed namespace remains non-retryable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path


HW = Path("/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07")
RESULTS = HW / "results"
REVIEW = HW / "reviews/m1145r6_m1143r6_c2_license_environment_failure_hammer_r1_20260830"
ATTEMPT = RESULTS / ".m1143r6_c2_frozen_netlist_mapped_vcs_successor_attempt_consumed"
FAILURE = RESULTS / ("m1143r6_c2_frozen_netlist_mapped_vcs_successor_r1_20260830."
                     "failed_or_incomplete.1671825.1788051182241746176.quarantine")
RESULT = RESULTS / "m1143r6_c2_frozen_netlist_mapped_vcs_successor_r1_20260830"
SOURCE = HW / "dc_handoff/scripts/run_m1143r6_c2_frozen_netlist_mapped_vcs_successor_source_r1.py"
ORIGINAL = HW / "dc_handoff/scripts/m1129r5_c2_real_module_async_observation_engine_source_r1.py"
AUTHOR = HW / "reviews/m1143r6_c2_frozen_netlist_mapped_vcs_successor_author_receipt_r1_20260830"
HAMMER = HW / "reviews/m1144r6_m1143r6_c2_final_source_launch_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "attempt_outer": "ebdb51e51cbb7a585a4d4b9bab20e48b2c1c510211de6eed014f0e3a2bdd527d",
    "failure_outer": "7607b08d35b8c76116f5b85b30e236e93a5339a2670185df025435909d03c06b",
    "source": "d112129e9c068d4b609852fc8e824dd986f6d3f923bf2cf132b3a6ac28298471",
    "original": "c8fd3366ecf6c4377b62e5717d959348c08192ea8bdbd0afd3b0e566bd6fbd0b",
    "author_outer": "7845dcb40c198c2ac92eb4324f16cf3a007e02b7112ac974baceb973f7d2cc31",
    "hammer_outer": "d893c976df2eda5d15dc228859a4a072a6d44b5030acaa94bb2137955b161201",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "SNPSLMD_LICENSE_FILE": {
        "present": True, "length": 18,
        "sha256": "5662b8aedce52c58e4cf6ac8e7b888f6ecf16cef8fb7998d724582d030af1d7b",
    },
    "LM_LICENSE_FILE": {
        "present": True, "length": 26,
        "sha256": "26bf3db829e694cbc7c0886311b8c5fedf2887e143313efcc81854e57f718adc",
    },
}


class AuditFailure(RuntimeError):
    pass


checks = 0


def require(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        raise AuditFailure(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sealed(directory: Path, expected_outer: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(), f"not regular dir: {directory}")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "manifest absent/symlink")
    require(outer.is_file() and not outer.is_symlink(), "outer absent/symlink")
    require(sha256(outer) == expected_outer, "outer identity drift")
    outer_fields = outer.read_text(encoding="utf-8").strip().split()
    require(len(outer_fields) == 2 and outer_fields[1] == "SHA256SUMS", "outer schema drift")
    require(sha256(manifest) == outer_fields[0], "manifest outer mismatch")
    listed = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.strip()
        member = directory / rel
        require(member.is_file() and not member.is_symlink(), f"member absent/symlink: {rel}")
        require(sha256(member) == digest, f"member digest drift: {rel}")
        listed.append(rel)
    actual = sorted(str(p.relative_to(directory)) for p in directory.rglob("*")
                    if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    require(sorted(listed) == actual, "unsealed member or stale manifest")


def environment_literal_keys(source_text: str) -> set[str]:
    tree = ast.parse(source_text)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and
                any(isinstance(target, ast.Name) and target.id == "environment"
                    for target in node.targets) and isinstance(node.value, ast.Dict)):
            keys = set()
            for key in node.value.keys:
                require(isinstance(key, ast.Constant) and isinstance(key.value, str),
                        "environment has non-literal key")
                keys.add(key.value)
            return keys
    raise AuditFailure("literal environment assignment absent")


def audit_source_text(source_text: str, original_text: str) -> dict:
    successor_keys = environment_literal_keys(source_text)
    require(successor_keys == {"LANG", "LC_ALL", "PATH", "VCS_HOME", "HOME"},
            "M1143 clean environment keys drift")
    require('"HOME": "/tmp"' in source_text, "M1143 HOME override evidence absent")
    require("SNPSLMD_LICENSE_FILE" not in successor_keys and
            "LM_LICENSE_FILE" not in successor_keys,
            "M1143 license omission evidence absent")
    require('route = os.environ.get("SNPSLMD_LICENSE_FILE") or os.environ.get("LM_LICENSE_FILE")'
            in original_text, "original route precedence/preflight drift")
    require('env = os.environ.copy()' in original_text,
            "original environment forwarding drift")
    require('LMUTIL), "lmstat", "-a", "-c", route' in original_text,
            "original license health preflight drift")
    return {
        "m1143_environment_keys": sorted(successor_keys),
        "m1143_license_keys": [],
        "m1143_home_override": True,
        "m1129_precedence": ["SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"],
        "m1129_caller_environment_forwarded": True,
        "m1129_lmstat_preflight": True,
    }


def route_metadata() -> dict:
    result = {}
    for name in ("SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"):
        value = os.environ.get(name)
        metadata = {
            "present": bool(value),
            "length": len(value) if value else 0,
            "sha256": hashlib.sha256(value.encode()).hexdigest() if value else None,
        }
        require(metadata == EXPECTED[name], f"current {name} identity drift")
        result[name] = metadata
    return result


def mutation_attacks(source_text: str, original_text: str) -> dict:
    attacks = {}
    mutations = {
        "hide_home_override": source_text.replace('"HOME": "/tmp"', '"TMP_ONLY": "/tmp"', 1),
        "inject_snps_route_into_failed_source": source_text.replace(
            '"HOME": "/tmp"', '"HOME": "/tmp", "SNPSLMD_LICENSE_FILE": "redacted"', 1),
        "inject_lm_route_into_failed_source": source_text.replace(
            '"HOME": "/tmp"', '"HOME": "/tmp", "LM_LICENSE_FILE": "redacted"', 1),
    }
    for name, mutant in mutations.items():
        try:
            audit_source_text(mutant, original_text)
        except (AuditFailure, SyntaxError) as exc:
            attacks[name] = type(exc).__name__ + ": " + str(exc)
        else:
            raise AuditFailure(f"mutation survived: {name}")
    original_mutations = {
        "remove_original_env_forwarding": original_text.replace(
            "env = os.environ.copy()", "env = {}", 1),
        "reverse_original_route_precedence": original_text.replace(
            'route = os.environ.get("SNPSLMD_LICENSE_FILE") or os.environ.get("LM_LICENSE_FILE")',
            'route = os.environ.get("LM_LICENSE_FILE") or os.environ.get("SNPSLMD_LICENSE_FILE")', 1),
        "remove_original_lmstat": original_text.replace(
            'LMUTIL), "lmstat", "-a", "-c", route',
            'LMUTIL), "version"', 1),
    }
    for name, mutant_original in original_mutations.items():
        try:
            audit_source_text(source_text, mutant_original)
        except (AuditFailure, SyntaxError) as exc:
            attacks[name] = type(exc).__name__ + ": " + str(exc)
        else:
            raise AuditFailure(f"mutation survived: {name}")
    require(len(attacks) == 6, "attack count drift")
    return attacks


def seal(directory: Path) -> tuple[str, str]:
    members = sorted(p for p in directory.iterdir()
                     if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = "".join(f"{sha256(p)}  {p.name}\n" for p in members)
    (directory / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    manifest_sha = sha256(directory / "SHA256SUMS")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        f"{manifest_sha}  SHA256SUMS\n", encoding="utf-8")
    return manifest_sha, sha256(directory / "SHA256SUMS.seal.sha256")


def main() -> None:
    require(sha256(SOURCE) == EXPECTED["source"], "M1143 source drift")
    require(sha256(ORIGINAL) == EXPECTED["original"], "M1129 source drift")
    require(sha256(DOCS359) == EXPECTED["docs359"], "docs/359 drift")
    verify_sealed(ATTEMPT, EXPECTED["attempt_outer"])
    verify_sealed(FAILURE, EXPECTED["failure_outer"])
    verify_sealed(AUTHOR, EXPECTED["author_outer"])
    verify_sealed(HAMMER, EXPECTED["hammer_outer"])

    names = sorted(p.name for p in RESULTS.iterdir()
                   if p.name.startswith("m1143r6_c2_frozen_netlist_mapped_vcs_successor") or
                   p.name.startswith(".m1143r6_c2_frozen_netlist_mapped_vcs_successor"))
    require(names == [ATTEMPT.name, FAILURE.name], "M1143 namespace cardinality drift")
    require(not RESULT.exists() and not RESULT.is_symlink(), "canonical result exists")
    require(not list(RESULTS.glob(".m1143r6_c2_frozen_netlist_mapped_vcs_successor_work.*")),
            "unquarantined work remains")

    attempt = load(ATTEMPT / "attempt.json")
    failure = load(FAILURE / "failure.json")
    require(attempt["status"] == "M1143R6_SINGLE_ATTEMPT_CONSUMED__NO_RETRY",
            "attempt status drift")
    require(attempt["compile_attempts"] == 1 and attempt["dc_attempts"] == 0 and
            attempt["automatic_retry"] is False, "attempt accounting drift")
    require(failure["status"] == "FAILED_OR_INCOMPLETE_DO_NOT_CITE" and
            failure["phase"] == "FROZEN_NETLIST_VCS_COMPILE_ONCE",
            "failure phase/status drift")
    require(failure["attempt_consumed"] is True and failure["automatic_retry"] is False and
            failure["dc_rerun"] is False, "failure no-retry boundary drift")

    files = sorted(str(p.relative_to(FAILURE)) for p in FAILURE.rglob("*") if p.is_file())
    require(files == ["SHA256SUMS", "SHA256SUMS.seal.sha256", "failure.json",
                      "mapped_vcs/compile.log"], "failure file set proves more than compile-only")
    require(not (FAILURE / "mapped_vcs/simv").exists() and
            not (FAILURE / "mapped_vcs/case0.log").exists(), "simulation artifact exists")
    compile_log = (FAILURE / "mapped_vcs/compile.log").read_text(
        encoding="utf-8", errors="replace")
    require(compile_log.count("Cannot find license file.") == 1, "license failure token drift")
    require("LM_LICENSE_FILE is pointing to the right location." in compile_log,
            "license route diagnostic absent")
    require("Chronologic VCS (TM)" in compile_log and "CPU time:" in compile_log,
            "VCS invocation evidence absent")
    for forbidden in ("Error-[", "Syntax error", "Undefined module", "PASS_M1112", "M1112_FIRST_X"):
        require(forbidden not in compile_log, f"unexpected compile/simulation evidence: {forbidden}")

    source_text = SOURCE.read_text(encoding="utf-8")
    original_text = ORIGINAL.read_text(encoding="utf-8")
    source_audit = audit_source_text(source_text, original_text)
    routes = route_metadata()
    attacks = mutation_attacks(source_text, original_text)

    review = {
        "schema": "m1145r6_m1143r6_c2_license_environment_failure_hammer_r1_v1",
        "status": ("PASS_M1145R6_M1143R6_LICENSE_ENVIRONMENT_OMISSION_FAILURE__"
                   "AUTHOR_ADDITIVE_LICENSE_ROUTE_SUCCESSOR_SOURCE_ONLY"),
        "verdict": ("GO_AUTHOR_ADDITIVE_LICENSE_ROUTE_SUCCESSOR_SOURCE_ONLY__"
                    "REMOVE_HOME_OVERRIDE__NO_RETRY_NO_VCS_NO_DC"),
        "score": 100,
        "checks": checks,
        "attacks": attacks,
        "identity": {
            "hammer_sha256": sha256(Path(__file__)),
            "attempt_outer_seal_file_sha256": EXPECTED["attempt_outer"],
            "failure_outer_seal_file_sha256": EXPECTED["failure_outer"],
            "m1143_source_sha256": EXPECTED["source"],
            "m1129_original_engine_sha256": EXPECTED["original"],
            "author_outer_seal_file_sha256": EXPECTED["author_outer"],
            "m1144_outer_seal_file_sha256": EXPECTED["hammer_outer"],
            "docs359_sha256": EXPECTED["docs359"],
        },
        "namespace": {
            "exactly_one_consumed_attempt": True,
            "exactly_one_failure_quarantine": True,
            "canonical_result_absent": True,
            "unquarantined_work_absent": True,
            "automatic_retry": False,
            "dc_rerun": False,
        },
        "execution_boundary": {
            "vcs_invocations": 1,
            "vcs_compile_completed": False,
            "elaboration_or_netlist_semantic_error_observed": False,
            "simulation_invocations": 0,
            "simv_absent": True,
            "case0_log_absent": True,
            "mapped_functionality": False,
        },
        "root_cause": {
            "classification": "LAUNCHER_ENVIRONMENT_OMISSION_BEFORE_COMPILATION",
            "compile_log_license_error_count": 1,
            "current_shell_routes_redacted": routes,
            "environment_comparison": source_audit,
            "home_policy_violation": "M1143 sets HOME=/tmp; successor must omit HOME entirely",
            "not_evidence_of_vcs_or_netlist_semantic_failure": True,
        },
        "successor_source_requirements": {
            "additive_new_namespace": True,
            "license_route_precedence": ["SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"],
            "route_preflight_presence_length_sha256_only": True,
            "route_value_must_never_be_logged_or_sealed": True,
            "selected_route_must_be_explicitly_inserted_into_child_environment": True,
            "home_key_forbidden": True,
            "caller_home_reuse_or_override_forbidden": True,
            "all_non_environment_frozen_inputs_unchanged": True,
        },
        "authorization": {
            "additive_license_route_successor_source_authoring": True,
            "direct_retry": False,
            "any_second_attempt_in_m1143_namespace": False,
            "vcs": False,
            "dc": False,
            "launch": False,
            "subject_modification": False,
        },
        "claim_boundary": {
            "read_only_failure_audit": True,
            "mapped_functionality": False,
            "cycles_speedup": False,
            "area_timing_power_energy": False,
            "paper_citable": False,
        },
    }
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    mechanical = {
        "status": "PASS", "checks": checks, "attacks": len(attacks),
        "real_vcs": 0, "real_dc": 0, "simulation_invocations_in_failed_attempt": 0,
        "license_values_recorded": 0,
    }
    (REVIEW / "mechanical_checks.json").write_text(
        json.dumps(mechanical, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REVIEW / "review.md").write_text(
        "# M1145R6 independent M1143R6 license-environment failure hammer\n\n"
        "Verdict: **PASS; the sole M1143R6 execution stopped at VCS license "
        "acquisition because its clean child environment omitted both Synopsys license "
        "route variables. This is a launcher-environment failure before RTL/netlist "
        "compilation, not evidence of a VCS or mapped-netlist semantic defect.**\n\n"
        "Exactly one sealed attempt and one sealed quarantine exist. The canonical "
        "result and unquarantined work are absent; automatic retry and DC rerun are "
        "false. The quarantine contains only `compile.log` plus failure/seal metadata: "
        "there is no `simv` and no `case0.log`, hence simulation invocations are zero. "
        "The attempt field reserving one case0 attempt must not be interpreted as an "
        "executed simulation.\n\n"
        "The current shell has both `SNPSLMD_LICENSE_FILE` and `LM_LICENSE_FILE`; this "
        "review records only presence, byte length, and SHA-256, never route values. "
        "Frozen M1129 checks SNPSLMD first, falls back to LM, runs `lmstat`, and copies "
        "the caller environment. M1143 instead builds a five-key environment that "
        "contains neither route. It also sets `HOME=/tmp`, which violates the current "
        "runtime constraint against repurposing HOME.\n\n"
        "Only additive successor **source authoring** is authorized. The successor must "
        "preflight and hash-bind the selected route (SNPSLMD first, LM fallback), insert "
        "that selected key/value into the child environment without logging or sealing "
        "the value, and omit HOME entirely. It must retain the frozen netlist, cell "
        "model, memory model, TB, command, no-SDF contract, and no-retry discipline. "
        "No direct retry, VCS/DC launch, mapped-functionality claim, or paper claim is "
        "authorized.\n",
        encoding="utf-8")
    (REVIEW / "READ_ONLY_FAILURE_AUDIT_NO_RETRY_NO_EDA.txt").write_text(
        "M1145R6 read-only failure audit; no retry, VCS, DC, or EDA was launched.\n",
        encoding="utf-8")
    (REVIEW / "RUN_COMPLETE.txt").write_text(
        "PASS_M1145R6_M1143R6_LICENSE_ENVIRONMENT_FAILURE_HAMMER\n", encoding="utf-8")
    manifest_sha, outer_sha = seal(REVIEW)
    print(json.dumps({"status": review["status"], "checks": checks,
                      "attacks": len(attacks), "manifest_sha256": manifest_sha,
                      "outer_seal_file_sha256": outer_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
