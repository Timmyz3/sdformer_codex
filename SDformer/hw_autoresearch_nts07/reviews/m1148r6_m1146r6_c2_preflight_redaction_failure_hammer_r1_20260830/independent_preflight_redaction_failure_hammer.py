#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only M1146R6 real-lmstat pre-attempt failure audit.

The raw lmstat output is captured only in process memory.  This hammer writes
only its return code, byte count, and a boolean saying whether the selected
route occurred; it never returns, logs, serializes, or seals raw output or a
route value.  It launches neither VCS nor DC and creates no attempt namespace.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


HW = Path("/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07")
RESULTS = HW / "results"
REVIEW = HW / "reviews/m1148r6_m1146r6_c2_preflight_redaction_failure_hammer_r1_20260830"
SOURCE = HW / "dc_handoff/scripts/run_m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_source_r1.py"
CONTRACT = HW / "contracts/m1146r6_c2_additive_license_route_successor_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1146r6_c2_additive_license_route_successor_author_receipt_r1_20260830"
HAMMER = HW / "reviews/m1147r6_m1146r6_c2_license_route_final_source_hammer_r1_20260830"
AUTHORITY = HW / "reviews/m1145r6_m1143r6_c2_license_environment_failure_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
RESULT = RESULTS / "m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_r1_20260830"
ATTEMPT = RESULTS / ".m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_attempt_consumed"
WORK_PREFIX = ".m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_work."
FAILURE_PREFIX = RESULT.name + ".failed_or_incomplete."

EXPECTED = {
    "source": "69c30ccfdc884aecca407b6b86b66bc82f97dd02abdb353561daa083934d591c",
    "contract_outer": "b28d565b1c1ef7b3c79724bf06bc4be202e55010f88b7b0274adb068a9fb82e6",
    "author_outer": "513813aa1915e72af18c1b059cfae77947c9ece37fc8699582cc202c489b98d1",
    "hammer_outer": "64007fe4ec37a26c54c197b80ae9f9565e8272c06fecfe3510c24aeb7c74d7e9",
    "authority_outer": "9edbc8abd3b47bbec576b35d00602cba5abca01cbee320081f954cca9e820148",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "SNPSLMD_LICENSE_FILE": {"present": True, "length": 18,
        "sha256": "5662b8aedce52c58e4cf6ac8e7b888f6ecf16cef8fb7998d724582d030af1d7b"},
    "LM_LICENSE_FILE": {"present": True, "length": 26,
        "sha256": "26bf3db829e694cbc7c0886311b8c5fedf2887e143313efcc81854e57f718adc"},
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


def verify_sealed_tree(directory: Path, expected_outer: str) -> dict:
    require(directory.is_dir() and not directory.is_symlink(), "sealed tree absent/symlink")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and outer.is_file() and not manifest.is_symlink() and
            not outer.is_symlink(), "seal files absent/symlink")
    require(sha256(outer) == expected_outer, "outer seal-file identity drift")
    outer_fields = outer.read_text(encoding="utf-8").strip().split()
    require(len(outer_fields) == 2 and outer_fields[1] == "SHA256SUMS" and
            sha256(manifest) == outer_fields[0], "outer/manifest mismatch")
    listed = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(None, 1); rel = rel.strip(); member = directory / rel
        require(member.is_file() and not member.is_symlink() and sha256(member) == digest,
                "sealed member drift: " + rel)
        listed.append(rel)
    actual = sorted(str(p.relative_to(directory)) for p in directory.rglob("*")
                    if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    require(sorted(listed) == actual, "tree manifest coverage drift")
    return load(directory / "review.json")


def verify_contract() -> dict:
    digest_file = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    require(CONTRACT.is_file() and digest_file.is_file() and outer.is_file(), "contract chain absent")
    require(sha256(outer) == EXPECTED["contract_outer"], "contract outer identity drift")
    fields = digest_file.read_text(encoding="utf-8").strip().split()
    require(len(fields) == 2 and fields[1] == CONTRACT.name and sha256(CONTRACT) == fields[0],
            "contract digest drift")
    outer_fields = outer.read_text(encoding="utf-8").strip().split()
    require(len(outer_fields) == 2 and outer_fields[1] == digest_file.name and
            sha256(digest_file) == outer_fields[0], "contract outer chain mismatch")
    return load(CONTRACT)


def namespace_snapshot() -> dict:
    return {
        "attempt": ATTEMPT.exists() or ATTEMPT.is_symlink(),
        "result": RESULT.exists() or RESULT.is_symlink(),
        "work_count": len(list(RESULTS.glob(WORK_PREFIX + "*"))),
        "failure_count": len(list(RESULTS.glob(FAILURE_PREFIX + "*"))),
    }


def route_metadata() -> dict:
    result = {}
    for name in ("SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"):
        value = os.environ.get(name)
        meta = {"present": bool(value), "length": len(value) if value else 0,
                "sha256": hashlib.sha256(value.encode()).hexdigest() if value else None}
        require(meta == EXPECTED[name], name + " route identity drift")
        result[name] = meta
    return result


def load_source_module():
    spec = importlib.util.spec_from_file_location("m1148_read_only_subject", SOURCE)
    require(spec is not None and spec.loader is not None, "source import spec absent")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def source_audit(text: str) -> dict:
    required = (
        'output, _ = process.communicate(timeout=30)',
        'require(value.encode() not in output, "lmstat output echoed secret route")',
        'return process.returncode == 0',
        'require("HOME" not in result',
        'key, value, route = _select_license_route(dict(os.environ))',
        'require(_run_lmstat(key, value, child), "selected license route lmstat unavailable")',
    )
    for token in required:
        require(token in text, "source evidence absent: " + token)
    execute = text[text.index("def _future_execute_once"):text.index("def main")]
    preflight_offset = execute.index("source_preflight(True)")
    attempt_offset = execute.index('ATTEMPT.mkdir(mode=0o700)')
    require(preflight_offset < attempt_offset, "preflight/attempt order drift")
    require('"HOME":' not in text[text.index("def _child_environment"):text.index("def _run_lmstat")],
            "HOME literal inserted into child environment")
    return {"raw_lmstat_captured_in_memory": True,
            "raw_route_presence_currently_treated_as_failure": True,
            "preflight_precedes_attempt_creation": True,
            "home_key_forbidden_and_absent": True}


def mutation_attacks(text: str) -> dict:
    mutations = {
        "remove_raw_capture": text.replace('output, _ = process.communicate(timeout=30)',
                                           'ignored, _ = process.communicate(timeout=30)', 1),
        "remove_overstrict_secret_test": text.replace(
            'require(value.encode() not in output, "lmstat output echoed secret route")',
            'pass  # repaired', 1),
        "remove_rc_test": text.replace('return process.returncode == 0', 'return True', 1),
        "add_home": text.replace('"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1", key: value}',
                                  '"VCS_HOME": "/opt/synopsys/vcs/V-2023.12-SP1", "HOME": "/tmp", key: value}', 1),
        "move_attempt_before_preflight": text.replace(
            'preflight, key, secret, child = source_preflight(True)',
            'ATTEMPT.mkdir(mode=0o700)\n    preflight, key, secret, child = source_preflight(True)', 1),
        "remove_license_selection": text.replace(
            'key, value, route = _select_license_route(dict(os.environ))',
            'key, value, route = None, None, None', 1),
    }
    caught = {}
    for name, mutant in mutations.items():
        try:
            source_audit(mutant)
        except (AuditFailure, ValueError) as exc:
            caught[name] = type(exc).__name__ + ": " + str(exc)
        else:
            raise AuditFailure("mutation survived: " + name)
    require(len(caught) == 6, "attack count drift")
    return caught


def seal(directory: Path) -> tuple[str, str]:
    members = sorted(p for p in directory.iterdir()
                     if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    (directory / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in members), encoding="utf-8")
    manifest_sha = sha256(directory / "SHA256SUMS")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        f"{manifest_sha}  SHA256SUMS\n", encoding="utf-8")
    return manifest_sha, sha256(directory / "SHA256SUMS.seal.sha256")


def main() -> None:
    require(sha256(SOURCE) == EXPECTED["source"], "source identity drift")
    require(sha256(DOCS359) == EXPECTED["docs359"], "docs/359 drift")
    contract = verify_contract()
    author = verify_sealed_tree(AUTHOR, EXPECTED["author_outer"])
    hammer = verify_sealed_tree(HAMMER, EXPECTED["hammer_outer"])
    authority = verify_sealed_tree(AUTHORITY, EXPECTED["authority_outer"])
    require(contract["status"] ==
            "SOURCE_ONLY__MOCK_AUTHOR_RECEIPT_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_VCS_NO_LAUNCH" and
            author["status"] ==
            "PASS_M1146R6_SOURCE_CONTRACT_CONTROLLED_MOCK__DIFFERENT_AUTHOR_HAMMER_REQUIRED" and
            hammer["status"] ==
            "PASS_M1147R6_FINAL_SOURCE_HAMMER__ROOT_EXTERNAL_PREFLIGHT_THEN_ONE_EXACT_LICENSE_ROUTED_MAPPED_VCS_EXECUTION_ONLY" and
            authority["status"] ==
            "PASS_M1145R6_M1143R6_LICENSE_ENVIRONMENT_OMISSION_FAILURE__AUTHOR_ADDITIVE_LICENSE_ROUTE_SUCCESSOR_SOURCE_ONLY",
            "authority status drift")

    before = namespace_snapshot()
    require(before == {"attempt": False, "result": False, "work_count": 0,
                       "failure_count": 0}, "M1146 namespace not fresh before probe")
    routes = route_metadata()
    text = SOURCE.read_text(encoding="utf-8")
    static = source_audit(text)
    attacks = mutation_attacks(text)

    module = load_source_module()
    key, value, public_route = module._select_license_route(dict(os.environ))
    child = module._child_environment(key, value)
    require(key == "SNPSLMD_LICENSE_FILE", "route precedence drift")
    require("HOME" not in child, "HOME present in child")
    probe = subprocess.run([str(module.LMUTIL), "lmstat", "-c", value],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env=child, timeout=30, check=False)
    raw_contains_route = value.encode() in probe.stdout
    raw_bytes = len(probe.stdout)
    require(probe.returncode == 0, "real lmstat return code nonzero")
    require(raw_contains_route, "real lmstat did not reproduce raw route echo")
    del probe

    try:
        module.source_preflight(True)
    except module.Failure as exc:
        failure_text = str(exc)
    else:
        raise AuditFailure("M1146 source_preflight unexpectedly passed")
    require(failure_text == "lmstat output echoed secret route", "preflight failure drift")
    after = namespace_snapshot()
    require(after == before, "preflight created attempt/result/work/failure")

    review = {
        "schema": "m1148r6_m1146r6_c2_preflight_redaction_failure_hammer_r1_v1",
        "status": ("PASS_M1148R6_M1146R6_REAL_LMSTAT_PREFLIGHT_FALSE_NEGATIVE__"
                   "AUTHOR_ADDITIVE_PREFLIGHT_REDACTION_REPAIR_SOURCE_ONLY"),
        "verdict": ("GO_AUTHOR_ADDITIVE_PREFLIGHT_REDACTION_REPAIR_SOURCE_ONLY__"
                    "NO_ATTEMPT_NO_LAUNCH_NO_VCS_NO_DC"),
        "score": 100, "checks": checks, "attacks": attacks,
        "identity": {"hammer_sha256": sha256(Path(__file__)),
                     "m1146_source_sha256": EXPECTED["source"],
                     "contract_outer_seal_file_sha256": EXPECTED["contract_outer"],
                     "author_outer_seal_file_sha256": EXPECTED["author_outer"],
                     "m1147_outer_seal_file_sha256": EXPECTED["hammer_outer"],
                     "m1145_outer_seal_file_sha256": EXPECTED["authority_outer"],
                     "docs359_sha256": EXPECTED["docs359"]},
        "real_preflight_probe": {
            "selected_variable": key,
            "selected_route_public_identity": public_route,
            "lmstat_returncode": 0,
            "raw_output_bytes": raw_bytes,
            "raw_output_contains_selected_route": True,
            "raw_output_returned": False,
            "raw_output_logged": False,
            "raw_output_serialized_or_sealed": False,
            "preflight_outcome": "Failure: lmstat output echoed secret route",
        },
        "namespace": {"before": before, "after": after, "fully_fresh": True,
                      "attempts": 0, "results": 0, "failures": 0,
                      "vcs": 0, "dc": 0},
        "root_cause": {
            "classification": "PREFLIGHT_SECRET_ECHO_FALSE_NEGATIVE_BEFORE_ATTEMPT",
            "static_source": static,
            "current_shell_routes_redacted": routes,
            "lmstat_success": True,
            "license_route_is_valid": True,
            "m1146_rejects_valid_route_because_tool_echoes_it": True,
            "home_policy_compliant_in_m1146": True,
        },
        "repair_requirements": {
            "additive_new_source_and_namespace": True,
            "raw_output_may_exist_transiently_in_memory": True,
            "raw_output_must_not_be_returned": True,
            "raw_output_must_not_be_written_to_log_json_or_seal": True,
            "diagnostics_must_be_redacted_before_persistence": True,
            "preferred_minimal_preflight": "discard raw output and accept only returncode zero",
            "route_public_identity_only": ["selected_variable", "present", "byte_length", "sha256"],
            "home_key_forbidden": True,
            "caller_home_reuse_or_override_forbidden": True,
            "all_frozen_non_preflight_inputs_unchanged": True,
        },
        "authorization": {"additive_preflight_redaction_repair_source_authoring": True,
                          "attempt": False, "launch": False, "vcs": False,
                          "dc": False, "automatic_retry": False,
                          "subject_modification": False},
        "claim_boundary": {"read_only_pre_attempt_audit": True,
                           "mapped_functionality": False, "cycles_speedup": False,
                           "area_timing_power_energy": False, "paper_citable": False},
    }
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    (REVIEW / "mechanical_checks.json").write_text(json.dumps({
        "status": "PASS", "checks": checks, "attacks": len(attacks),
        "real_lmstat_read_only_probes": 2, "attempts": 0, "real_vcs": 0,
        "real_dc": 0, "raw_license_values_persisted": 0,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REVIEW / "review.md").write_text(
        "# M1148R6 independent M1146R6 pre-attempt redaction hammer\n\n"
        "Verdict: **PASS; M1146R6 rejects a valid license route before consuming an "
        "attempt because successful `lmstat` output legitimately echoes the selected "
        "route. This is a preflight false negative.**\n\n"
        "Independent real probing returned `lmstat` rc=0 and confirmed that its raw "
        "stdout contains the selected SNPSLMD route. No raw bytes or route value are "
        "returned, logged, serialized, or sealed; only rc, byte count, route-presence "
        "boolean, and public route identity (variable, presence, length, SHA-256) are "
        "recorded. M1146 `source_preflight(True)` fails with the exact safe diagnostic "
        "`lmstat output echoed secret route`. Before and after probing, attempt, result, "
        "work, and failure counts are all zero; VCS/DC counts are zero.\n\n"
        "Raw tool output may transiently exist in memory. The safety boundary is that "
        "it must never be returned or persisted. The minimal repair should discard raw "
        "output and decide only from return code; if diagnostics are retained, redact "
        "before any persistence. M1146 correctly omits HOME, and every successor must "
        "continue to forbid HOME reuse or override.\n\n"
        "Only additive preflight-redaction repair **source authoring** is authorized. "
        "No attempt, launch, VCS, DC, retry, mapped-functionality claim, or paper claim "
        "is authorized.\n", encoding="utf-8")
    (REVIEW / "READ_ONLY_PREFLIGHT_AUDIT_NO_ATTEMPT_NO_EDA.txt").write_text(
        "M1148R6 read-only pre-attempt audit; no attempt, VCS, DC, or launch.\n",
        encoding="utf-8")
    (REVIEW / "RUN_COMPLETE.txt").write_text(
        "PASS_M1148R6_M1146R6_PREFLIGHT_REDACTION_FAILURE_HAMMER\n", encoding="utf-8")
    manifest, outer = seal(REVIEW)
    print(json.dumps({"status": review["status"], "checks": checks,
                      "attacks": len(attacks), "manifest_sha256": manifest,
                      "outer_seal_file_sha256": outer}, sort_keys=True))


if __name__ == "__main__":
    main()
