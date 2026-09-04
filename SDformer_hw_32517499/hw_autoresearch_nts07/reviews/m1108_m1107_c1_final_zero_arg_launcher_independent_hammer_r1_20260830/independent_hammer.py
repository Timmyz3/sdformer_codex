#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1108 independent, no-launch hammer for the M1107 zero-argument wrapper."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

sys.dont_write_bytecode = True
HW = Path(__file__).resolve().parents[2]
LAUNCHER = HW / "system_simulator/scripts/run_m1107_m1102_c1_work8_full_replay_zero_arg.py"
CONTRACT = HW / "contracts/m1107_m1104_m1102_c1_work8_zero_arg_launcher_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1107_m1104_m1102_c1_work8_zero_arg_launcher_author_receipt_r1_20260830"
M1104 = HW / "reviews/m1104_m1102_c1_source_atomic_independent_hammer_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = Path(__file__).with_name("mechanical_checks.json")

EXPECTED = {
    "launcher": "fa1929793c63a1b71fc25f674826fc970ee354de7cffd23e131ff584450d6c84",
    "contract": "aa76f099a05998dd7facff6e3b5c7daf76923f7c331030e99594d5fae2a31d9b",
    "contract_sidecar": "a482b4d455460f13bab559b4a3ce6a621fba1851956609d2aa52a7e751183cb1",
    "contract_outer_file": "a25ef4e4f3521fed7c364626f290083e8f02b0ba127795edd129baae9ea3b25c",
    "author_review": "1b0448f6170edff6f34cb653a529fa57abe72e746dd55be9afd1293bd20e82d0",
    "author_manifest": "1eab92fb835c2554d617b64072249241463854903825d22cd62b9c900fa189d5",
    "author_outer_file": "9a6749e754cb4a27f59b3cc7fe5ad4b324630077057a93bc6124769ad168aad1",
    "m1104_review": "341026dc3c28bbea421bf29c1281f0aadfa58ce2cd2a59af85e6ef8fd0ceb89f",
    "m1104_manifest": "f9947c686b98c062576b6af2207e3e0ed152b0278e44ee4393ba27e0e157ff61",
    "m1104_outer_file": "a3c28bb2e7c5040f83199dba4e70eefa46e86dc95a06eb5709b3be20a4bed237",
    "m1104_hammer": "94bc3b3a0186b0f5ccf8416b8292e1dba3204fc5937c29c07e4f92e566740013",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON " + token)))


def rejected(action: Callable[[], Any], label: str) -> str:
    try:
        action()
    except BaseException as error:
        return type(error).__name__ + ": " + str(error)
    raise RuntimeError("attack unexpectedly passed: " + label)


def verify_flat(directory: Path, review_sha: str, manifest_sha: str,
                outer_file_sha: str, status: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(), "flat directory shape")
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require((sha(review), sha(manifest), sha(outer)) ==
            (review_sha, manifest_sha, outer_file_sha), "flat root identity")
    require(outer.read_text(encoding="utf-8").split() ==
            [manifest_sha, "SHA256SUMS"], "flat outer content")
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        item = directory / relative
        require(relative not in seen and not Path(relative).is_absolute() and
                ".." not in Path(relative).parts and item.is_file() and
                not item.is_symlink() and sha(item) == digest, "flat member drift")
        seen.add(relative)
    require(strict_json(review).get("status") == status, "flat status drift")


def load_launcher():
    require(sha(LAUNCHER) == EXPECTED["launcher"], "launcher SHA")
    spec = importlib.util.spec_from_file_location("m1108_frozen_m1107", LAUNCHER)
    require(spec is not None and spec.loader is not None, "launcher import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def namespace_snapshot(module) -> dict[str, Any]:
    atomic = module.M1102
    return {
        "attempt_exists_or_symlink": atomic.ATTEMPT.exists() or atomic.ATTEMPT.is_symlink(),
        "result_exists_or_symlink": atomic.RESULT.exists() or atomic.RESULT.is_symlink(),
        "lock_exists_or_symlink": atomic.LOCK.exists() or atomic.LOCK.is_symlink(),
        "work_entries": sorted(path.name for path in atomic.RESULT.parent.glob(atomic.WORK_PREFIX + "*")),
        "failure_entries": sorted(path.name for path in atomic.RESULT.parent.glob(atomic.FAILURE_PREFIX + "*")),
    }


def main() -> None:
    require(sys.flags.isolated == 1 and sys.flags.no_user_site == 1,
            "hammer itself requires isolated Python")
    require(sha(CONTRACT) == EXPECTED["contract"] and
            sha(Path(str(CONTRACT) + ".sha256")) == EXPECTED["contract_sidecar"] and
            sha(Path(str(CONTRACT) + ".sha256.seal.sha256")) ==
                EXPECTED["contract_outer_file"] and
            sha(DOCS359) == EXPECTED["docs359"], "top-level identity drift")
    require(Path(str(CONTRACT) + ".sha256.seal.sha256").read_text().split() ==
            [EXPECTED["contract_sidecar"], Path(str(CONTRACT) + ".sha256").name],
            "contract outer content drift")
    verify_flat(AUTHOR, EXPECTED["author_review"], EXPECTED["author_manifest"],
                EXPECTED["author_outer_file"],
                "PASS_M1107_ZERO_ARG_LAUNCHER_AUTHOR_SOURCE__M1108_FINAL_HAMMER_REQUIRED")
    verify_flat(M1104, EXPECTED["m1104_review"], EXPECTED["m1104_manifest"],
                EXPECTED["m1104_outer_file"],
                "PASS_M1104_M1102_SOURCE_ATOMIC_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY")
    require(sha(M1104 / "independent_hammer.py") == EXPECTED["m1104_hammer"],
            "M1104 hammer pin drift")

    launcher = load_launcher()
    before = namespace_snapshot(launcher)
    require(before == {"attempt_exists_or_symlink": False,
                       "result_exists_or_symlink": False,
                       "lock_exists_or_symlink": False,
                       "work_entries": [], "failure_entries": []},
            "production namespace not fresh before hammer")
    authority = launcher.validate_hardcoded_authorities(enforce_runtime=False)
    oracle = launcher.source_static_self_test()
    require(authority["status"] == "PASS_M1107_HARDCODED_AUTHORITIES_NO_ATTEMPT" and
            oracle["status"] == "PASS_M1107_ZERO_ARG_LAUNCHER_SOURCE_SELF_TEST__NO_ATTEMPT" and
            oracle["attempt_created"] is False and
            oracle["full_replay_executed"] is False, "safe source gate drift")

    attacks: dict[str, str] = {}
    saved_argv = list(sys.argv)
    sys.argv[:] = [str(LAUNCHER), "--inject"]
    attacks["argv"] = rejected(
        lambda: launcher.validate_hardcoded_authorities(enforce_runtime=True), "argv")
    sys.argv[:] = [str(LAUNCHER)]
    require(launcher.validate_hardcoded_authorities(enforce_runtime=True)["status"] ==
            "PASS_M1107_HARDCODED_AUTHORITIES_NO_ATTEMPT", "runtime identity gate")
    sys.argv[:] = saved_argv

    source_text = LAUNCHER.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    main_node = next(node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                     node.name == "main")
    main_calls = [(node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id,
                   node.lineno)
                  for node in ast.walk(main_node) if isinstance(node, ast.Call) and
                  isinstance(node.func, (ast.Attribute, ast.Name))]
    call_lines = {name: [line for called, line in main_calls if called == name]
                  for name in ("validate_hardcoded_authorities", "namespace_freshness",
                               "sanitize_environment", "acquire_lock", "consume_attempt",
                               "execute_full", "publish_result", "quarantine_work",
                               "release_lock")}
    require(len(call_lines["consume_attempt"]) == len(call_lines["execute_full"]) ==
            len(call_lines["publish_result"]) == len(call_lines["quarantine_work"]) == 1,
            "production call multiplicity")
    require(call_lines["validate_hardcoded_authorities"][0] <
            call_lines["namespace_freshness"][0] < call_lines["sanitize_environment"][0] <
            call_lines["acquire_lock"][0] < call_lines["consume_attempt"][0] <
            call_lines["execute_full"][0] < call_lines["publish_result"][0],
            "production call order")
    require("while " not in source_text and "automatic_retry\": False" in source_text and
            "if attempt_consumed:" in source_text and "finally:" in source_text,
            "no-retry/failure structure drift")

    saved_environment = dict(os.environ)
    os.environ["PYTHONPATH"] = "/tmp/ATTACK"
    os.environ["M1102_AUTHORITY"] = "ATTACK"
    os.environ["LD_PRELOAD"] = "/tmp/ATTACK.so"
    launcher.sanitize_environment()
    require(dict(os.environ) == {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                                 "PATH": "/usr/bin:/bin", "TMPDIR": "/tmp",
                                 "PYTHONNOUSERSITE": "1",
                                 "PYTHONDONTWRITEBYTECODE": "1"},
            "environment sanitizer did not erase caller state")
    os.environ.clear(); os.environ.update(saved_environment)

    with tempfile.TemporaryDirectory(prefix="m1108_hammer_") as temporary:
        root = Path(temporary)
        wrong_python = root / "python3.10"
        wrong_python.write_bytes(launcher.PYTHON.read_bytes() + b"\n")
        old_python = launcher.PYTHON
        launcher.PYTHON = wrong_python
        attacks["python_identity"] = rejected(
            lambda: launcher.validate_hardcoded_authorities(enforce_runtime=True),
            "python identity")
        launcher.PYTHON = old_python

        mutated_contract = root / CONTRACT.name
        shutil.copy2(CONTRACT, mutated_contract)
        mutated_contract.write_bytes(mutated_contract.read_bytes() + b"\n")
        shutil.copy2(Path(str(CONTRACT) + ".sha256"), Path(str(mutated_contract) + ".sha256"))
        shutil.copy2(Path(str(CONTRACT) + ".sha256.seal.sha256"),
                     Path(str(mutated_contract) + ".sha256.seal.sha256"))
        old_contract = launcher.CONTRACT
        launcher.CONTRACT = mutated_contract
        attacks["contract_byte_mutation"] = rejected(
            lambda: launcher.validate_hardcoded_authorities(False), "contract mutation")
        launcher.CONTRACT = old_contract

        receipt_copy = root / "receipt"
        shutil.copytree(launcher.SOURCE_RECEIPT, receipt_copy)
        (receipt_copy / "review.json").write_bytes(
            (receipt_copy / "review.json").read_bytes() + b"\n")
        old_receipt = launcher.SOURCE_RECEIPT
        launcher.SOURCE_RECEIPT = receipt_copy
        attacks["source_receipt_byte_mutation"] = rejected(
            lambda: launcher.validate_hardcoded_authorities(False), "receipt mutation")
        launcher.SOURCE_RECEIPT = old_receipt

        author_copy = root / "author"
        shutil.copytree(AUTHOR, author_copy)
        (author_copy / "review.json").write_bytes(
            (author_copy / "review.json").read_bytes() + b"\n")
        attacks["author_receipt_byte_mutation"] = rejected(
            lambda: verify_flat(author_copy, EXPECTED["author_review"],
                                EXPECTED["author_manifest"],
                                EXPECTED["author_outer_file"],
                                "PASS_M1107_ZERO_ARG_LAUNCHER_AUTHOR_SOURCE__M1108_FINAL_HAMMER_REQUIRED"),
            "author receipt mutation")

        source_link = root / "source-link.py"
        source_link.symlink_to(launcher.SOURCE)
        old_source = launcher.SOURCE
        launcher.SOURCE = source_link
        attacks["authority_file_symlink"] = rejected(
            lambda: launcher.validate_hardcoded_authorities(False), "file symlink")
        launcher.SOURCE = old_source
        m1104_link = root / "m1104-link"
        m1104_link.symlink_to(launcher.M1104, target_is_directory=True)
        old_m1104 = launcher.M1104
        launcher.M1104 = m1104_link
        attacks["authority_directory_symlink"] = rejected(
            lambda: launcher.validate_hardcoded_authorities(False), "directory symlink")
        launcher.M1104 = old_m1104

        mutated_launcher = root / LAUNCHER.name
        mutated_launcher.write_bytes(LAUNCHER.read_bytes() + b"\n")
        old_file = launcher.__file__
        launcher.__file__ = str(mutated_launcher)
        self_derived = launcher.hardcoded_authority()
        launcher.M1102._validate_launch_authority(self_derived)
        require(self_derived["launch_wrapper_sha256"] == sha(mutated_launcher) and
                self_derived["launch_wrapper_sha256"] != EXPECTED["launcher"],
                "launcher self-hash boundary not exposed")
        launcher.__file__ = old_file
        attacks["launcher_byte_mutation_requires_external_tuple"] = (
            "INTERNAL_SELF_HASH_ACCEPTS__EXTERNAL_PREEXEC_SHA_REQUIRED"
        )

        atomic = launcher.M1102
        saved_paths = (atomic.RESULT, atomic.ATTEMPT, atomic.LOCK,
                       atomic.WORK_PREFIX, atomic.FAILURE_PREFIX)
        sandbox = root / "results"
        sandbox.mkdir()
        atomic.RESULT = sandbox / "m1102_result"
        atomic.ATTEMPT = sandbox / ".m1102_attempt"
        atomic.LOCK = sandbox / ".m1102_lock"
        atomic.WORK_PREFIX = ".m1102_work."
        atomic.FAILURE_PREFIX = "m1102_result.failed."
        legacy = sandbox / ".m1095_legacy_attempt_consumed"
        legacy.mkdir()
        require(launcher.namespace_freshness()["status"] ==
                "PASS_M1107_NAMESPACE_RESOURCE_FRESHNESS",
                "legacy namespace incorrectly blocks additive namespace")
        for label, path in (
            ("new_attempt", atomic.ATTEMPT),
            ("new_result", atomic.RESULT),
            ("new_lock", atomic.LOCK),
            ("new_work", sandbox / (atomic.WORK_PREFIX + "stale")),
            ("new_quarantine", sandbox / (atomic.FAILURE_PREFIX + "stale")),
        ):
            path.mkdir()
            attacks[label] = rejected(launcher.namespace_freshness, label)
            path.rmdir()

        unsealed_work = sandbox / (atomic.WORK_PREFIX + "partial")
        unsealed_work.mkdir()
        (unsealed_work / "partial.json").write_text("{}\n", encoding="utf-8")
        attacks["atomic_unsealed_partial_publish"] = rejected(
            lambda: atomic.publish_result(unsealed_work), "unsealed partial publish")
        require(not atomic.RESULT.exists(), "partial publish created result")

        preexisting_quarantine = sandbox / (atomic.FAILURE_PREFIX + "collision")
        preexisting_quarantine.mkdir()
        attacks["quarantine_collision"] = rejected(
            lambda: atomic.quarantine_work(
                sandbox / (atomic.WORK_PREFIX + "absent"),
                preexisting_quarantine, 1, "TEST"), "quarantine collision")
        require(not Path(str(preexisting_quarantine) + ".stage").exists(),
                "quarantine collision created stage")
        atomic.RESULT, atomic.ATTEMPT, atomic.LOCK, atomic.WORK_PREFIX, \
            atomic.FAILURE_PREFIX = saved_paths

    generic = launcher.M1102.M1102.validate_work
    canonical = launcher.M1102.M1102.validate_canonical_work
    require([generic(value) for value in (0, 8, 9, 14, 16)] == [0, 8, 9, 14, 16],
            "generic domain legal values")
    for value in (True, False, -1, *range(1, 8)):
        attacks["generic_reject_" + repr(value)] = rejected(
            lambda value=value: generic(value), "generic domain")
    require([canonical(value) for value in (0, 8, 16)] == [0, 8, 16],
            "canonical legal values")
    for value in (9, 10, 11, 12, 13, 14):
        attacks["canonical_reject_" + str(value)] = rejected(
            lambda value=value: canonical(value), "canonical domain")

    after = namespace_snapshot(launcher)
    require(after == before, "hammer changed production namespace")
    result = {
        "schema": "m1108_m1107_c1_final_zero_arg_launcher_independent_hammer_checks_v1",
        "status": "PASS_M1108_FINAL_LAUNCHER_HAMMER__EXTERNAL_EXACT_SHA_GATE_REQUIRED",
        "identities": EXPECTED,
        "safe_calls": {
            "validate_hardcoded_authorities": authority["status"],
            "source_static_self_test": oracle["status"],
            "launcher_main_called": False,
            "exhaustive_preflight_called": False,
            "full_replay_called": False,
            "attempt_created": False,
            "result_created": False
        },
        "static_main_call_lines": call_lines,
        "attacks": attacks,
        "production_namespace_before": before,
        "production_namespace_after": after,
        "launcher_self_hash_boundary": {
            "atomic_library_checks_sha_shape_not_reviewed_value": True,
            "external_preexec_launcher_sha_required": True,
            "closed_by_m1108_external_tuple_and_exact_command": True
        },
        "maximum_attempts": 1,
        "automatic_retry": False,
        "docs359_sha256": sha(DOCS359)
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")


if __name__ == "__main__":
    main()
