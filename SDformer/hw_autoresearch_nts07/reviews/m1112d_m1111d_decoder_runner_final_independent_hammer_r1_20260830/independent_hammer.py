#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1112D read-only/dry final hammer for the exact M1111D runner.

The runner main, production builder, canonical payload, and canonical attempt,
work, result, lock and quarantine namespaces are never invoked or created.
All mutation attacks operate under TemporaryDirectory.  The forged-publish
attack replaces rename_noreplace with a sentinel before the result boundary.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Callable


sys.dont_write_bytecode = True
HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "system_simulator/scripts/run_m1111d_m1105dr2_decoder_only_production_zero_arg.py"
CONTRACT = HW / "contracts/m1111d_m1105dr2_decoder_only_production_runner_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1111d_m1105dr2_decoder_only_production_runner_author_receipt_r1_20260830"
SOURCE = HW / "system_simulator/scripts/build_m1105dr2_decoder_only_address_timed_source.py"
SOURCE_CONTRACT = HW / "contracts/m1105dr2_decoder_only_address_timed_source_contract_r2_20260830.json"
SOURCE_RECEIPT = HW / "reviews/m1105dr2_decoder_source_trust_root_author_receipt_r1_20260830"
M1110D = HW / "reviews/m1110d_m1105dr2_decoder_source_contract_receipt_independent_hammer_r1_20260830"
MAPPER = HW / "system_simulator/scripts/map_m672_decoder_convtranspose_polyphase_workload_r3.py"
MAPPER_R2 = HW / "system_simulator/scripts/map_m670_decoder_convtranspose_polyphase_workload_r2.py"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = Path(__file__).with_name("mechanical_checks.json")

EXPECTED = {
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    "runner": "52407204479fa320f28f43bf7425abcf45acc7f126dfe83d076e7d9a8fe15f7a",
    "contract": "82bba9ed495f8b1d316ea02647e9f28868c4845da69f95723132f7921a8535f6",
    "contract_sidecar": "71f636ba43fc27321dda569e1133770cd77c9065974a2e974cd81b9b950df90e",
    "contract_outer": "7a94c5ce60291f76e8a460886486271868d628e523d96c8c63080dea6409b4dc",
    "author_review": "bccf2753b0c1e8bbc1c225fa43216cbb67a67516203e94150ad6f86077e1d4b6",
    "author_manifest": "56989e649ca0c2bc6cb6992e894eca22ac65c3972f3370b902788bd81d0924b3",
    "author_outer": "3e92ee453fe9e072b6b50f18845e720702333c888588a86afda8b8c751f25852",
    "source": "b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4",
    "source_contract": "cdbae0362d3ea093dbcb318aa2efad04e70677f8d984a9908cda44b0de3b80a4",
    "source_contract_sidecar": "37cdc8aa6b0c31103affa46f1aea80f073689540b16a40ea0eec68904a0fb4fe",
    "source_contract_outer": "4f95a616e16530bc30f94b68235247f7c7abe1b32956fc981412b3b1576193d3",
    "source_receipt_review": "16a628bb69d12b41a421d16dc1af5a9da0ae7593cfeeb9105a71ebc57bd9f952",
    "source_receipt_manifest": "e05ddc0c29c371e6a9b719a9e167b59ac2cecc33a51aac2959d0bd4b2a558cd8",
    "source_receipt_outer": "d16257e342be49f6e895bd1ca4b4c764eb6200da47bd27aa221abba7e6f6af25",
    "m1110_review": "feb6f554e15da36650d5d5d220d8bd75c2acb2f1c4a86dadb0d8359548285f7a",
    "m1110_manifest": "96c4450b26ecec7c1a8ea5516a4d4e301eac3a3e25798aae99e67e38ed7ba65a",
    "m1110_outer": "9caf64e422b4cb696a600b69415bd8265dc4694066fae7ec67a5f34019f39e23",
    "m1110_hammer": "6111ddaddbde977aaea7f278be196c7cba75c3f2f8df95f8b31c8aba3ed7d61c",
    "mapper": "989094c739ac12c448faf1e1388374bdabdb3bd5e4ebab6dd17aadf16ecf8254",
    "mapper_r2": "875b31ed1994729cc29321af0053fcea5586077aa468398d31eb4fe0fdb1596b",
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


def normalize_temporary(value: Any, prefix: str) -> Any:
    if isinstance(value, str):
        return value.replace(prefix, "$TMP")
    if isinstance(value, list):
        return [normalize_temporary(item, prefix) for item in value]
    if isinstance(value, dict):
        return {key: normalize_temporary(item, prefix) for key, item in value.items()}
    return value


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == expected,
            "regular identity " + str(path))


def verify_double(path: Path, identities: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identities[0]); verify_regular(side, identities[1])
    verify_regular(outer, identities[2])
    require(side.read_text().split() == [identities[0], path.name] and
            outer.read_text().split() == [identities[1], side.name],
            "double content")


def verify_flat_exact(directory: Path, identities: tuple[str, str, str],
                      status: str) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), "flat directory")
    review, manifest, outer = (directory / "review.json", directory / "SHA256SUMS",
                               directory / "SHA256SUMS.seal.sha256")
    verify_regular(review, identities[0]); verify_regular(manifest, identities[1])
    verify_regular(outer, identities[2])
    require(outer.read_text().split() == [identities[1], "SHA256SUMS"], "flat outer")
    listed = {}
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(maxsplit=1); relative = relative.lstrip("*")
        item = directory / relative
        require(relative not in listed and not Path(relative).is_absolute() and
                ".." not in Path(relative).parts, "unsafe flat member")
        verify_regular(item, digest); listed[relative] = digest
    actual = {item.relative_to(directory).as_posix() for item in directory.rglob("*")
              if (item.is_file() or item.is_symlink()) and
              item.relative_to(directory).as_posix() not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(set(listed) == actual, "flat exact coverage")
    require(strict_json(review).get("status") == status, "flat status")
    return {"members": len(actual), "exact_coverage": True, "regular_only": True}


def load_runner():
    verify_regular(RUNNER, EXPECTED["runner"])
    spec = importlib.util.spec_from_file_location("m1112d_frozen_m1111d", RUNNER)
    require(spec is not None and spec.loader is not None, "runner import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(module) -> dict[str, Any]:
    return {
        "attempt": module.ATTEMPT.exists() or module.ATTEMPT.is_symlink(),
        "result": module.RESULT.exists() or module.RESULT.is_symlink(),
        "lock": module.LOCK.exists() or module.LOCK.is_symlink(),
        "work": sorted(path.name for path in module.RESULT.parent.glob(
            module.WORK_PREFIX + "*")),
        "quarantine": sorted(path.name for path in module.RESULT.parent.glob(
            module.FAILURE_PREFIX + "*")),
    }


class PublishBoundaryReached(RuntimeError):
    pass


def main() -> None:
    require(sys.flags.isolated == 1 and sys.flags.no_user_site == 1,
            "hammer requires isolated Python")
    verify_regular(PYTHON, EXPECTED["python"])
    verify_regular(SOURCE, EXPECTED["source"])
    verify_regular(MAPPER, EXPECTED["mapper"])
    verify_regular(MAPPER_R2, EXPECTED["mapper_r2"])
    verify_regular(DOCS359, EXPECTED["docs359"])
    verify_double(CONTRACT, (EXPECTED["contract"], EXPECTED["contract_sidecar"],
                             EXPECTED["contract_outer"]))
    verify_double(SOURCE_CONTRACT, (EXPECTED["source_contract"],
                                    EXPECTED["source_contract_sidecar"],
                                    EXPECTED["source_contract_outer"]))
    flat = {
        "author": verify_flat_exact(AUTHOR, (EXPECTED["author_review"],
            EXPECTED["author_manifest"], EXPECTED["author_outer"]),
            "PASS_M1111D_DECODER_RUNNER_AUTHOR_SOURCE__DIFFERENT_AUTHOR_FINAL_HAMMER_REQUIRED"),
        "source_receipt": verify_flat_exact(SOURCE_RECEIPT,
            (EXPECTED["source_receipt_review"], EXPECTED["source_receipt_manifest"],
             EXPECTED["source_receipt_outer"]),
            "PASS_M1105DR2_FIXED_TRUST_SOURCE_AUTHOR_RECEIPT__INDEPENDENT_HAMMER_REQUIRED"),
        "m1110": verify_flat_exact(M1110D, (EXPECTED["m1110_review"],
            EXPECTED["m1110_manifest"], EXPECTED["m1110_outer"]),
            "PASS_M1110D_M1105DR2_FIXED_TRUST_SOURCE_HAMMER__RUNNER_AUTHORING_ONLY"),
    }
    verify_regular(M1110D / "independent_hammer.py", EXPECTED["m1110_hammer"])

    contract = strict_json(CONTRACT)
    require(contract["production_scope"]["calls"] == 120 and
            contract["production_scope"]["m700_external_input_allowed"] is False and
            contract["production_scope"]["final_checkpoint_rebind_required"] is True and
            contract["production_scope"]["d1"]["theta_word_uint32"] == 1065353139 and
            contract["production_scope"]["d1"]["theta_ieee754_le_hex"] == "b3ff7f3f" and
            contract["production_scope"]["d1"]["weight_folding_allowed"] is False and
            contract["claim_boundary"]["system_speedup_admitted"] is False and
            contract["claim_boundary"]["paper_citable_performance"] is False,
            "contract policy drift")

    runner = load_runner()
    before = snapshot(runner)
    require(before == {"attempt": False, "result": False, "lock": False,
                       "work": [], "quarantine": []}, "canonical namespace not fresh")
    authority = runner.validate_authorities(require_fresh=True)
    oracle = runner.source_static_self_test()
    require(authority["status"] == "PASS_M1111D_HARDCODED_AUTHORITIES_NO_PAYLOAD_NO_ATTEMPT" and
            oracle["status"] == "PASS_M1111D_RUNNER_SOURCE_STATIC_SELF_TEST__NO_PRODUCTION" and
            oracle["attempt_created"] is False and
            oracle["canonical_payload_opened"] is False and
            oracle["production_replay_executed"] is False,
            "safe runner self-test drift")

    source_text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    main_node = functions["main"]
    main_calls = [(node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id,
                   node.lineno) for node in ast.walk(main_node) if isinstance(node, ast.Call)
                  and isinstance(node.func, (ast.Attribute, ast.Name))]
    names = ("validate_authorities", "sanitize_environment", "resource_gate",
             "acquire_lock", "consume_attempt", "execute_production",
             "publish_result", "quarantine_work", "release_lock")
    call_lines = {name: sorted(line for called, line in main_calls if called == name)
                  for name in names}
    require(all(len(call_lines[name]) == 1 for name in names), "main multiplicity")
    require(call_lines["validate_authorities"][0] < call_lines["sanitize_environment"][0] <
            call_lines["resource_gate"][0] < call_lines["acquire_lock"][0] <
            call_lines["consume_attempt"][0] < call_lines["execute_production"][0] <
            call_lines["publish_result"][0], "main order")
    require(not any(isinstance(node, ast.While) for node in ast.walk(main_node)) and
            not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                    node.func.id == "main" for node in ast.walk(main_node)),
            "retry recursion/loop")
    require("ATTEMPT.mkdir(mode=0o700)" in source_text and
            '"maximum_attempts": 1' in source_text and
            '"automatic_retry": False' in source_text and
            'canonical["external_baseline_rejection"]["m700_admitted"] is False' in source_text and
            '"ratios_or_speedups": None' in source_text,
            "attempt/M700/ratio literal drift")

    attacks: dict[str, Any] = {}
    saved_argv = list(sys.argv)
    sys.argv[:] = [str(RUNNER), "--attack"]
    attacks["extra_argv"] = rejected(runner.main, "extra argv")
    sys.argv[:] = saved_argv

    saved_environment = dict(os.environ)
    os.environ["PYTHONPATH"] = "/tmp/ATTACK"
    os.environ["M700_INPUT"] = "/tmp/ATTACK"
    os.environ["DECODER_CHECKPOINT"] = "ATTACK"
    os.environ["LD_PRELOAD"] = "/tmp/ATTACK.so"
    runner.sanitize_environment()
    sanitized = dict(os.environ)
    require(sanitized == {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                          "PATH": "/usr/bin:/bin", "TMPDIR": "/tmp",
                          "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            "environment sanitizer")
    os.environ.clear(); os.environ.update(saved_environment)

    with tempfile.TemporaryDirectory(prefix="m1112d_dry_") as temporary:
        root = Path(temporary)
        bad_contract = root / CONTRACT.name
        shutil.copy2(CONTRACT, bad_contract)
        bad_contract.write_bytes(bad_contract.read_bytes() + b"\n")
        shutil.copy2(Path(str(CONTRACT) + ".sha256"), Path(str(bad_contract) + ".sha256"))
        shutil.copy2(Path(str(CONTRACT) + ".sha256.seal.sha256"),
                     Path(str(bad_contract) + ".sha256.seal.sha256"))
        old_contract = runner.CONTRACT; runner.CONTRACT = bad_contract
        attacks["contract_byte_mutation"] = rejected(
            lambda: runner.validate_authorities(False), "contract bytes")
        runner.CONTRACT = old_contract

        source_link = root / "source-link.py"; source_link.symlink_to(SOURCE)
        old_source = runner.SOURCE; runner.SOURCE = source_link
        attacks["source_symlink"] = rejected(
            lambda: runner.validate_authorities(False), "source symlink")
        runner.SOURCE = old_source
        m1110_link = root / "m1110-link"; m1110_link.symlink_to(M1110D, target_is_directory=True)
        old_m1110 = runner.M1110D; runner.M1110D = m1110_link
        attacks["authority_directory_symlink"] = rejected(
            lambda: runner.validate_authorities(False), "authority directory symlink")
        runner.M1110D = old_m1110

        flat_copy = root / "m1110-flat-copy"
        shutil.copytree(M1110D, flat_copy)
        (flat_copy / "SHA256SUMS").unlink(); (flat_copy / "SHA256SUMS.seal.sha256").unlink()
        (flat_copy / "SHA256SUMS").symlink_to(M1110D / "SHA256SUMS")
        (flat_copy / "SHA256SUMS.seal.sha256").symlink_to(
            M1110D / "SHA256SUMS.seal.sha256")
        runner.M1110D = flat_copy
        accepted_flat_symlink = runner.validate_authorities(False)["status"]
        runner.M1110D = old_m1110
        require(accepted_flat_symlink ==
                "PASS_M1111D_HARDCODED_AUTHORITIES_NO_PAYLOAD_NO_ATTEMPT",
                "expected flat-root symlink gap changed")
        attacks["flat_manifest_outer_symlink"] = "ACCEPTED_BY_RUNNER_VERIFY_FLAT"

        saved_paths = (runner.RESULT, runner.ATTEMPT, runner.LOCK,
                       runner.WORK_PREFIX, runner.FAILURE_PREFIX)
        sandbox = root / "sandbox"; sandbox.mkdir()
        runner.RESULT = sandbox / "decoder_result"
        runner.ATTEMPT = sandbox / ".decoder_attempt"
        runner.LOCK = sandbox / ".decoder_lock"
        runner.WORK_PREFIX = ".decoder_work."
        runner.FAILURE_PREFIX = "decoder_result.failed."
        require(runner.namespace_fresh(), "fresh sandbox namespace")
        for label, path in (("attempt_collision", runner.ATTEMPT),
                            ("result_collision", runner.RESULT),
                            ("lock_collision", runner.LOCK),
                            ("work_collision", sandbox / ".decoder_work.stale"),
                            ("quarantine_collision_namespace",
                             sandbox / "decoder_result.failed.stale")):
            path.mkdir()
            require(not runner.namespace_fresh(), "namespace accepted " + label)
            attacks[label] = "REJECTED_BY_NAMESPACE_FRESHNESS"
            path.rmdir()
        broken = sandbox / "decoder_result"
        broken.symlink_to(sandbox / "absent")
        require(not runner.namespace_fresh(), "broken result symlink accepted")
        attacks["broken_result_symlink"] = "REJECTED_BY_NAMESPACE_FRESHNESS"
        broken.unlink()

        unsealed = sandbox / ".decoder_work.unsealed"; unsealed.mkdir()
        (unsealed / "partial.json").write_text("{}\n", encoding="utf-8")
        attacks["unsealed_publish"] = rejected(
            lambda: runner.publish_result(unsealed), "unsealed publish")
        require(not runner.RESULT.exists(), "unsealed publish made result")

        forged = sandbox / ".decoder_work.forged"; forged.mkdir()
        forged_payload = {
            "status": "PASS_M1111D_DECODER_ONLY_DIAGNOSTIC_RESULT__FINAL_RESULT_HAMMER_REQUIRED",
            "diagnostic": {"ratios_or_speedups": {"forged_speedup": 999.0}},
            "claim_boundary": {
                "system_speedup_admitted": False,
                "paper_ppa_ready": False,
                "speedup_admitted": True,
                "paper_citable_performance": True,
                "final_checkpoint_rebind_required": False,
            },
        }
        (forged / runner.PAYLOAD).write_text(
            json.dumps(forged_payload, sort_keys=True) + "\n", encoding="utf-8")
        (forged / "RUN_COMPLETE.txt").write_text("FORGED\n", encoding="utf-8")
        runner.atomic_seal(forged)
        old_rename = runner.rename_noreplace
        def stop_before_publish(_source, _destination):
            raise PublishBoundaryReached("forged payload reached no-replace publish boundary")
        runner.rename_noreplace = stop_before_publish
        reached = rejected(lambda: runner.publish_result(forged), "forged publish boundary")
        runner.rename_noreplace = old_rename
        require(reached.startswith("PublishBoundaryReached:") and
                not runner.RESULT.exists(), "forged gate did not reach expected boundary")
        attacks["forged_forbidden_claims_missing_schedule"] = {
            "outcome": "ACCEPTED_THROUGH_ALL_PUBLISH_VALIDATION_TO_RENAME_BOUNDARY",
            "ratio_nonnull": True,
            "speedup_admitted_true": True,
            "paper_citable_performance_true": True,
            "final_checkpoint_rebind_required_false": True,
            "call_schedule_file_absent": True,
            "sealed_payload_member_count": 2,
            "result_created": False,
        }

        existing_quarantine = sandbox / "decoder_result.failed.collision"
        existing_quarantine.mkdir()
        attacks["quarantine_collision"] = rejected(
            lambda: runner.quarantine_work(sandbox / ".decoder_work.absent",
                existing_quarantine, "DRY"), "quarantine collision")
        require(not Path(str(existing_quarantine) + ".stage").exists(),
                "quarantine collision made stage")
        runner.RESULT, runner.ATTEMPT, runner.LOCK, runner.WORK_PREFIX, \
            runner.FAILURE_PREFIX = saved_paths

        mutated_runner = root / RUNNER.name
        mutated_runner.write_bytes(RUNNER.read_bytes() + b"\n")
        require(sha(mutated_runner) != EXPECTED["runner"], "runner mutation ineffective")
        attacks["runner_byte_mutation"] = "REJECTED_BY_REQUIRED_EXTERNAL_EXACT_SHA_GATE"

    attacks = normalize_temporary(attacks, str(root))
    after = snapshot(runner)
    require(after == before, "hammer changed canonical namespace")
    result = {
        "schema": "m1112d_m1111d_decoder_runner_final_independent_hammer_checks_v1",
        "status": "STOP_M1112D_PUBLISH_GATE_ACCEPTS_FORBIDDEN_CLAIMS_AND_INCOMPLETE_FILESET",
        "identities": EXPECTED,
        "flat_authorities": flat,
        "safe_calls": {
            "validate_authorities": authority["status"],
            "source_static_self_test": oracle["status"],
            "runner_main_called": False,
            "build_canonical_called": False,
            "canonical_payload_opened": False,
            "execute_production_called": False,
            "canonical_attempt_created": False,
            "canonical_result_created": False,
        },
        "static_main_call_lines": call_lines,
        "maximum_attempts": 1,
        "automatic_retry": False,
        "environment_after_sanitize": sanitized,
        "attacks": attacks,
        "canonical_namespace_before": before,
        "canonical_namespace_after": after,
        "p0": {
            "publish_gate_required_fields_checked": [
                "status", "system_speedup_admitted", "paper_ppa_ready"
            ],
            "publish_gate_missing_checks": [
                "exact three-file payload set", "120 strict call rows and ordinals",
                "call schedule and stream digests", "ratios_or_speedups is null",
                "speedup_admitted is false", "paper_citable_performance is false",
                "final_checkpoint_rebind_required is true", "full claim projection"
            ],
            "production_authorized": False,
        },
        "p1": {
            "runner_verify_flat_accepts_manifest_outer_symlinks": True,
            "new_runner_must_require_regular_root_files_and_exact_coverage": True,
        },
        "docs359_sha256": sha(DOCS359),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")


if __name__ == "__main__":
    main()
