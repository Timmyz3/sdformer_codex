#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1119r3 author static/mutation check; never launches engine or EDA."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
LAUNCHER = HW / "dc_handoff/scripts/run_m1112r3_c2_async_observation_authorized_launch_r1.py"
LAUNCHER_SHA = "fda1778b12caa9f365ab0ec4227fc587bb1b6bb784039c0963ebec642ebbe158"
ENGINE = HW / "dc_handoff/scripts/m1112r3_c2_async_observation_authorized_engine_source_r1.py"
ENGINE_SHA = "48616ebde16e07b132bbb2e686bd34a9f18270d0bc0693ab0ee956beb60f02be"
ENGINE_CONTRACT = HW / "contracts/m1112r3_c2_async_observation_shadow_source_contract_r1_20260830.json"
ENGINE_CONTRACT_ID = (
    "92117a56e50a946d674c82ce9fc084548b480df139e0a4e5a9b4aed391292bef",
    "cfe40a1d11bcdf77cd4ac33e149381b202c57cc8edc22cb1131559fba8e412fd",
    "ddda54a99c1638f39c828faf75775a7f5c0dae975ee26f7b251cbafa926906cf",
)
LAUNCH_RECEIPT = HW / "contracts/m1112r3_c2_async_observation_authorized_launch_receipt_r1_20260830.json"
LAUNCH_RECEIPT_ID = (
    "f24870b60b940d91f3280427995567931fe8e53e14db2bc58c68b61601b801f4",
    "6acb4b55754aadc0b3c1039e93476cfeea2b5ed5d58d92a635893514bfb225a0",
    "3fbb26e397bd5425d93ea8d364558582f94c1cd7d24d0ed6cfd2f85b828f14c3",
)
SOURCE_CONTRACT = HW / "contracts/m1119r3_m1112r3_c2_zero_arg_launcher_source_contract_r1_20260830.json"
SOURCE_CONTRACT_ID = (
    "c1437d77feb51072a30a697f2a36d099f7225b151f905f37314584941fbc5af0",
    "5ca822f0f0b880f7b8810230eda8b8c37b4890347a5f7287c61c4b5b38bf1cf0",
    "bd0322fa5eb0b369304667f256532b3c6108b646aee8aec240b91786c5ceab45",
)
M1117R3 = HW / "reviews/m1117r3_m1112r3_c2_async_observation_engine_hammer_r1_20260830"
M1117R3_OUTER = "41b4950ac4e1a175379e4d0ae34fd5335e339e320f716cd5e2b073dc9aa00d82"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
ATTEMPT = HW / "results/.m1112r3_c2_async_observation_dc_mapped_vcs_attempt_consumed"
RESULT = HW / "results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830"


checks: list[str] = []
attacks: list[str] = []


def require(value: bool, label: str) -> None:
    if not value:
        raise RuntimeError(label)
    checks.append(label)


def rejected(label: str, function, *args) -> None:
    try:
        function(*args)
    except Exception:
        attacks.append(label)
        return
    raise RuntimeError("attack accepted: " + label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " regular")
    require(sha(path) == expected, label + " sha")


def double(path: Path, identity: tuple[str, str, str], label: str) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, identity[0], label + " file")
    regular(side, identity[1], label + " side")
    regular(outer, identity[2], label + " outer")
    require(side.read_text(encoding="utf-8").split() ==
            [identity[0], path.relative_to(HW).as_posix()], label + " side content")
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], side.relative_to(HW).as_posix()], label + " outer content")


def definitions(path: Path, name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
               ast.ClassDef, ast.FunctionDef)
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": name}
    module = ast.Module(body=[node for node in tree.body if isinstance(node, allowed)],
                        type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def main() -> None:
    regular(LAUNCHER, LAUNCHER_SHA, "launcher")
    regular(ENGINE, ENGINE_SHA, "engine")
    regular(DOCS359, DOCS359_SHA, "docs359")
    double(ENGINE_CONTRACT, ENGINE_CONTRACT_ID, "engine contract")
    double(LAUNCH_RECEIPT, LAUNCH_RECEIPT_ID, "launch receipt")
    double(SOURCE_CONTRACT, SOURCE_CONTRACT_ID, "source contract")
    regular(M1117R3 / "SHA256SUMS.seal.sha256", M1117R3_OUTER, "M1117r3 outer")
    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink(), "attempt absent")
    require(not RESULT.exists() and not RESULT.is_symlink(), "result absent")
    require(not any((HW / "results").glob(
            ".m1112r3_c2_async_observation_dc_mapped_vcs_work.*")), "work absent")
    require(not any((HW / "results").glob(
            "m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*")),
            "failure absent")

    source = LAUNCHER.read_text(encoding="utf-8")
    engine_source = ENGINE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    require('ENGINE_SHA256 = "' + ENGINE_SHA + '"' in source, "literal engine pin")
    require('M1117R3_ID = (' in source and M1117R3_OUTER in source,
            "literal M1117r3 pin")
    require("len(sys.argv) == 1" in source, "zero argv source gate")
    require("set(os.environ) == ROOT_ENV_KEYS" in source, "exact env-i key gate")
    require("caller environment value is consulted" in source,
            "caller-blind environment declaration")
    require("tempfile.mkdtemp" in source and "private_home.chmod(0o700)" in source,
            "private HOME construction")
    require("shutil.rmtree(private_home)" in source, "private HOME exact cleanup")
    run_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
                 isinstance(node.func, ast.Attribute) and node.func.attr == "run"]
    require(len(run_calls) == 2, "one pgrep callsite plus one engine callsite")
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                     node.name == "main")
    main_text = ast.unparse(main_node)
    require(main_text.count("subprocess.run") == 1, "exactly one engine subprocess callsite")
    require("[str(PYTHON), '-I', str(ENGINE), '--authorized-launch']" in main_text,
            "exact engine argv")
    require("env=clean_child_environment(private_home)" in main_text,
            "constant child environment")
    require("namespace_resource_gate()" in main_text and
            main_text.index("namespace_resource_gate()") < main_text.index("tempfile.mkdtemp"),
            "fresh/resource/collision gate before private HOME/child")

    launch_receipt = json.loads(LAUNCH_RECEIPT.read_text(encoding="utf-8"))
    contract = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    expected_receipt_keys = {
        "schema", "status", "launcher_sha256", "engine_sha256",
        "engine_contract_sha256", "engine_contract_outer_seal_file_sha256",
        "engine_author_receipt_outer_seal_file_sha256",
        "m1116_outer_seal_file_sha256", "m1117r3_outer_seal_file_sha256",
        "arguments", "caller_selected_authority_allowed",
        "caller_environment_forwarded", "m1118r3_required", "launch_now",
        "attempt_now", "dc_now", "mapped_vcs_now", "maximum_attempts",
        "automatic_retry", "paper_citable",
    }
    require(set(launch_receipt) == expected_receipt_keys, "engine exact receipt schema")
    require(launch_receipt["launcher_sha256"] == LAUNCHER_SHA and
            launch_receipt["engine_sha256"] == ENGINE_SHA and
            launch_receipt["m1117r3_outer_seal_file_sha256"] == M1117R3_OUTER,
            "launch receipt identity pins")
    require(all(launch_receipt[key] is False for key in
                ("caller_selected_authority_allowed", "caller_environment_forwarded",
                 "launch_now", "attempt_now", "dc_now", "mapped_vcs_now",
                 "automatic_retry", "paper_citable")), "launch receipt false boundaries")
    require("m1118r3_outer_seal_file_sha256" not in launch_receipt,
            "launch receipt excludes future outer")
    require(LAUNCH_RECEIPT_ID[0] not in source and LAUNCH_RECEIPT_ID[2] not in source and
            "m1118r3_outer_seal_file_sha256" not in source,
            "launcher excludes receipt/future outer fixed point")
    require(contract["acyclicity"]["sha256_fixed_point_required"] is False and
            contract["launch_receipt"]["contains_future_m1118r3_outer"] is False,
            "source contract acyclic boundary")
    require(contract["claim_boundary"] == {
        "source_only": True, "attempt_consumed": False,
        "mapped_functionality": False, "activity_or_power": False,
        "performance": False, "system_speedup": False,
        "paper_citable": False, "paper_ppa_ready": False,
    }, "all source claim boundaries false")

    engine_flow = engine_source[engine_source.index("def flow()"):
                                engine_source.index("def quarantine(")]
    require(engine_flow.index("ATTEMPT.mkdir()") < engine_flow.index("FRESH_DC_M1112R3"),
            "attempt consumed before DC")
    require(engine_flow.count("DC_TARGET") == 1 and engine_flow.count("str(VCS)") == 1 and
            engine_flow.count("rc = run([str(simv)") == 1,
            "one DC compile and one mapped simulation")
    require("SHADOW_REGISTER_BITS = 337" in engine_source and
            "atomic_unknown_bitmap_each_cycle" in ENGINE_CONTRACT.read_text(encoding="utf-8") and
            "functional_feedback\": false" in ENGINE_CONTRACT.read_text(encoding="utf-8"),
            "async-reset observation/X/no-feedback preserved")

    namespace = definitions(LAUNCHER, "m1119r3_launcher_author_model")
    namespace["validate_hardcoded_authorities"](False)
    require(True, "hardcoded authority dry validation")
    with tempfile.TemporaryDirectory(prefix="m1119r3_author_") as temporary:
        root = Path(temporary)
        private_home = root / "home"
        private_home.mkdir(mode=0o700)
        environment = namespace["clean_child_environment"](private_home)
        require(set(environment) == {
            "HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE", "SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE",
        }, "child environment exact keys")
        require(environment["HOME"] == str(private_home) and
                environment["SNPSLMD_LICENSE_FILE"] == "27030@ic.ismd-nemo" and
                environment["PATH"] == "/usr/bin:/bin", "child environment exact values")
        poison = {"HOME": "/tmp/evil", "PATH": "/tmp/evil", "LD_PRELOAD": "/tmp/evil.so",
                  "PYTHONPATH": "/tmp/evil", "SNPSLMD_LICENSE_FILE": "evil"}
        backup = dict(os.environ)
        try:
            os.environ.clear(); os.environ.update(poison)
            require(namespace["clean_child_environment"](private_home) == environment,
                    "caller environment poison has no effect")
        finally:
            os.environ.clear(); os.environ.update(backup)
        good = root / "good"; good.write_bytes(b"good")
        changed = root / "changed"; changed.write_bytes(b"changed")
        link = root / "link"; link.symlink_to(good)
        rejected("regular symlink", namespace["verify_regular"], link, sha(good))
        rejected("regular byte drift", namespace["verify_regular"], changed, sha(good))

        sealed = root / "sealed"; sealed.mkdir()
        review = sealed / "review.json"; review.write_text(
            '{"status":"GOOD"}\n', encoding="utf-8")
        manifest = sealed / "SHA256SUMS"; manifest.write_text(
            sha(review) + "  review.json\n", encoding="utf-8")
        outer = sealed / "SHA256SUMS.seal.sha256"; outer.write_text(
            sha(manifest) + "  SHA256SUMS\n", encoding="utf-8")
        sealed_id = (sha(review), sha(manifest), sha(outer))
        namespace["verify_flat"](sealed, sealed_id, "GOOD")
        require(True, "temporary legal flat seal accepted")
        extra = sealed / "extra"; extra.write_text("x\n", encoding="utf-8")
        rejected("live seal extra", namespace["verify_flat"], sealed, sealed_id, "GOOD")
        extra.unlink()
        review_real = root / "review.real"; review.rename(review_real); review.symlink_to(review_real)
        rejected("live seal symlink", namespace["verify_flat"], sealed, sealed_id, "GOOD")

        old_attempt, old_result, old_lock = namespace["ATTEMPT"], namespace["RESULT"], namespace["LOCK"]
        old_hw, old_collision, old_mem = namespace["HW"], namespace["collision_gate"], namespace["read_meminfo"]
        fake_hw = root / "hw"; (fake_hw / "results").mkdir(parents=True)
        namespace.update({
            "HW": fake_hw, "ATTEMPT": fake_hw / "results/attempt",
            "RESULT": fake_hw / "results/result", "LOCK": root / "lock",
            "collision_gate": lambda: [],
            "read_meminfo": lambda: {"MemAvailable": 9 * 1024 * 1024,
                                      "CommitLimit": 20 * 1024 * 1024,
                                      "Committed_AS": 10 * 1024 * 1024},
        })
        namespace["namespace_resource_gate"]()
        require(True, "temporary fresh namespace accepted")
        namespace["ATTEMPT"].mkdir()
        rejected("stale attempt", namespace["namespace_resource_gate"])
        namespace.update({"HW": old_hw, "ATTEMPT": old_attempt, "RESULT": old_result,
                          "LOCK": old_lock, "collision_gate": old_collision,
                          "read_meminfo": old_mem})

        calls: list[tuple[Any, Any]] = []
        class Completed:
            returncode = 37
        original_run = namespace["subprocess"].run
        original_validate = namespace["validate_hardcoded_authorities"]
        original_gate = namespace["namespace_resource_gate"]
        original_mkdtemp = namespace["tempfile"].mkdtemp
        dry_home = Path("/tmp") / ("m1112r3_c2_home.authorcheck.%d" % os.getpid())
        require(not dry_home.exists() and not dry_home.is_symlink(),
                "dry private HOME namespace fresh")
        def fake_run(*args, **kwargs):
            calls.append((args, kwargs)); return Completed()
        def fake_mkdtemp(*, prefix, dir):
            require(prefix == "m1112r3_c2_home." and dir == "/tmp",
                    "private HOME fixed prefix/root")
            dry_home.mkdir(mode=0o700); return str(dry_home)
        old_argv = sys.argv[:]
        old_environ = dict(os.environ)
        try:
            namespace["subprocess"].run = fake_run
            namespace["validate_hardcoded_authorities"] = lambda enforce_runtime: require(
                enforce_runtime is True, "main requests runtime authority gate")
            namespace["namespace_resource_gate"] = lambda: require(True, "main requests namespace/resource gate")
            namespace["tempfile"].mkdtemp = fake_mkdtemp
            sys.argv = [str(LAUNCHER)]
            os.environ.clear(); os.environ.update({
                "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin",
                "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            })
            require(namespace["main"]() == 37, "dry main propagates child return")
            require(len(calls) == 1, "dry main exactly one child")
            argv = calls[0][0][0]; kwargs = calls[0][1]
            require(argv == [str(namespace["PYTHON"]), "-I", str(namespace["ENGINE"]),
                             "--authorized-launch"], "dry exact child argv")
            require(kwargs["cwd"] == str(HW) and kwargs["close_fds"] is True and
                    kwargs["check"] is False, "dry exact child controls")
            require(kwargs["env"]["HOME"] == str(dry_home) and not dry_home.exists(),
                    "dry private HOME used then removed")
        finally:
            namespace["subprocess"].run = original_run
            namespace["validate_hardcoded_authorities"] = original_validate
            namespace["namespace_resource_gate"] = original_gate
            namespace["tempfile"].mkdtemp = original_mkdtemp
            sys.argv = old_argv
            os.environ.clear(); os.environ.update(old_environ)

    forged = dict(launch_receipt); forged["m1118r3_outer_seal_file_sha256"] = "0" * 64
    require(set(forged) != expected_receipt_keys, "future outer injection rejected by exact schema")
    attacks.append("future outer injection")
    forged = dict(launch_receipt); forged["launcher_sha256"] = "0" * 64
    require(forged["launcher_sha256"] != LAUNCHER_SHA, "launcher pin mutation detected")
    attacks.append("launcher pin mutation")
    require(len(attacks) == 7, "all seven mutations rejected")

    result = {
        "schema": "m1119r3_m1112r3_c2_launcher_author_mechanical_v1",
        "status": "PASS_M1119R3_ZERO_ARG_LAUNCHER_AUTHOR_STATIC_AND_MUTATION_CHECK__NO_EXECUTION",
        "checks_passed": len(checks),
        "attacks_rejected": len(attacks),
        "attacks": attacks,
        "launcher_sha256": LAUNCHER_SHA,
        "launch_receipt_outer_seal_file_sha256": LAUNCH_RECEIPT_ID[2],
        "source_contract_outer_seal_file_sha256": SOURCE_CONTRACT_ID[2],
        "m1117r3_outer_seal_file_sha256": M1117R3_OUTER,
        "launcher_executed": False,
        "engine_executed": False,
        "attempt_created": False,
        "dc_executed": False,
        "mapped_vcs_executed": False,
        "future_hammer_required": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
