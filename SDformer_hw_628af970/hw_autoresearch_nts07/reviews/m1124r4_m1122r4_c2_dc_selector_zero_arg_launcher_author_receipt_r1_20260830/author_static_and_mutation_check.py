#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Source-only author checks for M1124r4; never execute launcher or engine."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


HW = Path(__file__).resolve().parents[2]
LAUNCHER = HW / "dc_handoff/scripts/run_m1122r4_c2_dc_selector_async_observation_authorized_launch_r1.py"
RECEIPT = HW / "contracts/m1122r4_c2_dc_selector_async_observation_authorized_launch_receipt_r1_20260830.json"
CONTRACT = HW / "contracts/m1124r4_m1122r4_c2_dc_selector_zero_arg_launcher_source_contract_r1_20260830.json"
ENGINE = HW / "dc_handoff/scripts/m1122r4_c2_dc_selector_async_observation_engine_source_r1.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed"
RESULT = HW / "results/m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830"
LOCK = Path("/tmp/m1122r4_c2_dc_selector_async_observation_eda.lock")
EXPECTED = {
    "launcher": "405cf1bd8a6af412ce44a727b47db90c14923054678946378bdfe2646a95ec78",
    "receipt": "bcdf5c00e32a8b39f40782232e197d3f450a3f0cc650da9a35ecf6d2da5cf138",
    "receipt_side": "b7aaa32cfbe821fc7313839d3c498b3febc6fca6a59c14cb1213d1156bcf6754",
    "receipt_outer": "532a1d07744d139288618bb7995291b9add2dd1c0b7376c7a2c917fa5a6d8113",
    "contract": "e2e55ebe7bd9699e020d1a35a9b1d6191071c9ae3f58cb1e00d4daab2aafbb89",
    "contract_side": "719c1edc38a24ef9efd4f3625bc31fa058872d144dc440f227ad522dbb1d5924",
    "contract_outer": "b5096402619ac2a9a49bda55c956a11a84fb104f2372fdb0dee6ec910a7e58d3",
    "engine": "f278052d251af0c2d150872391306c2f3922049ca04c7df2a0d9d3d074b55007",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


checks = 0
attacks = 0


def check(value: bool, message: str) -> None:
    global checks
    if not value:
        raise RuntimeError(message)
    checks += 1


def reject(action, message: str) -> None:
    global attacks
    try:
        action()
    except Exception:
        attacks += 1
        return
    raise RuntimeError("mutation survived: " + message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_loads(raw: str) -> Any:
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise RuntimeError("duplicate")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite " + token)))


def check_double(path: Path, primary: str, side_sha: str, outer_sha: str) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    check(sha(path) == primary, "primary SHA")
    check(sha(side) == side_sha, "side SHA")
    check(sha(outer) == outer_sha, "outer SHA")
    check(side.read_text().split() == [primary, path.relative_to(HW).as_posix()],
          "side content")
    check(outer.read_text().split() == [side_sha, side.relative_to(HW).as_posix()],
          "outer content")


def main() -> None:
    global checks
    check(tuple(sys.version_info[:3]) == (3, 10, 18), "Python 3.10.18")
    check(sha(LAUNCHER) == EXPECTED["launcher"], "launcher SHA")
    check(sha(ENGINE) == EXPECTED["engine"], "engine SHA")
    check(sha(DOCS359) == EXPECTED["docs359"], "docs359 SHA")
    check(stat.S_ISREG(LAUNCHER.lstat().st_mode) and not LAUNCHER.is_symlink(),
          "launcher direct regular")
    check_double(RECEIPT, EXPECTED["receipt"], EXPECTED["receipt_side"],
                 EXPECTED["receipt_outer"])
    check_double(CONTRACT, EXPECTED["contract"], EXPECTED["contract_side"],
                 EXPECTED["contract_outer"])

    source = LAUNCHER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    check(source.startswith("#!/opt/anaconda3/envs/pytorch310/bin/python3.10\n"),
          "pinned shebang")
    check("m1125r4_outer_seal_file_sha256" not in source,
          "no future hammer outer in launcher")
    check(EXPECTED["receipt"] not in source and EXPECTED["receipt_outer"] not in source,
          "no receipt backedge in launcher")
    check("--authorized-launch" in source and
          "m1122r4_c2_dc_selector_async_observation_engine_source_r1.py" in source,
          "exact engine mode present")
    check("common_shell_exec" in source and "common_shell_exe" in source,
          "selector collision aliases present")
    check("os.environ == ROOT_ENV" in source, "exact root environment gate")
    check("env=clean_child_environment(private_home)" in source,
          "constant child environment")
    check("automatic_retry" not in source, "launcher has no retry mechanism")
    run_calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and
                 isinstance(node.func, ast.Attribute) and
                 isinstance(node.func.value, ast.Name) and
                 node.func.value.id == "subprocess" and node.func.attr == "run"]
    check(len(run_calls) == 2, "one pgrep site plus one engine child site")
    check(sum("--authorized-launch" in ast.unparse(node) for node in run_calls) == 1,
          "exactly one engine child site")
    check(not any(isinstance(node, ast.Call) and
                  isinstance(node.func, ast.Attribute) and node.func.attr == "Popen"
                  for node in ast.walk(tree)), "no alternate child site")

    receipt = strict_loads(RECEIPT.read_text(encoding="utf-8"))
    expected_keys = {
        "schema", "status", "launcher_sha256", "engine_sha256",
        "engine_contract_sha256", "engine_contract_outer_seal_file_sha256",
        "engine_author_receipt_outer_seal_file_sha256",
        "m1121_outer_seal_file_sha256", "m1123r4_outer_seal_file_sha256",
        "arguments", "caller_selected_authority_allowed",
        "caller_environment_forwarded", "m1125r4_required", "launch_now",
        "attempt_now", "dc_now", "mapped_vcs_now", "maximum_attempts",
        "automatic_retry", "paper_citable",
    }
    check(set(receipt) == expected_keys, "receipt exact keys")
    check(receipt["status"] ==
          "M1122R4_LAUNCH_SOURCE_FROZEN__M1125R4_REQUIRED__NO_EDA",
          "receipt status")
    check(receipt["arguments"] == 0 and receipt["maximum_attempts"] == 1,
          "zero argument one-shot")
    check(receipt["m1125r4_required"] is True, "M1125 required")
    for key in ("caller_selected_authority_allowed", "caller_environment_forwarded",
                "launch_now", "attempt_now", "dc_now", "mapped_vcs_now",
                "automatic_retry", "paper_citable"):
        check(receipt[key] is False, key + " false")
    check("m1125r4_outer_seal_file_sha256" not in receipt,
          "receipt no future hash")

    contract = strict_loads(CONTRACT.read_text(encoding="utf-8"))
    check(contract["authorization"]["different_author_m1125r4_final_launcher_hammer"] is True,
          "only next hammer authorized")
    check(contract["authorization"]["launcher_execution"] is False and
          contract["authorization"]["engine_execution"] is False and
          contract["authorization"]["attempt"] is False and
          contract["authorization"]["dc"] is False and
          contract["authorization"]["mapped_vcs"] is False,
          "no execution authorization")
    check(contract["acyclicity"]["sha256_fixed_point_required"] is False,
          "acyclic chain")

    check(not ATTEMPT.exists() and not ATTEMPT.is_symlink(), "attempt absent")
    check(not RESULT.exists() and not RESULT.is_symlink(), "result absent")
    check(not LOCK.exists() and not LOCK.is_symlink(), "lock absent")
    check(not any((HW / "results").glob(
          ".m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_work.*")),
          "work absent")
    check(not any((HW / "results").glob(
          "m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*")),
          "failure absent")

    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("m1124r4_launcher_under_test", LAUNCHER)
    check(spec is not None and spec.loader is not None, "import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    authority = module.validate_hardcoded_authorities(enforce_runtime=False)
    check(authority["status"] == "PASS_M1124R4_PREEXISTING_HARDCODED_AUTHORITIES",
          "hardcoded authority validation")
    clean = module.clean_child_environment(Path("/tmp/m1124r4_mock_home"))
    check(set(clean) == {"HOME", "LANG", "LC_ALL", "PATH", "TMPDIR",
                         "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
                         "SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"},
          "child environment exact keys")
    check(clean["SNPSLMD_LICENSE_FILE"] == "27030@ic.ismd-nemo",
          "license route constant")

    reject(lambda: strict_loads('{"x":1,"x":2}'), "duplicate JSON")
    reject(lambda: strict_loads('{"x":NaN}'), "NaN")
    reject(lambda: strict_loads('{"x":Infinity}'), "Infinity")
    reject(lambda: module.require(False, "expected"), "require false")
    original_argv = list(sys.argv)
    original_env = dict(os.environ)
    try:
        sys.argv[:] = [str(LAUNCHER), "--caller-authority"]
        os.environ.clear()
        os.environ.update(module.ROOT_ENV)
        reject(lambda: module.validate_hardcoded_authorities(True), "caller argument")
        sys.argv[:] = [str(LAUNCHER)]
        os.environ["CALLER_AUTHORITY"] = "forbidden"
        reject(lambda: module.validate_hardcoded_authorities(True), "caller environment")
    finally:
        sys.argv[:] = original_argv
        os.environ.clear()
        os.environ.update(original_env)

    print(json.dumps({
        "schema": "m1124r4_author_static_and_mutation_checks_r1_v1",
        "status": "PASS_M1124R4_SOURCE_ONLY_STATIC_AND_MUTATION_CHECKS__NO_EDA",
        "checks_passed": checks,
        "attacks_rejected": attacks,
        "launcher_sha256": sha(LAUNCHER),
        "launch_receipt_outer_seal_file_sha256": sha(Path(str(RECEIPT) + ".sha256.seal.sha256")),
        "source_contract_outer_seal_file_sha256": sha(Path(str(CONTRACT) + ".sha256.seal.sha256")),
        "m1125r4_required": True,
        "launch_or_attempt_or_eda": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
