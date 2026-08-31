#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author, local-only final launch hammer for M1485/M1487.

This program performs no remote preflight, GPU query, capture, attempt
consumption, controller operation, or EDA invocation.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
SOURCE = HW / "scripts/run_m1485_m1480_nested_m1233_config_compat_one_shot.py"
TEST = HW / "tests/test_run_m1485_m1480_nested_m1233_config_compat_one_shot.py"
TEST1480 = HW / "tests/test_run_m1480_m1475_exact_type_config_compat_one_shot.py"
TEST1475 = HW / "tests/test_run_m1475_m1458_config_content_compat_one_shot.py"
CONTRACT = HW / "contracts/m1485_m1480_nested_m1233_config_compat_source_contract_r1_20260831.json"
RELEASE = HW / "contracts/m1487_m1485_nested_m1233_config_compat_launch_release_r1_20260831.json"
BLIND = HW / "reviews/m1486_m1485_nested_m1233_config_compat_source_blind_hammer_r1_20260831"
M1483 = HW / "reviews/m1483_m1482_m1480_m1475_exact_type_config_compat_final_launch_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "d9779f52bd6342898b26f14b05f8052888fd81cb35d73d10168319ade6d8db9a",
    TEST: "7ff297bfc5a16e3dc01b2bac089d216fb5a899a5acae889ba9f072734da4510c",
    CONTRACT: "44e8d98a5b3d997a16bdac158936e27e95eb4f66787602abc0c78edbd7aa7e2e",
    RELEASE: "f8fd63e34f0d1f983f083dc0d596528c81c6a1d1c60bfc83a215e3a0c51b9b1c",
    BLIND / "review.json": "28e4cb0df58276185800a0857d06a925d4f229fa787e092f0e889f95435ea78a",
    BLIND / "SHA256SUMS": "9bc332b6e0a3424588926778a4776410c5057699f2ef68fdfde69d3704afc7f9",
    BLIND / "SHA256SUMS.seal.sha256": "0314c98679a7dd25559fab5e06b9acd1f7d73949fbf7e874d531b4ea6485515f",
    BLIND / "hammer_output.json": "900605bf369a7ec22afbcd3af822cfec5877caa93bbbf51fe02a40339bc2a00b",
    M1483 / "review.json": "7df093c24f2826fe7ddd1127a429d1fdad4330deabf24f81245869074636caed",
    M1483 / "SHA256SUMS": "0c5beeac1c3cfa1b1319506b96810e3b30b7b24c19e61fdf7589da3c87894f21",
    M1483 / "SHA256SUMS.seal.sha256": "e07a9ac4e8e057b1ec29fa67d7486bb3c8c0f1249383b0aa08f08349c023df5d",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
STATUS = "PASS_M1488_M1485_NESTED_M1233_CONFIG_COMPAT_FINAL_LAUNCH"
RELEASE_STATUS = "AUTHORIZE_ONE_M1485_NESTED_M1233_CONFIG_COMPAT_M1458_ATTEMPT"
RUNNER_SHA = EXPECTED[SOURCE]
RELEASE_SHA = EXPECTED[RELEASE]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise RuntimeError("duplicate JSON key")
            value[key] = item
        return value
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    if type(value) is not dict:
        raise RuntimeError("JSON root is not object")
    return value


def regular_exact(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or path.is_symlink() or sha(path) != expected:
        raise RuntimeError("identity mismatch: " + str(path))


def verify_seal(root: Path, review_sha: str, manifest_sha: str,
                outer_sha: str) -> dict[str, Any]:
    regular_exact(root / "review.json", review_sha)
    regular_exact(root / "SHA256SUMS", manifest_sha)
    regular_exact(root / "SHA256SUMS.seal.sha256", outer_sha)
    if (root / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8") != \
            manifest_sha + "  SHA256SUMS\n":
        raise RuntimeError("outer seal content mismatch")
    rows = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in rows:
            raise RuntimeError("duplicate manifest member")
        rows[name] = digest
        member = ROOT / name if name != "review.json" else root / name
        if member.exists() and sha(member) != digest:
            raise RuntimeError("sealed member mismatch: " + name)
    rel = str((root / "review.json").relative_to(ROOT))
    if rows.get(rel, rows.get("review.json")) != review_sha:
        raise RuntimeError("review not sealed")
    return strict_json(root / "review.json")


def exact_authorization(value: Any, launch: bool) -> None:
    if type(value) is not dict or set(value) != {
            "launch", "runs", "automatic_retry", "controller_restore"}:
        raise RuntimeError("authorization shape")
    if not (value["launch"] is launch and type(value["runs"]) is int and
            value["runs"] == (1 if launch else 0) and
            value["automatic_retry"] is False and
            value["controller_restore"] is False):
        raise RuntimeError("authorization exact type/value")


def validate_release(value: Any, module: Any) -> None:
    if type(value) is not dict:
        raise RuntimeError("release object")
    capture = module.M1480.M1475.M1458
    required = {
        "schema", "status", "date", "objective", "runner_sha256", "result",
        "attempt", "log", "authorization", "m1485_source", "m1486_blind",
        "m1480_authority_chain_retained", "runtime_scope", "one_shot_policy",
        "final_gate", "release_author_execution", "claim_boundary", "docs359_sha256",
    }
    if set(value) != required:
        raise RuntimeError("release key shape")
    if value["schema"] != "m1487_m1485_nested_m1233_config_compat_launch_release_r1_v1" or \
            value["status"] != RELEASE_STATUS or value["runner_sha256"] != RUNNER_SHA:
        raise RuntimeError("release schema/status/runner")
    exact_authorization(value["authorization"], True)
    expected_names = (
        str(capture.CANONICAL_RESULT.relative_to(ROOT)),
        str(capture.CANONICAL_ATTEMPT.relative_to(ROOT)),
        str(capture.CANONICAL_LOG.relative_to(ROOT)),
    )
    if (value["result"], value["attempt"], value["log"]) != expected_names:
        raise RuntimeError("canonical namespace drift")
    blind = value["m1486_blind"]
    if type(blind) is not dict or blind.get("status") != \
            "PASS_M1486_M1485_NESTED_M1233_CONFIG_COMPAT_SOURCE" or \
            blind.get("review_sha256") != EXPECTED[BLIND / "review.json"] or \
            blind.get("manifest_sha256") != EXPECTED[BLIND / "SHA256SUMS"] or \
            blind.get("outer_file_sha256") != EXPECTED[BLIND / "SHA256SUMS.seal.sha256"] or \
            blind.get("false_negatives") != 0:
        raise RuntimeError("M1486 binding drift")
    exact_authorization(blind.get("authorization"), False)
    retained = value["m1480_authority_chain_retained"]
    if retained != {
        "runner_sha256": "3a0235f91d8d6acd4c94168b3b611cb53504f50e3843580c09bc1673042df4ce",
        "release_sha256": "5f458009e15e759e29b54d9306ade72ba74cd927bc62e0cf1c4ca49513fb1697",
        "final_review_sha256": EXPECTED[M1483 / "review.json"],
        "final_manifest_sha256": EXPECTED[M1483 / "SHA256SUMS"],
        "final_outer_file_sha256": EXPECTED[M1483 / "SHA256SUMS.seal.sha256"],
    }:
        raise RuntimeError("M1483 retained authority drift")
    one = value["one_shot_policy"]
    if type(one) is not dict or set(one) != {
            "attempt_create", "runs", "automatic_retry", "controller_restore"} or \
            one["attempt_create"] != "O_EXCL" or type(one["runs"]) is not int or \
            one["runs"] != 1 or one["automatic_retry"] is not False or \
            one["controller_restore"] is not False:
        raise RuntimeError("one-shot drift")
    gate = value["final_gate"]
    if type(gate) is not dict or set(gate) != {
            "path", "fresh_different_author_required", "present_at_release_authoring",
            "actual_launch_ready", "required_status"} or \
            gate["path"] != str(OUT.relative_to(ROOT)) or \
            gate["fresh_different_author_required"] is not True or \
            gate["present_at_release_authoring"] is not False or \
            gate["actual_launch_ready"] is not False or \
            gate["required_status"] != STATUS:
        raise RuntimeError("final gate drift")
    if value["docs359_sha256"] != EXPECTED[DOCS359]:
        raise RuntimeError("docs359 drift")


def rejected(thunk) -> bool:
    try:
        thunk()
    except BaseException:
        return True
    return False


def load_source():
    spec = importlib.util.spec_from_file_location("m1488_bound_m1485", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import M1485")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    checks = []
    attacks = []
    def check(name, passed):
        checks.append({"check": name, "pass": bool(passed)})
    def attack(name, thunk):
        caught = rejected(thunk)
        attacks.append({"attack": name, "rejected": caught,
                        "false_negative": not caught})

    for path, expected in EXPECTED.items():
        check("sha_" + path.name, path.is_file() and sha(path) == expected)
    module = load_source()
    release = strict_json(RELEASE)
    validate_release(release, module)
    check("release_exact", True)

    blind = verify_seal(BLIND, EXPECTED[BLIND / "review.json"],
                        EXPECTED[BLIND / "SHA256SUMS"],
                        EXPECTED[BLIND / "SHA256SUMS.seal.sha256"])
    exact_authorization(blind.get("authorization"), False)
    prior_out = strict_json(BLIND / "hammer_output.json")
    check("m1486_seal_status", blind.get("status") ==
          "PASS_M1486_M1485_NESTED_M1233_CONFIG_COMPAT_SOURCE")
    check("m1486_reused_attacks", prior_out.get("check_count") == 24 and
          prior_out.get("attack_count") == 65 and
          prior_out.get("false_negatives") == 0 and
          all(row.get("pass") is True for row in prior_out.get("checks", [])) and
          all(row.get("rejected") is True for row in prior_out.get("attacks", [])))

    prior = verify_seal(M1483, EXPECTED[M1483 / "review.json"],
                        EXPECTED[M1483 / "SHA256SUMS"],
                        EXPECTED[M1483 / "SHA256SUMS.seal.sha256"])
    check("m1483_seal_status", prior.get("status") ==
          "PASS_M1483_M1480_EXACT_TYPE_CONFIG_COMPAT_FINAL_LAUNCH")
    exact_authorization(prior.get("authorization"), True)

    native = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "pytest", "-q",
         str(TEST1475), str(TEST1480), str(TEST)], cwd=ROOT, text=True,
        capture_output=True, check=False)
    check("native_43", native.returncode == 0 and "43 passed" in native.stdout)

    # Exact release mutation campaign.  The authority tuple and canonical
    # namespaces must not survive scalar-type or value substitutions.
    for key in ("schema", "status", "runner_sha256", "result", "attempt", "log",
                "docs359_sha256"):
        changed = copy.deepcopy(release); changed[key] = str(changed[key]) + ".drift"
        attack("release_" + key, lambda changed=changed: validate_release(changed, module))
    changed = copy.deepcopy(release); changed["extra"] = 0
    attack("release_extra_key", lambda: validate_release(changed, module))
    for key, bad in (("launch", 1), ("runs", True), ("runs", 1.0),
                     ("automatic_retry", 0), ("controller_restore", 0)):
        changed = copy.deepcopy(release); changed["authorization"][key] = bad
        attack("authorization_type_" + key + "_" + type(bad).__name__,
               lambda changed=changed: validate_release(changed, module))
    for key in ("review_sha256", "manifest_sha256", "outer_file_sha256", "status"):
        changed = copy.deepcopy(release); changed["m1486_blind"][key] += ".drift"
        attack("m1486_" + key, lambda changed=changed: validate_release(changed, module))
    changed = copy.deepcopy(release); changed["m1486_blind"]["false_negatives"] = True
    attack("m1486_false_negative_bool", lambda: validate_release(changed, module))
    for key in ("final_review_sha256", "final_manifest_sha256",
                "final_outer_file_sha256"):
        changed = copy.deepcopy(release); changed["m1480_authority_chain_retained"][key] += ".drift"
        attack("m1483_" + key, lambda changed=changed: validate_release(changed, module))
    for key, bad in (("runs", True), ("automatic_retry", 0),
                     ("controller_restore", 0)):
        changed = copy.deepcopy(release); changed["one_shot_policy"][key] = bad
        attack("oneshot_" + key, lambda changed=changed: validate_release(changed, module))
    changed = copy.deepcopy(release); changed["final_gate"]["required_status"] += ".drift"
    attack("final_status_drift", lambda: validate_release(changed, module))
    changed = copy.deepcopy(release); changed["final_gate"]["actual_launch_ready"] = 0
    attack("final_ready_type", lambda: validate_release(changed, module))

    failed = [row for row in checks if not row["pass"]]
    false = [row for row in attacks if row["false_negative"]]
    verdict = "PASS" if not failed and not false else "FAIL"
    output = {
        "schema": "m1488_m1487_m1485_final_launch_hammer_output_r1_v1",
        "verdict": verdict, "check_count": len(checks),
        "attack_count": len(attacks), "failed_checks": failed,
        "false_negatives": len(false), "checks": checks, "attacks": attacks,
        "reused_m1486_campaign": "24/24 checks; 65/65 attacks; 0 false negatives",
        "native_pytest_stdout": native.stdout, "native_pytest_stderr": native.stderr,
        "execution": {"ssh": 0, "remote_preflight": 0, "gpu_queries": 0,
                      "capture_runs": 0, "attempts_consumed": 0,
                      "controller_operations": 0, "eda_runs": 0},
    }
    (OUT / "hammer_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if verdict != "PASS":
        raise RuntimeError("M1488 hammer failed")

    review = {
        "schema": "m1488_m1487_m1485_nested_m1233_config_compat_final_launch_hammer_r1_v1",
        "status": STATUS, "score": 100, "p0_count": 0, "p1_count": 0,
        "date": "2026-08-31",
        "verdict": "PASS. Exact M1487 release, canonical M1458 namespaces, exact M1486 and M1483 double seals, native 43/43 tests, reused M1486 65/65 mutations, and final release mutations pass with zero false negatives. This local-only hammer authorizes exactly one M1485-wrapped M1458 attempt with no retry and no controller restore.",
        "bindings": {
            "runner_sha256": RUNNER_SHA, "release_sha256": RELEASE_SHA,
            "m1486_review_sha256": EXPECTED[BLIND / "review.json"],
            "m1486_manifest_sha256": EXPECTED[BLIND / "SHA256SUMS"],
            "m1486_outer_file_sha256": EXPECTED[BLIND / "SHA256SUMS.seal.sha256"],
            "m1483_review_sha256": EXPECTED[M1483 / "review.json"],
            "m1483_manifest_sha256": EXPECTED[M1483 / "SHA256SUMS"],
            "m1483_outer_file_sha256": EXPECTED[M1483 / "SHA256SUMS.seal.sha256"],
            "docs359_sha256": EXPECTED[DOCS359],
        },
        "authorization": {"launch": True, "runs": 1,
                          "automatic_retry": False, "controller_restore": False},
        "verification": {"native_tests": "43/43 PASS",
                         "m1486_campaign": "24/24 checks; 65/65 rejected; 0 false negatives",
                         "final_checks": f"{len(checks)}/{len(checks)} PASS",
                         "final_mutations": f"{len(attacks)}/{len(attacks)} rejected; 0 false negatives",
                         "canonical_namespaces": "exact unchanged M1458 result/attempt/log"},
        "claim_boundary": {"launch_authority_only": True, "production_result": False,
                           "capture_executed": False, "hardware_result": False,
                           "cycles": False, "speedup": False, "energy": False,
                           "ppa": False, "headline": False},
        "execution": output["execution"],
    }
    (OUT / "review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "review.md").write_text(
        "# M1488 final launch hammer\n\nPASS. Exact M1487/M1486/M1483 authority and canonical M1458 one-shot namespaces are closed. No remote or production action was performed.\n",
        encoding="utf-8")
    (OUT / "mechanical_checks.txt").write_text(
        f"checks={len(checks)}/{len(checks)} PASS\nattacks={len(attacks)}/{len(attacks)} rejected\nfalse_negatives=0\nnative_tests=43/43 PASS\n",
        encoding="utf-8")
    (OUT / "PASS.txt").write_text(STATUS + "\n", encoding="utf-8")
    (OUT / "RUN_COMPLETE.txt").write_text("PASS\n", encoding="utf-8")

    members = [path for path in OUT.iterdir() if path.is_file() and
               path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}]
    lines = [sha(path) + "  " + str(path.relative_to(ROOT)) for path in sorted(members)]
    manifest = "\n".join(lines) + "\n"
    (OUT / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    manifest_sha = sha(OUT / "SHA256SUMS")
    (OUT / "SHA256SUMS.seal.sha256").write_text(
        manifest_sha + "  SHA256SUMS\n", encoding="utf-8")
    print(STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
