#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only production-provenance addendum for the M1512 capture review."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M1512 = HW / (
    "reviews/m1512_m1501_m1458_ep34_capture_source_result_"
    "independent_hammer_r1_20260831")
RESULT = HW / (
    "results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_"
    "s40_r1_20260831")
LOG = HW / (
    "results/.m1458_m1434_motion_ep34_live93_unified_hardware_capture_"
    "s40_r1_20260831.production.log")
ATTEMPT = HW / (
    "results/.m1458_m1434_motion_ep34_live93_unified_hardware_capture_"
    "s40_r1_20260831.attempt_consumed")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    "m1512_review": "b302e94375f925d84a45eb798579f243fa68b13724d3f63fabfe2810948dbb74",
    "m1512_manifest": "2af7a59b6a4df07dc6047c0d48c52b7798b7f0803e31e290b2ad842e6c154b81",
    "m1512_outer": "ccbcd7bf1b99fd944062a6fb220d7ec719d96da91c190697db125cbd4ad58f7c",
    "result_manifest": "f7f7a08696611875837196b990575453141b5e8edbf6d4aae61f7db1ed238b8e",
    "result_outer": "7cf434b834d30c003153eef8e83e70d574b1c5a7d20ca4c2208902c6e0c76eed",
    "production_log": "21cca1464323c3f506a885a049d0edd0653b32116b62b2d7a6d962f80a1b9122",
    "attempt": "1569412d598f8889c6b7cebaacf43908cb1f853fa7353ee57314706833929cc2",
    "checkpoint": "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    "config": "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39",
    "profile": "144ba2d94eeafd2b6549a7b0aa7d0c89d2b334fe814a7d45f71d6990670e379c",
    "m1458_runner": "e81c20056dd261619f88884f2f097c9b594887927121d9e599a4f89185d33154",
    "m1434_source": "b28c8507f077b754048fc54afd9fe04900dac854b273df2ba1981fa5f892b6ed",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
CONTROLLER = {
    "argv": [
        "/opt/conda/envs/sdformerflow/bin/python", "-u",
        "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
        "run_mvsec_strict_c00_continuation_20260830.py"],
    "cwd": "/root/private_data/work/sdformer_codex/SDformer",
    "exe": "/opt/conda/envs/sdformerflow/bin/python3.10",
    "pid": 3804343, "ppid": 1, "start_ticks": 703730691, "state": "T",
}
EXPECTED_LOG = {
    "automatic_retry": False,
    "canonical_result_promotion_permitted": True,
    "controller": CONTROLLER,
    "controller_restore_permitted": True,
    "controller_restore_permitted_after_success": True,
    "controller_restored_by_runner": False,
    "detail": "result double seal verified; later restore only",
    "failure_quarantine_required": False,
    "schema": "m1458_m1434_ep34_live93_production_log_r1_v1",
    "status": "PASS",
}
EXPECTED_ATTEMPT = {
    "automatic_retry": False,
    "controller": CONTROLLER,
    "controller_restore_permitted": False,
    "gpu_uuid": "GPU-499236d3-b46c-5d25-4a22-530d47ed5112",
    "m1434_source_sha256": PINS["m1434_source"],
    "runner_sha256": PINS["m1458_runner"],
    "schema": "m1458_m1434_ep34_live93_attempt_r1_v1",
    "status": "ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular_exact(path: Path, digest: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError("missing " + label) from error
    if (not stat.S_ISREG(mode) or path.is_symlink()
            or sha(path) != digest):
        raise RuntimeError(label + " identity drift")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise RuntimeError("duplicate JSON key")
            output[key] = value
        return output
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON")))
    if type(value) is not dict:
        raise RuntimeError("JSON root")
    return value


def verify_seal(root: Path, manifest_sha: str, outer_sha: str) -> set[str]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("sealed root invalid")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular_exact(manifest, manifest_sha, "manifest")
    regular_exact(outer, outer_sha, "outer")
    if outer.read_text().split() != [manifest_sha, "SHA256SUMS"]:
        raise RuntimeError("outer content drift")
    listed = set()
    for row in manifest.read_text().splitlines():
        digest, name = row.split(maxsplit=1)
        name = name.lstrip("*")
        relative = Path(name)
        if (re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or name in listed or relative.is_absolute()
                or ".." in relative.parts):
            raise RuntimeError("unsafe manifest row")
        regular_exact(root / relative, digest, "sealed member")
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    if listed != actual:
        raise RuntimeError("sealed population drift")
    return listed


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1513_MUTATED"
    if type(value) is list:
        return list(value) + ["M1513_MUTATED"]
    if type(value) is dict:
        output = copy.deepcopy(value)
        output["m1513_mutated"] = True
        return output
    raise TypeError(type(value))


def main() -> int:
    checks = []
    attacks = []
    def check(name: str, value: bool) -> None:
        checks.append({"check": name, "pass": bool(value)})
    def attack(name: str, candidate: dict[str, Any], expected: dict[str, Any]):
        rejected = candidate != expected
        attacks.append({"attack": name, "rejected": rejected,
                        "false_negative": not rejected})

    regular_exact(DOCS359, PINS["docs359"], "docs359")
    verify_seal(M1512, PINS["m1512_manifest"], PINS["m1512_outer"])
    regular_exact(M1512 / "review.json", PINS["m1512_review"], "M1512 review")
    m1512 = strict_json(M1512 / "review.json")
    if (m1512.get("status") !=
            "PASS_M1512_M1501_M1458_EP34_CAPTURE_SOURCE_AND_RESULT"
            or m1512.get("claim_boundary", {}).get(
                "capture_content_validated") is not True
            or m1512.get("production_log_boundary", {}).get(
                "status_asserted") is not False
            or m1512.get("verification", {}).get("identity") != {
                "checkpoint_sha256": PINS["checkpoint"],
                "config_sha256": PINS["config"],
                "profile_sha256": PINS["profile"]}):
        raise RuntimeError("M1512 binding drift")
    check("m1512_capture_content_pass", True)

    result_members = verify_seal(
        RESULT, PINS["result_manifest"], PINS["result_outer"])
    check("result_top_seal", "manifest.json" in result_members)
    regular_exact(LOG, PINS["production_log"], "production log")
    regular_exact(ATTEMPT, PINS["attempt"], "attempt token")
    log = strict_json(LOG)
    attempt = strict_json(ATTEMPT)
    if log != EXPECTED_LOG:
        raise RuntimeError("production log exact-set/value drift")
    if attempt != EXPECTED_ATTEMPT:
        raise RuntimeError("attempt exact-set/value drift")
    if log["controller"] != attempt["controller"]:
        raise RuntimeError("controller cross-binding drift")
    check("production_log_exact_pass", True)
    check("attempt_exact_consumed_no_retry", True)
    check("controller_cross_binding", True)

    for key, value in EXPECTED_LOG.items():
        candidate = copy.deepcopy(EXPECTED_LOG)
        candidate[key] = changed(value)
        attack("log_" + key, candidate, EXPECTED_LOG)
    for key, value in EXPECTED_ATTEMPT.items():
        candidate = copy.deepcopy(EXPECTED_ATTEMPT)
        candidate[key] = changed(value)
        attack("attempt_" + key, candidate, EXPECTED_ATTEMPT)
    p0 = sum(not item["rejected"] for item in attacks)
    p1 = sum(not item["pass"] for item in checks)
    output = {
        "schema": "m1513_m1512_m1458_ep34_production_provenance_addendum_output_r1_v1",
        "status": "PASS_M1513_COMPLETE_M1458_EP34_PRODUCTION_PROVENANCE"
                  if p0 == 0 and p1 == 0 else "FAIL_CLOSED_DO_NOT_CITE",
        "checks": checks, "attacks": attacks,
        "summary": {"checks_passed": sum(item["pass"] for item in checks),
                    "checks_total": len(checks),
                    "attacks_rejected": sum(item["rejected"] for item in attacks),
                    "attacks_total": len(attacks),
                    "p0_count": p0, "p1_count": p1},
        "production_log": {"sha256": PINS["production_log"],
            "status": "PASS", "automatic_retry": False,
            "canonical_result_promotion_permitted": True,
            "failure_quarantine_required": False,
            "controller_restore_permitted": True},
        "attempt": {"sha256": PINS["attempt"],
            "status": "ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE",
            "automatic_retry": False,
            "runner_sha256": PINS["m1458_runner"],
            "m1434_source_sha256": PINS["m1434_source"]},
        "capture_binding": {"m1512_review_sha256": PINS["m1512_review"],
            "result_manifest_sha256": PINS["result_manifest"],
            "result_outer_file_sha256": PINS["result_outer"],
            "checkpoint_sha256": PINS["checkpoint"],
            "config_sha256": PINS["config"],
            "profile_sha256": PINS["profile"]},
        "claim_boundary": {"production_provenance_complete": True,
            "capture_content_validated_by_m1512": True,
            "cycles": False, "speedup": False, "energy": False,
            "ppa": False, "system_speedup": False, "headline": False},
        "execution": {"remote": 0, "gpu": 0, "capture": 0,
                      "controller_signal": 0, "eda": 0},
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if p0 == 0 and p1 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
