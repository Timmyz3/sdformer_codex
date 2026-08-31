#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author M1134r6 full-authority fixture hammer; no launch or EDA."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
SOURCE = HW / "dc_handoff/scripts/m1133r6_c2_authority_schema_repair_engine_source_r1.py"
CONTRACT = HW / "contracts/m1133r6_c2_authority_schema_repair_engine_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1133r6_c2_authority_schema_repair_engine_author_receipt_r1_20260830"
STOP_R5 = HW / "reviews/m1132r5_m1129r5_c2_dc_selector_launch_hammer_r1_20260830"
R4_ATTEMPT = HW / "results/.m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed"
R4_FAILURE = HW / "results/m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.580027.quarantine"
R3_ATTEMPT = HW / "results/.m1112r3_c2_async_observation_dc_mapped_vcs_attempt_consumed"
R3_FAILURE = HW / "results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.213812.quarantine"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUTPUT = HERE / "mechanical_checks.json"

EXPECTED = {
    "source": "1f8a190d7d1c8b7804e7302c8b6a38c30a49df466b6394a82e8f0cf4cec2ee40",
    "contract": "4dc16ffccb3c4a145f69f565500d67407ca821304ee838f93659918055a3ac8a",
    "contract_side": "bfd415d8540c2cb44b66683e127d66b7f3444b70840423ca4c66cf51f58e5ec7",
    "contract_outer": "82b6d6a6568fc8fc95f1a1b7b6bf05690e06e064a143de41eadfa0e76ac9b849",
    "author_review": "8fb36c424903047227a05c059ef3435e8a7769dd261245362c15292e22fe0777",
    "author_manifest": "8412b78900050b0bee3556244d4283c1a84847946f7a0622d6deada1c5473b04",
    "author_outer": "5b2e0a659992c006d5caee72f5bcd72fd28dfdc07266d7edd2c814f1bc4a3b68",
    "stop_r5_outer": "bc073b90787189710986381b74c18b9a3afbe4ccd2f7969e85b596d3df1adf48",
    "r4_attempt_outer": "8a012c8638c2e8a8da743cbf570a13f5c8bc8d85716b433882d03405e12e5e37",
    "r4_failure_outer": "2f9173b1e988b1f639e6c3d683fdf720fa9debfeaca8caf27bf5845a36527f83",
    "r3_attempt_outer": "b3355ec5ad9e896512f09609d46336b32554889604a352d87dbdd11200a93816",
    "r3_failure_outer": "537981717cddd3c70fc0ddc9bd6297158884f15b5cceee7c51eab9388a1562d6",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

checks = 0
attacks: list[str] = []


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise HammerFailure(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_regular(path: Path, digest: str) -> None:
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and sha(path) == digest,
            "regular identity drift: " + str(path))


def verify_double(path: Path, primary: str, side_sha: str, outer_sha: str) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, primary); verify_regular(side, side_sha); verify_regular(outer, outer_sha)
    require(side.read_text(encoding="utf-8").split() ==
            [primary, path.relative_to(HW).as_posix()], "double side content")
    require(outer.read_text(encoding="utf-8").split() ==
            [side_sha, side.relative_to(HW).as_posix()], "double outer content")


def verify_flat(directory: Path, outer_sha: str) -> dict:
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(),
            "sealed directory drift")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, outer_sha)
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"],
            "sealed outer content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(name not in expected and not rel.is_absolute() and ".." not in rel.parts,
                "manifest member safety")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact-member set")
    for name, digest in expected.items(): verify_regular(directory / name, digest)
    payload = next(name for name in ("review.json", "attempt.json", "failure.json")
                   if (directory / name).exists())
    return json.loads((directory / payload).read_text(encoding="utf-8"))


def expect_gate(label: str, operation) -> None:
    try:
        operation()
    except (E.GateFailure, RuntimeError, KeyError):
        attacks.append(label)
        return
    raise HammerFailure("attack accepted: " + label)


def canonical_snapshot() -> dict:
    results = HW / "results"
    return {
        "r5_attempt": E.R5_ATTEMPT.exists(), "r5_result": E.R5_RESULT.exists(),
        "r5_work": sorted(path.name for path in results.glob(E.R5_WORK_GLOB)),
        "r5_failure": sorted(path.name for path in results.glob(E.R5_FAILURE_GLOB)),
        "r5_lock": E.R5_LOCK.exists(),
        "r6_attempt": E.ATTEMPT.exists(), "r6_result": E.RESULT.exists(),
        "r6_work": sorted(path.name for path in results.glob(E.WORK_GLOB)),
        "r6_failure": sorted(path.name for path in results.glob(E.FAILURE_GLOB)),
        "r6_lock": E.LOCK.exists(),
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seal_flat(directory: Path) -> str:
    members = sorted(path for path in directory.rglob("*") if path.is_file() and
                     path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha(path)}  {path.relative_to(directory).as_posix()}\n" for path in members),
        encoding="utf-8")
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n", encoding="utf-8")
    return sha(outer)


def seal_double(path: Path) -> str:
    side = Path(str(path) + ".sha256")
    side.write_text(f"{sha(path)}  {path.relative_to(HW).as_posix()}\n", encoding="utf-8")
    outer = Path(str(path) + ".sha256.seal.sha256")
    outer.write_text(f"{sha(side)}  {side.relative_to(HW).as_posix()}\n", encoding="utf-8")
    return sha(outer)


spec = importlib.util.spec_from_file_location("m1134r6_independent_subject", SOURCE)
assert spec is not None and spec.loader is not None
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def build_fixture(root: Path, mutate=None, engine_mutate=None, launch_mutate=None) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    launcher = root / "launcher.py"
    launcher.write_text("# controlled mock; never executed\n", encoding="utf-8")
    m1134 = root / "m1134"
    m1134.mkdir()
    author_outer = EXPECTED["author_outer"]
    engine_review = {
        "status": "PASS_M1134R6_M1133R6_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA",
        "identity": {
            "engine_sha256": sha(SOURCE),
            "contract_sha256": E.CONTRACT_SHA256,
            "author_receipt_outer_seal_file_sha256": author_outer,
        },
    }
    if engine_mutate is not None:
        engine_mutate(engine_review)
    write_json(m1134 / "review.json", engine_review)
    m1134_outer = seal_flat(m1134)
    receipt = root / "receipt.json"
    value = {
        "schema": "m1133r6_c2_authority_schema_repair_authorized_launch_receipt_r1_v1",
        "status": "M1133R6_LAUNCH_SOURCE_FROZEN__M1136R6_REQUIRED__NO_EDA",
        "launcher_sha256": sha(launcher),
        "engine_sha256": sha(SOURCE),
        "engine_contract_sha256": E.CONTRACT_SHA256,
        "engine_contract_outer_seal_file_sha256": E.CONTRACT_OUTER_SHA256,
        "engine_author_receipt_outer_seal_file_sha256": author_outer,
        "m1121_outer_seal_file_sha256": E.M1121_OUTER_SHA256,
        "m1132r5_stop_outer_seal_file_sha256": E.M1132R5_STOP_OUTER_SHA256,
        "m1134r6_outer_seal_file_sha256": m1134_outer,
        "arguments": 0,
        "caller_selected_authority_allowed": False,
        "caller_environment_forwarded": False,
        "m1136r6_required": True,
        "launch_now": False,
        "attempt_now": False,
        "dc_now": False,
        "mapped_vcs_now": False,
        "maximum_attempts": 1,
        "automatic_retry": False,
        "paper_citable": False,
    }
    if mutate is not None:
        mutate(value)
    write_json(receipt, value)
    receipt_outer = seal_double(receipt)
    m1136 = root / "m1136"
    m1136.mkdir()
    launch_review = {
        "status": "PASS_M1136R6_M1133R6_FINAL_LAUNCH_HAMMER__GO_ONE_ATTEMPT",
        "identity": {
            "launch_receipt_outer_seal_file_sha256": receipt_outer,
            "launcher_sha256": sha(launcher),
            "engine_sha256": sha(SOURCE),
            "engine_contract_outer_seal_file_sha256": E.CONTRACT_OUTER_SHA256,
            "engine_author_receipt_outer_seal_file_sha256": author_outer,
            "m1121_outer_seal_file_sha256": E.M1121_OUTER_SHA256,
            "m1132r5_stop_outer_seal_file_sha256": E.M1132R5_STOP_OUTER_SHA256,
            "m1134r6_outer_seal_file_sha256": m1134_outer,
        },
    }
    if launch_mutate is not None:
        launch_mutate(launch_review)
    write_json(m1136 / "review.json", launch_review)
    seal_flat(m1136)
    return {"launcher": launcher, "receipt": receipt, "m1134": m1134, "m1136": m1136}


def bind_fixture(paths: dict) -> None:
    E.LAUNCHER = paths["launcher"]
    E.LAUNCH_RECEIPT = paths["receipt"]
    E.M1134R6 = paths["m1134"]
    E.M1136R6 = paths["m1136"]
    E.BASE.LAUNCHER = paths["launcher"]
    E.BASE.LAUNCH_RECEIPT = paths["receipt"]
    # Only /proc parent identity is controlled; all sealed authority functions
    # and static source checks remain the real subject implementation.
    E.BASE.verify_parent_launcher = lambda _receipt: None


def expect_failure(paths: dict, label: str) -> None:
    bind_fixture(paths)
    try:
        E.verify_future_authority()
    except E.GateFailure:
        attacks.append(label)
        return
    raise AssertionError(label + " did not fail closed")


def main() -> int:
    original_argv = list(sys.argv)
    before = canonical_snapshot()
    require(before == {"r5_attempt": False, "r5_result": False, "r5_work": [],
                       "r5_failure": [], "r5_lock": False, "r6_attempt": False,
                       "r6_result": False, "r6_work": [], "r6_failure": [],
                       "r6_lock": False}, "canonical r5/r6 namespaces must be absent")
    verify_regular(SOURCE, EXPECTED["source"])
    verify_double(CONTRACT, EXPECTED["contract"], EXPECTED["contract_side"],
                  EXPECTED["contract_outer"])
    author = verify_flat(AUTHOR, EXPECTED["author_outer"])
    stop = verify_flat(STOP_R5, EXPECTED["stop_r5_outer"])
    verify_regular(AUTHOR / "review.json", EXPECTED["author_review"])
    verify_regular(AUTHOR / "SHA256SUMS", EXPECTED["author_manifest"])
    verify_regular(DOCS359, EXPECTED["docs359"])
    require(author["status"] ==
            "PASS_M1133R6_AUTHORITY_SCHEMA_REPAIR_ENGINE_AUTHOR_RECEIPT__M1134R6_REQUIRED__NO_EDA",
            "author status")
    require(stop["status"] ==
            "FAIL_M1132R5_M1129R5_POSTSEAL_FUTURE_AUTHORITY__ADDITIVE_R6_REQUIRED__NO_LAUNCH" and
            stop["authorization"]["r5_command_withdrawn"] is True,
            "r5 stop status")
    source_text = SOURCE.read_text(encoding="utf-8")
    require('engine_identity["m1121_outer_seal_file_sha256"]' not in source_text and
            'set(engine_identity) != {"engine_sha256", "contract_sha256",' in source_text and
            'receipt["m1121_outer_seal_file_sha256"] != M1121_OUTER_SHA256' in source_text,
            "schema repair source boundary")

    old_rows = []
    for directory, outer, payload, retry_key in (
            (R4_ATTEMPT, EXPECTED["r4_attempt_outer"], "attempt.json", None),
            (R4_FAILURE, EXPECTED["r4_failure_outer"], "failure.json", "m1122r4_retry"),
            (R3_ATTEMPT, EXPECTED["r3_attempt_outer"], "attempt.json", None),
            (R3_FAILURE, EXPECTED["r3_failure_outer"], "failure.json", "m1112_retry")):
        old = verify_flat(directory, outer)
        if retry_key is None:
            require(old["dc_attempts"] == 1, "old attempt count")
        else:
            require(old["status"] == "FAILED_DIAGNOSTIC_DO_NOT_CITE" and
                    old[retry_key] is False, "old permanent no-retry")
        old_rows.append({"directory": directory.name, "outer": outer,
                         "retry": None if retry_key is None else old[retry_key]})

    with tempfile.TemporaryDirectory(prefix=".m1133r6_fixture.", dir=HW / "results") as tmp:
        tmp_root = Path(tmp)
        paths = build_fixture(tmp_root / "valid")
        bind_fixture(paths)
        result = E.verify_future_authority()
        require(result["m1121_outer_seal_file_sha256"] == E.M1121_OUTER_SHA256 and
                result["m1132r5_stop_outer_seal_file_sha256"] == E.M1132R5_STOP_OUTER_SHA256 and
                "m1136r6_outer_seal_file_sha256" in result,
                "verify_future_authority full return")
        sys.argv[:] = [str(SOURCE), "--authorized-launch"]
        static = E.static_gate()
        require(static["m1121_exact_static_authority"] is True and
                static["m1132r5_stop_exact_static_authority"] is True and
                static["m1121_outer_seal_file_sha256"] == E.M1121_OUTER_SHA256,
                "static_gate full return")

        def missing(value):
            del value["m1121_outer_seal_file_sha256"]

        def extra(value):
            value["unexpected_authority"] = False

        def wrong(value):
            value["m1121_outer_seal_file_sha256"] = "0" * 64

        for label, mutation in (("missing", missing), ("extra", extra), ("wrong", wrong)):
            expect_failure(build_fixture(tmp_root / ("receipt_" + label), mutation),
                           "receipt_m1121_" + label)

        def engine_extra_m1121(review):
            review["identity"]["m1121_outer_seal_file_sha256"] = E.M1121_OUTER_SHA256

        def engine_missing(review):
            del review["identity"]["contract_sha256"]

        def engine_wrong(review):
            review["identity"]["author_receipt_outer_seal_file_sha256"] = "0" * 64

        for label, mutation in (("extra_m1121", engine_extra_m1121),
                                ("missing_contract", engine_missing),
                                ("wrong_author", engine_wrong)):
            expect_failure(build_fixture(tmp_root / ("engine_" + label), engine_mutate=mutation),
                           "engine_identity_" + label)

        def launch_missing(review):
            del review["identity"]["m1121_outer_seal_file_sha256"]

        def launch_extra(review):
            review["identity"]["unexpected"] = False

        def launch_wrong(review):
            review["identity"]["m1121_outer_seal_file_sha256"] = "f" * 64

        for label, mutation in (("missing", launch_missing), ("extra", launch_extra),
                                ("wrong", launch_wrong)):
            expect_failure(build_fixture(tmp_root / ("launch_" + label), launch_mutate=mutation),
                           "launch_identity_m1121_" + label)

        bind_fixture(paths)
        r5_marker = tmp_root / "consumed_r5_attempt"
        r5_marker.mkdir()
        saved_r5_attempt = E.R5_ATTEMPT
        try:
            E.R5_ATTEMPT = r5_marker
            expect_gate("r5_namespace_reuse", E.static_gate)
        finally:
            E.R5_ATTEMPT = saved_r5_attempt

        bad_stop = tmp_root / "bad_stop"
        shutil.copytree(STOP_R5, bad_stop)
        (bad_stop / "live_extra.attack").write_text("attack\n", encoding="utf-8")
        saved_stop = E.M1132R5_STOP
        try:
            E.M1132R5_STOP = bad_stop
            expect_gate("r5_stop_live_extra", E.static_gate)
        finally:
            E.M1132R5_STOP = saved_stop

        for label, original, outer, retry_key in (
                ("r4_no_retry", R4_FAILURE, EXPECTED["r4_failure_outer"], "m1122r4_retry"),
                ("r3_no_retry", R3_FAILURE, EXPECTED["r3_failure_outer"], "m1112_retry")):
            bad = tmp_root / label
            shutil.copytree(original, bad)
            payload = json.loads((bad / "failure.json").read_text(encoding="utf-8"))
            payload[retry_key] = True
            write_json(bad / "failure.json", payload)
            seal_flat(bad)
            expect_gate(label + "_mutation", lambda bad=bad, outer=outer:
                        E.BASE.verify_exact_flat(bad, outer))
    sys.argv[:] = original_argv
    require(sha(E.BASE_ENGINE) == E.BASE_ENGINE_SHA256 and
            sha(E.BASE.RTL) == E.RTL_SHA256 and sha(E.BASE.TB) == E.TB_SHA256 and
            sha(E.BASE.FILELIST) == E.FILELIST_SHA256, "frozen implementation identity")
    after = canonical_snapshot()
    require(after == before, "canonical namespaces changed")
    result_payload = {
        "schema": "m1134r6_m1133r6_full_authority_fixture_hammer_r1_v1",
        "status": "PASS_M1134R6_FULL_AUTHORITY_FIXTURE__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA",
        "checks_passed": checks,
        "attacks_rejected": len(attacks),
        "attack_labels": attacks,
        "full_return": {"verify_future_authority": True, "static_gate": True,
                        "m1121_keyerror_eliminated": True},
        "old_no_retry": old_rows,
        "identity": {"engine_sha256": sha(SOURCE), "contract_sha256": sha(CONTRACT),
                     "author_receipt_outer_seal_file_sha256": sha(AUTHOR / "SHA256SUMS.seal.sha256"),
                     "m1132r5_stop_outer_seal_file_sha256": sha(STOP_R5 / "SHA256SUMS.seal.sha256"),
                     "docs359_sha256": sha(DOCS359)},
        "canonical_namespace_before": before, "canonical_namespace_after": after,
        "execution": {"engine_main": False, "launcher_authored": False,
                      "launcher_executed": False, "eda": False, "vcs": False,
                      "attempt_created": False},
    }
    OUTPUT.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result_payload["status"], "checks": checks,
                      "attacks": len(attacks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
