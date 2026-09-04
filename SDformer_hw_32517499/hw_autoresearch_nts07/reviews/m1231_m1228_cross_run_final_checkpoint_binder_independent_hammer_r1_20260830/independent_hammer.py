#!/opt/anaconda3/envs/python310/bin/python3.10
"""Independent synthetic hammer for M1228. Never reads production candidates."""

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py"
TEST = ROOT / "hw_autoresearch_nts07/tests/test_build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py"
CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1228_motion_cross_run_final_checkpoint_rebind_binder_source_contract_r1_20260830.json"
AUTHOR = ROOT / "hw_autoresearch_nts07/reviews/m1228_motion_cross_run_final_checkpoint_rebind_binder_source_author_r1_20260830"
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_sidecar(path):
    first = path.with_name(path.name + ".sha256")
    outer = path.with_name(path.name + ".sha256.seal.sha256")
    expected, name = first.read_text(encoding="utf-8").split()
    assert name == path.name and expected == sha(path)
    expected, name = outer.read_text(encoding="utf-8").split()
    assert name == first.name and expected == sha(first)
    return {"payload_sha256": sha(path), "sidecar_sha256": sha(first),
            "outer_file_sha256": sha(outer)}


def verify_recursive(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    expected, name = outer.read_text(encoding="utf-8").split()
    assert name == "SHA256SUMS" and expected == sha(manifest)
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("*")
        member = root / name
        assert member.is_file() and not member.is_symlink() and sha(member) == digest
        assert name not in rows
        rows[name] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    assert actual == set(rows)
    return {"manifest_sha256": sha(manifest), "outer_file_sha256": sha(outer),
            "members": rows}


M = load("m1231_m1228_source", SOURCE)
T = load("m1231_m1228_test_fixture", TEST)


def fixture():
    case = T.M1228CrossRunBinderTest("test_cross_run_selection_and_selected_config_are_bound")
    case.setUp()
    return case


def expect_error(label, mutate):
    case = fixture()
    try:
        mutate(case)
        try:
            M.build(case.policy)
        except M.BinderError:
            return {"attack": label, "status": "REJECTED"}
        return {"attack": label, "status": "ACCEPTED_UNEXPECTEDLY"}
    finally:
        case.tearDown()


def mutate_profile(case, index, function):
    case._mutate_profile(index, function)


def set_policy_candidates(case, candidates):
    case.policy = replace(case.policy, candidates=tuple(candidates))


def run():
    policy = json.loads(CONTRACT.read_text(encoding="utf-8"))
    identities = {
        "source_sha256": sha(SOURCE), "test_sha256": sha(TEST),
        "contract_sha256": sha(CONTRACT), "docs359_sha256": sha(DOCS359),
    }
    assert identities["source_sha256"] == policy["source_identity"]["builder"]["sha256"]
    assert identities["test_sha256"] == policy["source_identity"]["tests"]["sha256"]
    assert identities["docs359_sha256"] == policy["protected_file"]["sha256"]
    assert policy["scope"]["selection_made_now"] is False
    contract_seal = verify_sidecar(CONTRACT)
    author_seal = verify_recursive(AUTHOR)

    constants = M.PRODUCTION_POLICY.candidates
    assert [row.candidate_id for row in constants] == [
        "legacy_ep29", "resume_ep30", "resume_ep32", "resume_ep34"]
    assert [row.epoch for row in constants] == [29, 30, 32, 34]
    assert len({row.run_dir for row in constants}) == 2
    assert len({row.config for row in constants}) == 2
    assert constants[0].config_sha256 == M.OLD_CONFIG_SHA256
    assert all(row.config_sha256 == M.NEW_CONFIG_SHA256 for row in constants[1:])

    case = fixture()
    try:
        baseline = M.build(case.policy)
        assert len(baseline["candidate_population"]) == 4
        assert baseline["selected"]["candidate_id"] == "resume_ep32"
        assert baseline["selected"]["checkpoint"]["sha256"] == sha(
            case.new_run / "checkpoint_epoch32.pth")
        assert baseline["selected"]["configuration"]["sha256"] == sha(case.new_config)
        assert [row["id"] for row in baseline[
            "e0_e8_activation_dependent_invalidation_and_rebind_targets"]] == [
                "E{}".format(index) for index in range(9)]
        assert all(row["state_after_selection"] != "REUSE_UNCONDITIONALLY" for row in baseline[
            "e0_e8_activation_dependent_invalidation_and_rebind_targets"])
    finally:
        case.tearDown()

    attacks = []
    for index in range(4):
        attacks.append(expect_error("missing_checkpoint_candidate_{}".format(index),
                                    lambda case, i=index: (case.policy.candidates[i].run_dir /
                                      "checkpoint_epoch{}.pth".format(case.policy.candidates[i].epoch)).unlink()))
    attacks.extend([
        expect_error("candidate_population_three", lambda case: set_policy_candidates(
            case, case.policy.candidates[:-1])),
        expect_error("old_new_run_collapsed", lambda case: set_policy_candidates(
            case, (case.policy.candidates[0],) + tuple(
                replace(row, run_dir=case.old_run) for row in case.policy.candidates[1:]))),
        expect_error("old_new_config_collapsed", lambda case: set_policy_candidates(
            case, (case.policy.candidates[0],) + tuple(
                replace(row, config=case.old_config) for row in case.policy.candidates[1:]))),
        expect_error("new_config_sha_drift", lambda case: set_policy_candidates(
            case, (case.policy.candidates[0],
                   replace(case.policy.candidates[1], config_sha256="0" * 64),
                   *case.policy.candidates[2:]))),
        expect_error("legacy_checkpoint_sha_drift", lambda case: set_policy_candidates(
            case, (replace(case.policy.candidates[0], expected_checkpoint_sha256="0" * 64),
                   *case.policy.candidates[1:]))),
        expect_error("samples_bool", lambda case: mutate_profile(
            case, 1, lambda row: row.__setitem__("samples", True))),
        expect_error("load_missing_bool", lambda case: mutate_profile(
            case, 1, lambda row: row["checkpoint_load_audit"].__setitem__("missing_count", False))),
        expect_error("module_count_extra", lambda case: mutate_profile(
            case, 2, lambda row: row["module_counts"].__setitem__("extra", 1))),
        expect_error("aee_nan", lambda case: mutate_profile(
            case, 3, lambda row: row["metrics"].__setitem__("AEE", float("nan")))),
    ])

    case = fixture()
    try:
        for index, value in enumerate((0.5, 0.5, 0.8, 0.9)):
            case._write_profile(case.policy.candidates[index], value)
        tied = M.build(case.policy)
        tie_break = {"status": "PASS", "selected_epoch": tied["selected"]["epoch"]}
        assert tie_break["selected_epoch"] == 29
    finally:
        case.tearDown()

    case = fixture()
    try:
        mutate_profile(case, 3, lambda row: row["metrics"].__setitem__("AEE", -1.0))
        negative = M.build(case.policy)
        negative_attack = {
            "attack": "negative_AEE_is_mathematically_invalid",
            "status": "ACCEPTED_UNEXPECTEDLY",
            "selected_epoch": negative["selected"]["epoch"],
            "selected_AEE": negative["selected"]["accuracy_metrics"]["AEE"],
        }
    finally:
        case.tearDown()

    case = fixture()
    original_strict = M.strict_json
    try:
        target = case._profile_path(case.policy.candidates[2])
        fired = [False]

        def racing_strict(path):
            value = original_strict(path)
            if Path(path) == target and not fired[0]:
                altered = json.loads(json.dumps(value))
                altered["metrics"]["AEE"] = 9.0
                target.write_text(json.dumps(altered, sort_keys=True) + "\n", encoding="utf-8")
                fired[0] = True
            return value

        M.strict_json = racing_strict
        raced = M.build(case.policy)
        on_disk = original_strict(target)
        race_attack = {
            "attack": "profile_parse_to_hash_TOCTOU",
            "status": "ACCEPTED_UNEXPECTEDLY",
            "selected_epoch": raced["selected"]["epoch"],
            "ranked_AEE": raced["selected"]["accuracy_metrics"]["AEE"],
            "actual_recorded_file_AEE": on_disk["metrics"]["AEE"],
            "recorded_profile_sha_matches_actual_file": (
                raced["selected"]["profile"]["sha256"] == sha(target)),
        }
        assert race_attack["recorded_profile_sha_matches_actual_file"] is True
        assert race_attack["ranked_AEE"] != str(race_attack["actual_recorded_file_AEE"])
    finally:
        M.strict_json = original_strict
        case.tearDown()

    rejected = sum(row["status"] == "REJECTED" for row in attacks)
    return {
        "schema": "m1231_m1228_cross_run_final_checkpoint_binder_independent_hammer_r1_v1",
        "status": "FAIL_CLOSED_REVIEW__TWO_P0_SOURCE_DEFECTS__PRODUCTION_RELEASE_NOT_AUTHORIZED",
        "identities": identities,
        "contract_seal": contract_seal,
        "author_seal": {"manifest_sha256": author_seal["manifest_sha256"],
                        "outer_file_sha256": author_seal["outer_file_sha256"],
                        "members": len(author_seal["members"])},
        "baseline": {"status": "PASS", "candidates": 4,
                     "selected_candidate": "resume_ep32", "E0_E8_targets": 9},
        "tie_break": tie_break,
        "independent_fail_closed_attacks": {"total": len(attacks), "rejected": rejected,
                                             "details": attacks},
        "defects": [negative_attack, race_attack],
        "verdict": {
            "production_binder_release_authorized": False,
            "checkpoint_selected_now": False,
            "required_successor_fixes": [
                "require every accuracy error metric, especially AEE, to be finite and nonnegative",
                "parse and hash the same immutable profile byte snapshot, then record that exact SHA",
                "rerun author tests and a fresh different-author hammer"
            ]
        },
        "execution": {"production_paths_accessed": False, "remote": False, "gpu": False,
                      "checkpoint": False, "valid825": False, "eda": False,
                      "selection": False}
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
