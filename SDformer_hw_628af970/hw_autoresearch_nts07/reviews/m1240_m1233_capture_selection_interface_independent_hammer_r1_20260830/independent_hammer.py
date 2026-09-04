#!/opt/anaconda3/envs/python310/bin/python3.10
"""Independent local-only hammer for the M1233 capture-selection interface."""

import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1233_motion_final_checkpoint_unified_hardware_selection_interface_r2.py")
TEST = ROOT / "hw_autoresearch_nts07/tests/test_m1233_motion_final_checkpoint_unified_capture_selection_interface_source.py"
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1233_motion_final_checkpoint_unified_capture_selection_interface_successor_source_contract_r1_20260830.json")
AUTHOR = ROOT / "hw_autoresearch_nts07/reviews/m1233_motion_final_checkpoint_unified_capture_selection_interface_successor_author_r1_20260830"
M1230 = ROOT / "hw_autoresearch_nts07/reviews/m1230_m1227_motion_final_checkpoint_unified_capture_source_hammer_r1_20260830"
M1234_SOURCE = ROOT / "hw_autoresearch_nts07/scripts/build_m1234_motion_cross_run_final_checkpoint_rebind_binder_successor.py"
M1234_CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1234_motion_cross_run_final_checkpoint_rebind_binder_successor_source_contract_r1_20260830.json"
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
        assert member.is_file() and not member.is_symlink()
        assert sha(member) == digest and name not in rows
        rows[name] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    assert actual == set(rows)
    return {"manifest_sha256": sha(manifest), "outer_file_sha256": sha(outer),
            "members": rows}


M = load("m1240_m1233_source", SOURCE)
T = load("m1240_m1233_fixture", TEST)


def fixture():
    case = T.M1233SelectionInterfaceTest("test_03_exact_m1234_shape_passes_and_keyerror_is_regressed")
    case.setUp()
    return case


def rejected(label, mutate_selection=None, mutate_hammer=None):
    case = fixture()
    try:
        selection_entry = case.selection_entry
        hammer_entry = case.hammer_entry
        if mutate_selection is not None:
            value = copy.deepcopy(case.selection)
            mutate_selection(value)
            selection_entry = case._write_selection(value)
        if mutate_hammer is not None:
            value = case._hammer(selection_entry,
                                 (json.loads((case.selection_root / selection_entry[
                                     "selection_member"]).read_text()))["selected"])
            mutate_hammer(value)
            hammer_entry = case._write_hammer(value)
        try:
            M.validate_final_selection(selection_entry, hammer_entry)
        except M.M1233Error:
            return {"attack": label, "status": "REJECTED"}
        return {"attack": label, "status": "ACCEPTED_UNEXPECTEDLY"}
    finally:
        case.tearDown()


def demonstrate_missing_source_hammer_gate():
    case = fixture()
    try:
        with tempfile.TemporaryDirectory(prefix=".m1240_launch_", dir=M.HW / "contracts") as name:
            launch_path = Path(name) / "launch.json"
            launch = {
                "schema": M.LAUNCH_SCHEMA,
                "status": M.LAUNCH_STATUS,
                "contract_path": str(launch_path.relative_to(M.ROOT)),
                "inputs": {
                    "launcher": {"path": str(SOURCE.relative_to(M.ROOT)), "sha256": sha(SOURCE)},
                    "source_contract": {"path": str(CONTRACT.relative_to(M.ROOT)),
                                        "sha256": sha(CONTRACT)},
                    "final_selection_result": case.selection_entry,
                    "final_selection_result_hammer": case.hammer_entry,
                },
                "cohort": {"samples": []},
                "one_shot": {"attempt_marker": "attempt"},
                "output": {"path": "result"},
                "production_log": {"path": "log"},
            }
            launch_path.write_text(json.dumps(launch) + "\n", encoding="utf-8")

            def mapped(value, missing_leaf=False):
                if value == "attempt":
                    return M.CANONICAL_ATTEMPT
                if value == "result":
                    return M.CANONICAL_RESULT
                if value == "log":
                    return M.CANONICAL_LOG
                raise AssertionError("unexpected path " + value)

            with mock.patch.object(M.R1, "validate_m1224", return_value={}), \
                 mock.patch.object(M.R1, "validate_cohort", return_value=[]), \
                 mock.patch.object(M.R1, "safe_repo_path", side_effect=mapped):
                binding = M.validate_launch_contract(launch, launch_path)
            assert "source_hammer" not in launch["inputs"]
            return {"attack": "launch_omits_M1240_source_hammer",
                    "status": "ACCEPTED_UNEXPECTEDLY",
                    "binding_candidate": binding["identity"]["candidate_id"]}
    finally:
        case.tearDown()


def run():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    author = verify_recursive(AUTHOR)
    m1230 = verify_recursive(M1230)
    identities = {
        "source_sha256": sha(SOURCE), "test_sha256": sha(TEST),
        "contract_sha256": sha(CONTRACT), "docs359_sha256": sha(DOCS359),
    }
    assert identities["source_sha256"] == contract["source"]["sha256"]
    assert identities["test_sha256"] == contract["test"]["sha256"]
    assert identities["docs359_sha256"] == contract["docs359_sha256"]
    assert author["members"]["review.json"] == "7e30f47d03097ac67fa542db8b17689d7fb5b03440b83e1265211b5e3725f9e4"
    assert m1230["members"]["review.json"] == contract["m1230_binding"]["review_sha256"]

    m1234_contract = json.loads(M1234_CONTRACT.read_text(encoding="utf-8"))
    assert M.ALLOWED_SELECTION_SCHEMA == m1234_contract["fixed_result_interface"]["schema"]
    assert M.ALLOWED_SELECTION_STATUS == m1234_contract["fixed_result_interface"]["status"]
    m1234_text = M1234_SOURCE.read_text(encoding="utf-8")
    assert M.ALLOWED_SELECTION_SCHEMA in m1234_text
    assert "PASS_M1234_CROSS_RUN_FINAL_CHECKPOINT_SELECTED_R2__" in m1234_text
    assert "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY" in m1234_text

    case = fixture()
    try:
        baseline = M.validate_final_selection(case.selection_entry, case.hammer_entry)
        assert baseline["checkpoint_path"] == case.checkpoint
        assert baseline["config_path"] == case.configuration
        assert baseline["profile_path"] == case.profile
        assert baseline["identity"]["candidate_id"] == "resume_ep32"
        assert baseline["identity"]["epoch"] == 32
    finally:
        case.tearDown()

    attacks = [
        rejected("top_level_config_identical", lambda row: row.__setitem__(
            "configuration", copy.deepcopy(row["selected"]["configuration"]))),
        rejected("top_level_config_conflicting", lambda row: row.__setitem__(
            "configuration", dict(row["selected"]["configuration"], sha256="0" * 64))),
        rejected("selection_schema_drift", lambda row: row.__setitem__("schema", "wrong")),
        rejected("selection_status_drift", lambda row: row.__setitem__("status", "wrong")),
        rejected("selected_extra_key", lambda row: row["selected"].__setitem__("extra", 1)),
        rejected("candidate_epoch_splice", lambda row: row["selected"].__setitem__("epoch", 34)),
        rejected("profile_shape_missing_samples", lambda row: row["selected"]["profile"].pop("samples")),
        rejected("profile_module_count_drift", lambda row: row["selected"]["profile"][
            "module_counts"].__setitem__("ATLIFTernaryPSN", 104)),
        rejected("hammer_schema_drift", mutate_hammer=lambda row: row.__setitem__("schema", "wrong")),
        rejected("hammer_status_drift", mutate_hammer=lambda row: row.__setitem__("status", "wrong")),
        rejected("hammer_not_independent", mutate_hammer=lambda row: row.__setitem__(
            "independence", {"different_author": False})),
        rejected("hammer_production_capture_true", mutate_hammer=lambda row: row[
            "authorization"].__setitem__("production_capture", True)),
        rejected("hammer_selection_sha_splice", mutate_hammer=lambda row: row[
            "selection_authority"].__setitem__("selection_sha256", "0" * 64)),
        rejected("hammer_checkpoint_sha_splice", mutate_hammer=lambda row: row[
            "selection_authority"].__setitem__("selected_checkpoint_sha256", "0" * 64)),
        rejected("hammer_config_sha_splice", mutate_hammer=lambda row: row[
            "selection_authority"].__setitem__("selected_config_sha256", "0" * 64)),
        rejected("hammer_profile_sha_splice", mutate_hammer=lambda row: row[
            "selection_authority"].__setitem__("selected_profile_sha256", "0" * 64)),
    ]
    assert all(row["status"] == "REJECTED" for row in attacks)

    aliases = {
        "EXPECTED_STATIC_COUNTS": M.EXPECTED_STATIC_COUNTS is M.R1.EXPECTED_STATIC_COUNTS,
        "EXPECTED_LIVE_COUNTS": M.EXPECTED_LIVE_COUNTS is M.R1.EXPECTED_LIVE_COUNTS,
        "DEAD_SN_V": M.DEAD_SN_V is M.R1.DEAD_SN_V,
        "audit_call_matrix": M.audit_call_matrix is M.R1.audit_call_matrix,
        "audit_attention_population": M.audit_attention_population is M.R1.audit_attention_population,
        "validate_payload_population": M.validate_payload_population is M.R1.validate_payload_population,
        "atomic_sample_snapshot": M.atomic_sample_snapshot is M.R1.atomic_sample_snapshot,
        "final_validate_and_seal": M.final_validate_and_seal is M.R1.final_validate_and_seal,
    }
    assert all(aliases.values())
    assert sum(M.EXPECTED_STATIC_COUNTS.values()) == 259
    assert sum(M.EXPECTED_LIVE_COUNTS.values()) == 247
    assert len(M.DEAD_SN_V) == 12
    assert contract["unchanged_capture_contract"]["ordered_records"] == 9880
    assert contract["unchanged_capture_contract"]["attention_records"] == 480
    assert contract["unchanged_capture_contract"]["payload_files"] == 640

    launch_defect = demonstrate_missing_source_hammer_gate()
    validate_source = inspect.getsource(M.validate_launch_contract)
    assert "source_hammer" not in validate_source and "m1240" not in validate_source.lower()
    assert contract["future_release"]["source_hammer_required"] is True

    return {
        "schema": "m1240_m1233_capture_selection_interface_independent_hammer_r1_v1",
        "status": "FAIL_P0_SOURCE_HAMMER_NOT_CONSUMED__PRODUCTION_RELEASE_NOT_AUTHORIZED",
        "identities": identities,
        "author_seal": {"manifest_sha256": author["manifest_sha256"],
                        "outer_file_sha256": author["outer_file_sha256"],
                        "members": len(author["members"])},
        "m1230_seal": {"manifest_sha256": m1230["manifest_sha256"],
                       "outer_file_sha256": m1230["outer_file_sha256"]},
        "M1230_P0_closure": {
            "exact_M1234_shape": "PASS", "top_level_config_splice": "REJECTED",
            "same_selected_checkpoint_config_profile": "PASS",
            "fixed_M1234_schema_status": "PASS",
            "M1237_double_seal_cross_SHA_authority": "PASS",
        },
        "capture_aliases": aliases,
        "capture_population": {"static": 259, "live": 247, "dead_sn_v": 12,
                               "ordered": 9880, "attention": 480, "payload": 640,
                               "atomic_snapshot_alias": True},
        "independent_mutations": {"total": len(attacks), "rejected": len(attacks),
                                  "details": attacks},
        "blocking_defect": launch_defect,
        "verdict": {
            "source_hammer_pass": False,
            "production_release_authorized": False,
            "required_successor": (
                "launch contract must bind its fresh different-author source-hammer review path/"
                "review SHA/manifest SHA/outer-file SHA and validate exact schema/status/"
                "source-contract-test cross-SHAs plus explicit production-capture authorization")
        },
        "execution": {"remote": False, "gpu": False, "checkpoint": False,
                      "capture": False, "release": False, "eda": False,
                      "production_paths": False}
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
