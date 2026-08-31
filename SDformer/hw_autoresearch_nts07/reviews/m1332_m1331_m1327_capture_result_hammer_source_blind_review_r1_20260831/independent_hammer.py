#!/usr/bin/env python3
"""Different-author blind hammer for sealed M1331 result-hammer source."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/hammer_m1331_m1327_final_ep34_capture_result_source.py"
TEST = HW / "tests/test_hammer_m1331_m1327_final_ep34_capture_result_source.py"
CONTRACT = HW / "contracts/m1331_m1327_final_ep34_capture_result_hammer_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1331_m1327_final_ep34_capture_result_hammer_source_author_r1_20260831"
EXPECTED = {
    SOURCE: "44297a2225be726d56b5769ef536458148933f489e1ea8c318dde779afbff5b1",
    TEST: "a443885767d955a79962e0ee2509fecc9aa0cc6e15601029beb39a05a180679a",
    CONTRACT: "57a779d27f8bdec7afae7f8a72aa8142badfb3dc49bd72fbc56f965cce3d145a",
    AUTHOR / "SHA256SUMS": "4400ba98425f4cacf6bfd0121839034c057048e8961dd543af70d7a7937790f7",
    AUTHOR / "SHA256SUMS.seal.sha256": "fda47ca9c4bb2c08f0aaac569777b598bd7e2e8883ef278c0ce68a7550cee687",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def main():
    checks = []
    for path, digest in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                "sealed author identity drift")
    for line in (AUTHOR / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split(None, 1)
        require(sha(AUTHOR / relative.lstrip("*")) == digest, "author member drift")
    require((AUTHOR / "SHA256SUMS.seal.sha256").read_text().split() ==
            [sha(AUTHOR / "SHA256SUMS"), "SHA256SUMS"], "author outer drift")
    checks.append("exact_author_graph")

    before = os.path.lexists(str(HW /
        "results/m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831"))
    require(not before, "canonical result unexpectedly exists before blind hammer")
    author = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", "-m", "unittest", "-q", str(TEST)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    require(author.returncode == 0 and "OK" in author.stdout, "author tests failed")
    selfcheck = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(SOURCE), "--source-self-check"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    require(selfcheck.returncode == 0 and "PASS_M1331_SOURCE_SELF_CHECK" in selfcheck.stdout,
            "source self-check failed")
    missing = subprocess.run(
        ["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(SOURCE),
         "--validate-canonical-result"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    require(missing.returncode != 0 and "does not yet exist" in missing.stdout,
            "missing canonical result did not fail closed")
    require(not os.path.lexists(str(HW /
        "results/m1327_motion_ep34_unified_hardware_capture_s40_r1_20260831")),
            "blind hammer created canonical result")
    checks.extend(["author_10_of_10", "source_self_check",
                   "missing_canonical_fail_closed", "canonical_never_created_or_read"])

    M = load("m1332_target", SOURCE)
    T = load("m1332_author_fixture", TEST)

    fixture = T.Fixture()
    try:
        result = M.validate_result(fixture.root)
        require(result["status"] == "PASS_M1331_M1327_EP34_CAPTURE_RESULT",
                "positive fixture failed")
        first = json.loads((fixture.root / "unified_ordered_records.jsonl").read_text().splitlines()[0])
        require("global_order" not in first and first == {
            "category": "fixture", "global_sample_id": 0, "name": "module.0"},
            "author fixture unexpectedly binds real ordered identity")
        attention = json.loads((fixture.root / "attention_qk/manifest.json").read_text())
        require(attention["records"][0] == {"i": 0},
                "author fixture unexpectedly binds attention identity")
        checks.append("positive_fixture_is_invented_inventory_without_global_order_or_attention_identity")
    finally:
        fixture.close()

    false_negatives = []
    fixture = T.Fixture()
    try:
        (fixture.root / "broken_unsealed_symlink").symlink_to("missing_target")
        T.seal(fixture.root)
        M.validate_result(fixture.root)
        false_negatives.append("extra_broken_symlink_ignored_by_recursive_seal")
    finally:
        fixture.close()

    fixture = T.Fixture()
    try:
        del fixture.manifest["identity"]["checkpoint_load_audit"]["missing_count"]
        del fixture.manifest["identity"]["checkpoint_load_audit"]["unexpected_count"]
        T.write_json(fixture.root / "manifest.json", fixture.manifest); T.seal(fixture.root)
        M.validate_result(fixture.root)
        false_negatives.append("missing_checkpoint_load_audit_keys_default_to_zero")
    finally:
        fixture.close()

    require(false_negatives == [
        "extra_broken_symlink_ignored_by_recursive_seal",
        "missing_checkpoint_load_audit_keys_default_to_zero"],
        "false-negative reproduction drift")

    # Control mutations requested by the review brief still reject.
    controls = []
    fixture = T.Fixture()
    try:
        fixture.admission["ordered"] = 9879
        T.write_json(fixture.root / "m1227_admission.json", fixture.admission); T.seal(fixture.root)
        try: M.validate_result(fixture.root)
        except M.M1331Error: controls.append("admission_count")
    finally: fixture.close()
    fixture = T.Fixture()
    try:
        fixture.manifest["cohort"]["samples"][0]["sha256"] = "0" * 64
        T.write_json(fixture.root / "manifest.json", fixture.manifest); T.seal(fixture.root)
        try: M.validate_result(fixture.root)
        except M.M1331Error: controls.append("cohort_sha")
    finally: fixture.close()
    fixture = T.Fixture()
    try:
        fixture.manifest["identity"]["selection"]["selected"]["epoch"] = 35
        T.write_json(fixture.root / "manifest.json", fixture.manifest); T.seal(fixture.root)
        try: M.validate_result(fixture.root)
        except M.M1331Error: controls.append("epoch")
    finally: fixture.close()
    require(controls == ["admission_count", "cohort_sha", "epoch"],
            "control mutation rejection drift")
    checks.append("count_cohort_epoch_controls_reject")

    print(json.dumps({
        "schema": "m1332_m1331_capture_result_hammer_blind_output_r1_v1",
        "status": "FAIL_DO_NOT_CITE__ORDERED_ATTENTION_AND_RECURSIVE_SEAL_FALSE_NEGATIVES",
        "checks_passed": checks,
        "false_negatives": false_negatives + [
            "invented_247_module_inventory_without_global_order_is_accepted",
            "arbitrary_480_attention_records_without_cartesian_or_payload_identity_are_accepted"],
        "authorization": {"additive_successor_source_authoring": True,
                          "production_result_hammer": False, "remote": False,
                          "gpu": False, "capture": False, "eda": False},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
