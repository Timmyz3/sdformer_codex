#!/usr/bin/env python3
"""Independent source-only engineering QA for M1583.

The exact M1583 module is imported, but its captured actual worker is never
called.  All behavioral checks use in-memory witnesses passed to
``_build_one_shot``.  No payload, GPU, capture, production, RTL, or EDA path is
available here.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "system_simulator/scripts/build_m1583_ep34_decoder_one_process_one_config_source.py"
TEST = HW / "system_simulator/tests/test_m1583_ep34_decoder_one_process_one_config_source.py"
M1573 = HW / "system_simulator/scripts/build_m1573_ep34_decoder_fresh_worker_gate_successor_source.py"
M1577 = HW / "reviews/m1577_m1573_decoder_fresh_worker_gate_successor_independent_hammer_r1_20260901"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    SOURCE: "f92c91f0a6f3a3d79e53ec232fee339ead72edcf14d22a2d51e6f9e86e3f48c4",
    TEST: "77488d0b285918b4d05bc367b8f12fbcefacbdeab4ff4b4a36556a34be98e04d",
    M1573: "f26203424c4034230ee696ecf3b6d95685ed21647f41eb0c38b6961f0c83d02c",
    M1577 / "review.json": "cbc8dbd19d56584c09a2a54f017415e0409975a5bb2cfbe673227fddcff5a131",
    M1577 / "SHA256SUMS": "4c05456cbe119aa4fdc1372af9b056103c9a621426399370b1a7b49e9b778b8f",
    M1577 / "SHA256SUMS.seal.sha256": "730f86346be93ca9d390896b9e422e4956c3cfd9e96c93eeb8acb88f165166e5",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


for _path, _expected in EXPECTED.items():
    require(_path.is_file() and not _path.is_symlink() and
            digest(_path) == _expected, "identity drift: " + str(_path))
require((M1577 / "SHA256SUMS.seal.sha256").read_text(
    encoding="ascii").split() == [EXPECTED[M1577 / "SHA256SUMS"], "SHA256SUMS"],
    "M1577 outer seal content drift")
for _row in (M1577 / "SHA256SUMS").read_text(encoding="ascii").splitlines():
    _member_digest, _member_name = _row.split(None, 1)
    require(digest(M1577 / _member_name.strip()) == _member_digest,
            "M1577 member seal drift: " + _member_name)

SPEC = importlib.util.spec_from_file_location("m1592_exact_m1583", str(SOURCE))
require(SPEC is not None and SPEC.loader is not None, "cannot import M1583")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def clean_row(config="DENSE_TYPED_K8"):
    return {
        "configuration": config,
        "resource_manifest_sha256": M.RESOURCE_SHA256,
        "total_cycles": 101,
        "request_count": 3,
        "kind_counts": {"compute": 2, "commit": 1},
        "byte_counts": {"compute": 288, "commit": 3},
        "transaction_address_sha256": "a" * 64,
        "commit_sequence_sha256": "b" * 64,
        "streaming": {"materialized_transaction_list": False,
                      "destinations": 7, "timesteps": 10},
        "schema": "upstream",
        "pilot_call_ordinal": 0,
        "module_ordinal": 0,
        "timesteps": 10,
        "diagnostic_only": True,
        "paper_result": False,
        "product_capture": False,
        "production": False,
        "payload_fd_sha256": "c" * 64,
        "payload_fd_size": 4096,
        "m1573_rss": {"gate_calls": 2,
                      "baseline_current_rss_kib": 100,
                      "baseline_peak_rss_kib": 120,
                      "max_current_rss_kib": 130,
                      "max_peak_rss_kib": 140,
                      "absolute_limit_kib": M.RSS_LIMIT_KIB,
                      "fresh_exec_required": True},
        "fresh_exec_required": True,
    }


def must_reject(function):
    try:
        function()
    except (M.M1583Error, AssertionError, AttributeError, KeyError,
            RuntimeError, TypeError, ValueError):
        return True
    raise AssertionError("invalid witness accepted")


def run(output):
    attacks = []

    def attack(name, function):
        require(function() is not False, "attack returned false: " + name)
        attacks.append(name)

    source_text = SOURCE.read_text(encoding="utf-8")
    test_text = TEST.read_text(encoding="utf-8")
    description = M.describe()
    require(description["claim_boundary"]["actual_execution"] is False and
            description["claim_boundary"]["cycles"] is False and
            description["fresh_interpreter_per_configuration"] is True and
            description["one_call_token_consumed_before_payload"] is True,
            "description boundary drift")
    require("--actual" not in source_text and "--pilot" not in source_text and
            "one_shot_worker_entry = _build_one_shot(U.fresh_worker_entry)" in
            source_text, "source capability drift")

    # Inspect the closure cell without invoking the captured actual entry.
    names = M.one_shot_worker_entry.__code__.co_freevars
    cells = dict(zip(names, [cell.cell_contents
                             for cell in M.one_shot_worker_entry.__closure__]))
    require("bound_entry" in cells and cells["bound_entry"] is M.U.fresh_worker_entry,
            "one-shot closure did not capture exact M1573 entry")
    captured = cells["bound_entry"]
    original_attribute = M.U.fresh_worker_entry
    sentinel = lambda _config: (_ for _ in ()).throw(
        AssertionError("mutable M1573 attribute consulted"))
    M.U.fresh_worker_entry = sentinel
    try:
        names_after = M.one_shot_worker_entry.__code__.co_freevars
        cells_after = dict(zip(names_after, [cell.cell_contents for cell in
                                             M.one_shot_worker_entry.__closure__]))
        require(cells_after["bound_entry"] is captured and captured is not sentinel,
                "captured M1573 entry followed mutable module attribute")
    finally:
        M.U.fresh_worker_entry = original_attribute

    calls = []
    worker = M._build_one_shot(
        lambda config: calls.append(config) or clean_row(config))
    require(worker("DENSE_TYPED_K8")["configuration"] == "DENSE_TYPED_K8",
            "clean one-shot rejected")
    attack("second_configuration_before_bound_entry", lambda: must_reject(
        lambda: worker("BIT_TYPED_K8")))
    require(calls == ["DENSE_TYPED_K8"], "second configuration reached bound entry")

    calls = []
    worker = M._build_one_shot(
        lambda config: calls.append(config) or clean_row(config))
    attack("product_before_bound_entry", lambda: must_reject(
        lambda: worker(M.FORBIDDEN_CONFIG)))
    require(calls == [], "product reached bound entry")
    require(worker("BIT_TYPED_K8")["configuration"] == "BIT_TYPED_K8" and
            calls == ["BIT_TYPED_K8"], "rejected product consumed valid token")

    calls = []

    def failing_entry(config):
        calls.append(config)
        raise RuntimeError("synthetic upstream failure")

    worker = M._build_one_shot(failing_entry)
    attack("first_entry_failure", lambda: must_reject(
        lambda: worker("DENSE_TYPED_K8")))
    attack("token_consumed_before_failed_entry", lambda: must_reject(
        lambda: worker("BIT_TYPED_K8")))
    require(calls == ["DENSE_TYPED_K8"], "failed first entry left token reusable")

    required = {
        "configuration", "resource_manifest_sha256", "total_cycles",
        "request_count", "kind_counts", "byte_counts",
        "transaction_address_sha256", "commit_sequence_sha256", "streaming",
        "schema", "pilot_call_ordinal", "module_ordinal", "timesteps",
        "diagnostic_only", "paper_result", "product_capture", "production",
        "payload_fd_sha256", "payload_fd_size", "m1573_rss",
        "fresh_exec_required",
    }
    for field in sorted(required):
        row = clean_row(); del row[field]
        attack("missing_" + field, lambda row=row: must_reject(
            lambda: M.validate_result("DENSE_TYPED_K8", row)))

    mutations = [
        ("configuration", "BIT_TYPED_K8"),
        ("resource_manifest_sha256", "d" * 64),
        ("total_cycles", 0),
        ("request_count", 0),
        ("transaction_address_sha256", "x" * 64),
        ("transaction_address_sha256", "a" * 63),
        ("commit_sequence_sha256", "x" * 64),
        ("commit_sequence_sha256", "b" * 65),
        ("payload_fd_sha256", "x" * 64),
        ("payload_fd_sha256", "c" * 63),
        ("payload_fd_size", 0),
        ("pilot_call_ordinal", 1),
        ("module_ordinal", 1),
        ("timesteps", 9),
        ("diagnostic_only", False),
        ("paper_result", True),
        ("product_capture", True),
        ("production", True),
        ("fresh_exec_required", False),
    ]
    for ordinal, pair in enumerate(mutations):
        field, value = pair
        row = clean_row(); row[field] = value
        attack("field_mutation_{:02d}_{}".format(ordinal, field),
               lambda row=row: must_reject(
                   lambda: M.validate_result("DENSE_TYPED_K8", row)))

    row = clean_row(); row["kind_counts"]["commit"] = 2
    attack("request_kind_conservation", lambda: must_reject(
        lambda: M.validate_result("DENSE_TYPED_K8", row)))
    row = clean_row(); row["kind_counts"] = {"compute": -1, "commit": 4}
    attack("negative_kind_count", lambda: must_reject(
        lambda: M.validate_result("DENSE_TYPED_K8", row)))
    row = clean_row(); row["byte_counts"] = {"compute": -1}
    attack("negative_byte_count", lambda: must_reject(
        lambda: M.validate_result("DENSE_TYPED_K8", row)))
    row = clean_row(); row["streaming"]["materialized_transaction_list"] = True
    attack("materialized_transaction_list", lambda: must_reject(
        lambda: M.validate_result("DENSE_TYPED_K8", row)))
    row = clean_row(); row["streaming"]["destinations"] = 0
    attack("zero_destinations", lambda: must_reject(
        lambda: M.validate_result("DENSE_TYPED_K8", row)))
    row = clean_row(); row["streaming"]["timesteps"] = 9
    attack("streaming_timestep_drift", lambda: must_reject(
        lambda: M.validate_result("DENSE_TYPED_K8", row)))

    rss_mutations = [
        ("gate_calls", 0),
        ("absolute_limit_kib", M.RSS_LIMIT_KIB - 1),
        ("fresh_exec_required", False),
        ("baseline_current_rss_kib", M.RSS_LIMIT_KIB),
        ("baseline_peak_rss_kib", M.RSS_LIMIT_KIB),
        ("max_current_rss_kib", M.RSS_LIMIT_KIB),
        ("max_peak_rss_kib", M.RSS_LIMIT_KIB),
        ("baseline_current_rss_kib", -1),
        ("baseline_peak_rss_kib", -1),
    ]
    for ordinal, pair in enumerate(rss_mutations):
        field, value = pair
        row = clean_row(); row["m1573_rss"][field] = value
        attack("rss_mutation_{:02d}_{}".format(ordinal, field),
               lambda row=row: must_reject(
                   lambda: M.validate_result("DENSE_TYPED_K8", row)))
    row = clean_row(); row["m1573_rss"]["max_current_rss_kib"] = 99
    attack("rss_current_monotonicity", lambda: must_reject(
        lambda: M.validate_result("DENSE_TYPED_K8", row)))
    row = clean_row(); row["m1573_rss"]["max_peak_rss_kib"] = 119
    attack("rss_peak_monotonicity", lambda: must_reject(
        lambda: M.validate_result("DENSE_TYPED_K8", row)))

    require(M.validate_result("DENSE_TYPED_K8", clean_row())[
                "resource_manifest_sha256"] == M.RESOURCE_SHA256,
            "clean result rejected")
    require("test_02_one_shot_token_precedes_second_call" in test_text and
            "test_03_product_rejected_before_bound_entry" in test_text and
            "test_05_request_conservation_rejected" in test_text,
            "author unit test surface drift")

    result = {
        "schema": "m1592_m1583_decoder_engineering_qa_runtime_r1_v1",
        "status": "PASS_M1592_M1583_SOURCE_ENGINEERING_QA__RUNNER_SOURCE_AUTHORING_ONLY__NO_ACTUAL",
        "runtime": {"implementation": sys.implementation.name,
                    "version": ".".join(str(value) for value in
                                          sys.version_info[:3])},
        "identity": {"m1583_source_sha256": EXPECTED[SOURCE],
                     "m1583_test_sha256": EXPECTED[TEST],
                     "m1573_source_sha256": EXPECTED[M1573],
                     "m1577_review_sha256": EXPECTED[M1577 / "review.json"],
                     "docs359_sha256": EXPECTED[DOCS359]},
        "closure": {"captures_exact_m1573_entry": True,
                    "captured_entry_immune_to_attribute_replacement": True,
                    "actual_entry_called": False},
        "ordering": {"second_config_rejected_before_call": True,
                     "product_rejected_before_call": True,
                     "token_consumed_before_failed_entry": True},
        "result_gate": {"clean_row": True,
                        "required_fields_removed_and_rejected": len(required),
                        "request_kind_conservation": True,
                        "three_digests_hex64": True,
                        "rss_gate_calls_positive": True,
                        "strict_current_peak_below_8gib": True,
                        "rss_equality_boundary_rejected": True},
        "attacks": {"count": len(attacks), "passed": len(attacks),
                    "names": attacks},
        "authorization": {"independent_process_runner_source_authoring": True,
                          "actual_execution": False, "payload": False,
                          "gpu": False, "eda": False},
    }
    output = Path(output)
    require(not output.exists(), "output exists")
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(result["status"] + " attacks={}".format(len(attacks)))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
