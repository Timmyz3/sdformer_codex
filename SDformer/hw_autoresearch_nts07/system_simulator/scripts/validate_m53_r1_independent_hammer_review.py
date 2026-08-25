#!/usr/bin/env python3
"""Fail-closed validator for the independent M53-r1 hammer review."""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
REVIEW_DIR = HW_ROOT / "results/m53_r1_independent_hammer_20260823"
REVIEW = REVIEW_DIR / "m53_r1_independent_hammer_review.json"
EXPECTED = {
    "review": "815899991fd3dd9d16302c3d31469d74dfb45175c5f5007fd21fd385fe91bb6f",
    "contract": "e1dd6eb10a4b580115ff8cfe9d28605167256dfe81942ea2e2ea92d5fba88e03",
    "analyzer": "638809bd72ab7f66fc69b51f4cb726f2c0d1c7712f71188066b4ef04cbdda531",
    "result": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "producer_validator": "8c8a410db277f3f431a22d8f9db62aa7897952eca268e1fc8c28d78834e3b5d4",
    "checker": "96f262d922722fe14c71652fb83253810fa1328ba42d11a008e1c7a6d12725ba",
    "reconstruction": "42afea8ed8e92f0689a5ed7512ad8415ed3ee1bbfb2a2583b2d6c904e87e23e9",
    "attack_runner": "10752bf648ea48fc433294885729112072ae05f10ad5ec063cd71ee32c8cabdf",
    "attack_receipt": "10dbad31e13f8681d8e4c69fa30d6f09892cf8272e8bdd7ffac7d441ce7567dc",
    "durable": "344ae1f777e0640d46b19118f0b6d451465046350d68a9f33b1faae124747bb4",
    "m54_review": "5b1f66e8e0c8e235984adb1d3fd2ecf9680a8bb1e062973093c00d9190da2393",
    "m54_validator": "0fac4ccd4d03f16802f7fc7dee2024d04a4a653225cc595aef9d83a19d7b51cf",
    "m54_receipt": "7649e41ab4c622ca52c33f0e7af32a6736489e1bb91df66c014d22ee8b600f5c",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: {}".format(raw))

    def pairs_hook(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: {}".format(key))
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def resolve_hw(row):
    return (HW_ROOT / row["path"]).resolve()


def run(command, label):
    result = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)
    require(result.returncode == 0,
            "{} failed rc={} stdout={} stderr={}".format(
                label, result.returncode, result.stdout[-3000:],
                result.stderr[-3000:]))
    return result


def validate(rerun):
    require(REVIEW.is_file() and sha256_path(REVIEW) == EXPECTED["review"],
            "review SHA mismatch")
    review = strict_json(REVIEW)
    require(review["schema"] == "m53_r1_independent_hammer_review_v1" and
            review["status"] ==
            "PASS_INDEPENDENT_M53_LEDGER_GO_TRANSACTION_DSE_ONLY_WITH_INTEGRATION_P1" and
            review["verdict"] ==
            "GO_M53_TRANSACTION_DSE_ONLY_M54_STANDALONE_CLOSES_FIFO_TAG_NO_GO_M53_RTL_OR_SYSTEM_CYCLES" and
            review["score_0_to_100"] == 86 and
            review["severity_counts"] == {"P0": 0, "P1": 1, "P2": 6},
            "review score/verdict/severity drift")
    require([row["id"] for row in review["findings"]] == [
        "M53-P1-01", "M53-P2-01", "M53-P2-02", "M53-P2-03",
        "M53-P2-04", "M53-P2-05", "M53-P2-06"],
        "finding order/identity drift")
    p1 = review["findings"][0]
    require(p1["severity"] == "P1" and
            "generic standalone K4-C16 RTL scope" in p1["finding"] and
            "M53 maximum_metadata_occupancy remains a clamped ready-window proxy"
            in p1["impact"] and
            "accepted request/response/output handshakes" in p1["required_repair"],
            "M53/M54 P1 boundary drift")

    producer_expected = {
        "contract": EXPECTED["contract"], "analyzer": EXPECTED["analyzer"],
        "result": EXPECTED["result"],
        "producer_validator": EXPECTED["producer_validator"],
    }
    for name, row in review["producer_anchors"].items():
        path = resolve_hw(row)
        require(path.is_file() and row["sha256"] == producer_expected[name] and
                sha256_path(path) == row["sha256"],
                "producer anchor mismatch: {}".format(name))
    independent_expected = {
        "reconstruction_checker": EXPECTED["checker"],
        "reconstruction": EXPECTED["reconstruction"],
        "attack_runner": EXPECTED["attack_runner"],
        "attack_receipt": EXPECTED["attack_receipt"],
        "durable_all3_rerun": EXPECTED["durable"],
    }
    for name, row in review["independent_evidence"].items():
        path = resolve_hw(row)
        require(path.is_file() and row["sha256"] == independent_expected[name] and
                sha256_path(path) == row["sha256"],
                "independent anchor mismatch: {}".format(name))
    m54_expected = {
        "review": EXPECTED["m54_review"],
        "validator": EXPECTED["m54_validator"],
        "validation_receipt": EXPECTED["m54_receipt"],
    }
    for name, row in review["m54_independent_anchors"].items():
        path = resolve_hw(row)
        require(path.is_file() and row["sha256"] == m54_expected[name] and
                sha256_path(path) == row["sha256"],
                "M54 anchor mismatch: {}".format(name))

    reconstruction = strict_json(resolve_hw(
        review["independent_evidence"]["reconstruction"]))
    require(reconstruction["status"] ==
            "PASS_INDEPENDENT_M53_ALL10_LEDGER_AND_RAW_PARENT_RECONSTRUCTION" and
            reconstruction["independent_mismatch_count"] == 0 and
            reconstruction["scope"] == {
                "configuration_record_rows_reaggregated": 120,
                "sample_ledgers_reaggregated": 30,
                "raw_parent_records_independently_decoded": 40,
                "m52_spatial_records_exactly_reconciled": 40,
                "durable_all3_rerun_byte_exact": True,
                "dc_run_performed": False,
                "open_source_hdl_tools_run": False,
                "m53_cycles_admitted_as_rtl_or_system": False,
            }, "reconstruction scope/status drift")
    configs = dict((row["name"], row)
                   for row in reconstruction["configurations"])
    require(set(configs) == {"K2_CTX16_TEMPORAL", "K4_CTX16_SPATIAL",
                             "K4_CTX16_TEMPORAL"},
            "reconstruction configuration set drift")
    require((configs["K2_CTX16_TEMPORAL"]["aggregate_source_only_cycles"],
             configs["K2_CTX16_TEMPORAL"]["aggregate_integrated_cycles"],
             configs["K2_CTX16_TEMPORAL"]["integrated_p95_nearest_rank"]) ==
            (83847720, 90755624, 9192368) and
            (configs["K4_CTX16_SPATIAL"]["aggregate_source_only_cycles"],
             configs["K4_CTX16_SPATIAL"]["aggregate_integrated_cycles"],
             configs["K4_CTX16_SPATIAL"]["integrated_p95_nearest_rank"]) ==
            (70821488, 81921184, 8376280) and
            (configs["K4_CTX16_TEMPORAL"]["aggregate_source_only_cycles"],
             configs["K4_CTX16_TEMPORAL"]["aggregate_integrated_cycles"],
             configs["K4_CTX16_TEMPORAL"]["integrated_p95_nearest_rank"]) ==
            (68847096, 79869808, 8139624),
            "reconstructed frozen metrics drift")
    raw = reconstruction["raw_parent_reconstruction"]
    require(raw == {
        "record_mismatch_count": 0,
        "previous_timestep_choices_at_timestep_zero": 0,
        "previous_timestep_choices_after_timestep_zero": 301274,
        "previous_timestep_dag_boundaries": 360,
        "output_block_expanded_boundaries": 2880,
    }, "raw parent/DAG reconstruction drift")
    require(reconstruction["storage_capacity_reconstruction"] == {
        "frame_bytes": 68400, "frame_count": 2,
        "two_frame_bytes": 136800, "third_frame_bytes": 0,
        "combined_capacity_bytes": 176688, "headroom_bytes": 17040,
        "minimum_headroom_bytes": 16384, "headroom_surplus_bytes": 656,
        "headroom_unit": "bytes",
    }, "storage/capacity reconstruction drift")
    require(reconstruction["m54_scope_distinction"] == {
        "generic_standalone_real_response_fifo_closed_in_vcs": True,
        "generic_standalone_finite_context_tag_lifecycle_closed_in_vcs": True,
        "m53_adaptive_temporal_parent_arithmetic_state_integrated_in_m54": False,
        "m53_all10_transaction_cycles_replayed_in_m54_rtl": False,
        "m53_response_fifo_event_ledger_field_remains_unadmitted": True,
        "m53_system_cycles_or_speedup_admitted": False,
    }, "M54 closure distinction drift")

    attacks = strict_json(resolve_hw(
        review["independent_evidence"]["attack_receipt"]))
    expected_attacks = [
        "aggregate_source_increment", "sample_integrated_increment",
        "p95_nearest_rank_increment", "previous_timestep_choice_removed",
        "illegal_t0_previous_inserted", "headroom_changed",
        "headroom_unit_bits", "third_frame_added",
        "fifo_event_ledger_promoted", "temporal_rtl_promoted",
        "system_speedup_promoted", "conditional_ratio_promoted",
    ]
    require(attacks["status"] ==
            "PASS_BASELINE_AND_FAIL_CLOSED_12_OF_12_MUTATIONS" and
            attacks["attack_count"] == attacks["rejected_count"] == 12 and
            [row["name"] for row in attacks["attacks"]] == expected_attacks and
            all(row["rejected"] is True for row in attacks["attacks"]) and
            attacks["producer_files_modified"] is False and
            attacks["dc_run_performed"] is False and
            attacks["open_source_hdl_tools_run"] is False,
            "independent attack receipt drift")
    require(review["review_scope"] == {
        "durable_all_three_configuration_rerun": "PASS_BYTE_EXACT",
        "independent_all10_record_sample_percentile_reconstruction":
            "PASS_120_RECORDS_30_SAMPLES_ZERO_MISMATCH",
        "independent_raw_parent_reconstruction":
            "PASS_40_RECORDS_SPATIAL_AND_TEMPORAL_ZERO_MISMATCH",
        "m52_spatial_baseline_reconciliation": "PASS_40_RECORDS_ZERO_MISMATCH",
        "mutation_attacks": "PASS_12_OF_12_REJECTED",
        "producer_files_modified": False,
        "dc_run_performed": False,
        "open_source_hdl_tools_run": False,
    }, "review scope drift")
    forbidden = " ".join(review["not_admitted"])
    for token in ("M53", "RTL", "system", "1.0256839981385708x",
                  "3.0849138159980614x", "DC", "DATE", "best-paper"):
        require(token in forbidden, "missing claim-boundary token: {}".format(token))

    if rerun:
        producer_validator = resolve_hw(
            review["producer_anchors"]["producer_validator"])
        producer = run(["/usr/bin/python3.6", str(producer_validator), "--rerun"],
                       "producer validator/durable rerun")
        require("rerun_byte_identical=True" in producer.stdout,
                "producer durable-rerun marker missing")
        with tempfile.TemporaryDirectory(prefix="m53_review_rerun_") as directory:
            temp = Path(directory)
            checker_output = temp / "reconstruction.json"
            checker = resolve_hw(
                review["independent_evidence"]["reconstruction_checker"])
            run([sys.executable, str(checker),
                 "--output", str(checker_output)],
                "independent raw/ledger reconstruction rerun")
            require(sha256_path(checker_output) == EXPECTED["reconstruction"],
                    "independent reconstruction rerun SHA mismatch")
            attack_output = temp / "attacks.json"
            attack_runner = resolve_hw(
                review["independent_evidence"]["attack_runner"])
            run([sys.executable, str(attack_runner),
                 "--output", str(attack_output)], "independent attack rerun")
            require(sha256_path(attack_output) == EXPECTED["attack_receipt"],
                    "independent attack rerun SHA mismatch")

    return {
        "schema": "m53_r1_independent_hammer_validator_result_v1",
        "status": "PASS_M53_R1_INDEPENDENT_HAMMER_REVIEW_VALIDATED",
        "review_sha256": sha256_path(REVIEW),
        "score_0_to_100": 86,
        "severity_counts": {"P0": 0, "P1": 1, "P2": 6},
        "verdict": review["verdict"],
        "rerun": bool(rerun),
        "independent_mismatch_count": 0,
        "mutation_attacks_rejected": 12,
        "m54_standalone_fifo_tag_closed": True,
        "m53_all10_rtl_integration_closed": False,
        "producer_files_modified": False,
        "dc_run_performed": False,
        "open_source_hdl_tools_run": False,
        "m53_cycles_admitted_as_rtl_or_system": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = validate(args.rerun)
    if args.output is not None:
        require(not args.output.exists(), "refusing validator receipt overwrite")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print("PASS M53-r1 independent review score=86 P0=0 P1=1 P2=6 "+
          "GO transaction DSE only; NO-GO M53 RTL/system cycles")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M53 independent review: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
