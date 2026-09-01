#!/usr/bin/env python3
"""M1643 source-only paired evaluator for Motion ep34 S2/CCBS16.

This module is deliberately an in-memory accounting kernel.  It has no file,
payload, GPU, model, simulator, RTL or EDA entry point.  A later, separately
reviewed runner may decode the sealed M1624 forty-sample reduced-binary capture
and a same-cohort cycle baseline, then pass plain Python objects here.

The accounting boundary is strict:

* one decision covers a padded 16-source by 16-output block;
* every positive-epsilon decision costs one uint16 bound (two bytes);
* a dropped block is credited only if its decision precedes every resource it
  claims to suppress;
* saved weight bytes, compute operations and psum bytes are derived only from
  the paired baseline block ledger;
* speedup is ratio-of-sums over exactly forty paired samples;
* epsilon zero bypasses CCBS and must exactly reproduce the current path; and
* if TSBG is admitted later, that admitted path must be the paired baseline.

Python syntax is compatible with CPython 3.6.
"""
from __future__ import print_function

import json
import math
import sys


SCHEMA = "m1643_motion_ep34_s2_ccbs16_paired_evaluation_source_r1_v1"
STATUS = "SOURCE_ONLY__IN_MEMORY_PAIRED_ACCOUNTING__NO_PAYLOAD_NO_EXECUTION"
INPUT_SCHEMA = "m1643_motion_ep34_s2_ccbs16_paired_input_r1_v1"

M1624_RESULT_NAMESPACE = (
    "hw_autoresearch_nts07/results/"
    "m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_r1_20260901")
M1624_SOURCE_CONTRACT_SHA256 = (
    "2ba3445c2c40c437124c62f49881db1b8443344aa19afc504f4f45aa1c1eacd9")
M1626_RELEASE_SHA256 = (
    "ce15529bcfceda5be92084bdb411330b0c56c8fe47c7024dd9b35a1a0490e273")
CHECKPOINT_SHA256 = (
    "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48")
SAMPLE_ORDER_SHA256 = (
    "d4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773")

SAMPLE_COUNT = 40
SOURCE_GROUP = 16
OUTPUT_TILE = 16
WEIGHTS_PER_BLOCK = SOURCE_GROUP * OUTPUT_TILE
METADATA_BYTES_PER_BLOCK = 2
EPSILON_AXIS = (0.0, 0.01, 0.02, 0.05, 0.10)

OVERALL_AEE_DELTA_MAX = 0.02
PER_SEQUENCE_AEE_DELTA_MAX = 0.03
METADATA_TO_BASELINE_WEIGHT_BYTES_MAX = 0.02
LOCAL_SAME_RESOURCE_SPEEDUP_MIN = 1.15

BASELINE_WITHOUT_TSBG = "CURRENT_TYPED_K8_SAME_RESOURCE"
BASELINE_WITH_TSBG = "ADMITTED_TSBG_TYPED_K8_SAME_RESOURCE"

CLAIM_BOUNDARY = {
    "source_only": True,
    "in_memory_synthetic_or_future_decoded_objects_only": True,
    "actual_payload": False,
    "payload_loader": False,
    "aee_result": False,
    "cycle_result": False,
    "performance_claim": False,
    "paper_result": False,
    "gpu": False,
    "dse": False,
    "rtl": False,
    "eda": False,
    "release": False,
}


class M1643Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1643Error(message)


def _is_number(value):
    return type(value) in (int, float) and not isinstance(value, bool)


def _finite_nonnegative(value, label):
    require(_is_number(value) and math.isfinite(float(value)) and value >= 0,
            label + " must be finite and nonnegative")
    return float(value)


def _positive_integer(value, label):
    require(type(value) is int and value > 0, label + " must be positive integer")
    return value


def _nonnegative_integer(value, label):
    require(type(value) is int and value >= 0,
            label + " must be nonnegative integer")
    return value


def _sha256_string(value, label):
    require(type(value) is str and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            label + " must be lowercase SHA256")
    return value


def _sample_key(row):
    require(type(row) is dict, "sample row must be object")
    sequence = row.get("sequence")
    ordinal = row.get("sample_ordinal")
    global_id = row.get("global_sample_id")
    require(type(sequence) is str and sequence and
            type(ordinal) is int and ordinal >= 0 and
            type(global_id) is str and global_id,
            "invalid sample identity")
    return (sequence, ordinal, global_id)


def _block_key(row):
    require(type(row) is dict, "block row must be object")
    key = row.get("block_id")
    require(type(key) is str and key, "block_id must be nonempty string")
    require(row.get("source_group") == SOURCE_GROUP and
            row.get("output_tile") == OUTPUT_TILE,
            "CCBS block geometry must be fixed 16x16")
    return key


def _verify_capture_identity(capture):
    require(type(capture) is dict, "capture identity must be object")
    require(capture.get("producer") == "M1624" and
            capture.get("result_namespace") == M1624_RESULT_NAMESPACE and
            capture.get("source_contract_sha256") ==
            M1624_SOURCE_CONTRACT_SHA256 and
            capture.get("release_sha256") == M1626_RELEASE_SHA256 and
            capture.get("checkpoint_sha256") == CHECKPOINT_SHA256 and
            capture.get("sample_order_sha256") == SAMPLE_ORDER_SHA256 and
            capture.get("samples") == SAMPLE_COUNT and
            capture.get("reduced_binary") is True,
            "M1624 capture identity/population drift")
    _sha256_string(capture.get("result_manifest_sha256"),
                   "M1624 result manifest")
    _sha256_string(capture.get("result_outer_seal_file_sha256"),
                   "M1624 result outer seal file")
    _sha256_string(capture.get("different_author_result_review_sha256"),
                   "M1624 different-author result review")
    require(capture.get("different_author_result_review_status") ==
            "PASS_M1624_REDUCED_BINARY_RESULT__PAIRED_EVALUATION_ONLY",
            "M1624 result is not released for paired evaluation")


def _verify_baseline_identity(identity, tsbg):
    require(type(identity) is dict and type(tsbg) is dict,
            "baseline/TSBG identity must be objects")
    admitted = tsbg.get("admitted")
    require(type(admitted) is bool, "TSBG admitted flag must be boolean")
    expected_mode = BASELINE_WITH_TSBG if admitted else BASELINE_WITHOUT_TSBG
    require(identity.get("mode") == expected_mode and
            identity.get("same_resource") is True and
            identity.get("same_cohort") is True and
            identity.get("sample_order_sha256") == SAMPLE_ORDER_SHA256 and
            identity.get("checkpoint_sha256") == CHECKPOINT_SHA256,
            "paired baseline identity/resource/cohort drift")
    _sha256_string(identity.get("cycle_model_sha256"), "cycle model")
    _sha256_string(identity.get("resource_model_sha256"), "resource model")
    _sha256_string(identity.get("baseline_receipt_sha256"), "baseline receipt")
    if admitted:
        _sha256_string(tsbg.get("admission_receipt_sha256"),
                       "TSBG admission receipt")
        require(identity.get("includes_admitted_tsbg") is True,
                "TSBG-admitted evaluation must include TSBG in baseline")
    else:
        require(tsbg.get("admission_receipt_sha256") is None and
                identity.get("includes_admitted_tsbg") is False,
                "non-admitted TSBG must not enter baseline")
    require(identity.get("component_speedup_multiplication_allowed") is False,
            "paired S2 speedup must never be multiplied by TSBG")


def _validate_baseline_samples(rows):
    require(type(rows) is list and len(rows) == SAMPLE_COUNT,
            "baseline must contain exactly forty samples")
    samples = {}
    ordered_keys = []
    for sample in rows:
        key = _sample_key(sample)
        require(key not in samples, "duplicate baseline sample")
        aee = _finite_nonnegative(sample.get("aee"), "baseline AEE")
        cycles = _positive_integer(sample.get("cycle_count"),
                                   "baseline cycle_count")
        blocks = sample.get("blocks")
        require(type(blocks) is list and blocks,
                "baseline sample requires nonempty block ledger")
        block_map = {}
        for block in blocks:
            block_id = _block_key(block)
            require(block_id not in block_map, "duplicate baseline block")
            weight_bytes = _positive_integer(block.get("weight_bytes"),
                                             "baseline weight_bytes")
            compute_ops = _positive_integer(block.get("compute_ops"),
                                            "baseline compute_ops")
            psum_bytes = _nonnegative_integer(block.get("psum_bytes"),
                                              "baseline psum_bytes")
            block_cycles = _positive_integer(block.get("service_cycles"),
                                              "baseline block service_cycles")
            starts = []
            for field in ("first_weight_fetch_cycle", "first_compute_cycle",
                          "first_psum_cycle"):
                value = block.get(field)
                _nonnegative_integer(value, "baseline " + field)
                starts.append(value)
            block_map[block_id] = {
                "weight_bytes": weight_bytes,
                "compute_ops": compute_ops,
                "psum_bytes": psum_bytes,
                "service_cycles": block_cycles,
                "first_resource_cycle": min(starts),
            }
        samples[key] = {"aee": aee, "cycle_count": cycles,
                        "blocks": block_map}
        ordered_keys.append(key)
    require(len(set(key[0] for key in ordered_keys)) >= 2,
            "forty-sample cohort must retain sequence stratification")
    return samples, tuple(ordered_keys)


def _epsilon_equal(left, right):
    return abs(float(left) - float(right)) <= 1.0e-12


def _validate_epsilon_axis(points):
    require(type(points) is list and len(points) == len(EPSILON_AXIS),
            "epsilon point count drift")
    values = []
    for point in points:
        require(type(point) is dict, "epsilon point must be object")
        epsilon = point.get("epsilon")
        _finite_nonnegative(epsilon, "epsilon")
        values.append(float(epsilon))
    require(all(_epsilon_equal(a, b) for a, b in zip(values, EPSILON_AXIS)),
            "epsilon axis/order drift")


def _evaluate_point(point, baseline, baseline_order):
    epsilon = float(point["epsilon"])
    rows = point.get("samples")
    require(type(rows) is list and len(rows) == SAMPLE_COUNT,
            "candidate point must contain exactly forty samples")
    candidate = {}
    for row in rows:
        key = _sample_key(row)
        require(key not in candidate, "duplicate candidate sample")
        candidate[key] = row
    require(tuple(candidate.keys()) == baseline_order,
            "candidate sample order/cohort differs from paired baseline")

    per_sample = []
    sequence_deltas = {}
    baseline_cycle_sum = 0
    candidate_cycle_sum = 0
    metadata_bytes = 0
    baseline_weight_bytes = 0
    saved_weight_bytes = 0
    saved_compute_ops = 0
    saved_psum_bytes = 0
    dropped_blocks = 0
    total_blocks = 0

    for key in baseline_order:
        base = baseline[key]
        row = candidate[key]
        candidate_aee = _finite_nonnegative(row.get("aee"), "candidate AEE")
        candidate_cycles = _positive_integer(row.get("cycle_count"),
                                             "candidate cycle_count")
        decisions = row.get("decisions")
        require(type(decisions) is list, "candidate decisions must be list")
        baseline_cycle_sum += base["cycle_count"]
        candidate_cycle_sum += candidate_cycles
        delta = candidate_aee - base["aee"]
        per_sample.append({"sequence": key[0], "sample_ordinal": key[1],
                           "global_sample_id": key[2],
                           "baseline_aee": base["aee"],
                           "candidate_aee": candidate_aee,
                           "aee_delta": delta,
                           "baseline_cycles": base["cycle_count"],
                           "candidate_cycles": candidate_cycles})
        sequence_deltas.setdefault(key[0], []).append(delta)

        if _epsilon_equal(epsilon, 0.0):
            require(not decisions and candidate_aee == base["aee"] and
                    candidate_cycles == base["cycle_count"],
                    "epsilon zero must bypass CCBS and exactly reproduce baseline")
            continue

        decision_map = {}
        for decision in decisions:
            block_id = _block_key(decision)
            require(block_id not in decision_map, "duplicate candidate decision")
            require(type(decision.get("drop")) is bool,
                    "drop decision must be boolean")
            decision_cycle = _nonnegative_integer(decision.get("decision_cycle"),
                                                  "decision_cycle")
            decision_map[block_id] = (decision["drop"], decision_cycle)
        require(set(decision_map) == set(base["blocks"]),
                "positive-epsilon decisions must cover exact baseline blocks")

        for block_id in sorted(base["blocks"]):
            block = base["blocks"][block_id]
            total_blocks += 1
            metadata_bytes += METADATA_BYTES_PER_BLOCK
            baseline_weight_bytes += block["weight_bytes"]
            drop, decision_cycle = decision_map[block_id]
            if drop:
                require(decision_cycle < block["first_resource_cycle"],
                        "drop decision did not precede weight/compute/psum")
                dropped_blocks += 1
                saved_weight_bytes += block["weight_bytes"]
                saved_compute_ops += block["compute_ops"]
                saved_psum_bytes += block["psum_bytes"]

    overall_delta = sum(row["aee_delta"] for row in per_sample) / SAMPLE_COUNT
    per_sequence = dict((sequence, sum(values) / len(values))
                        for sequence, values in sorted(sequence_deltas.items()))
    speedup = float(baseline_cycle_sum) / float(candidate_cycle_sum)
    metadata_ratio = (0.0 if baseline_weight_bytes == 0 else
                      float(metadata_bytes) / float(baseline_weight_bytes))
    gates = {
        "overall_aee_delta_le_0p02":
            overall_delta <= OVERALL_AEE_DELTA_MAX + 1.0e-12,
        "every_sequence_aee_delta_le_0p03": all(
            value <= PER_SEQUENCE_AEE_DELTA_MAX + 1.0e-12
            for value in per_sequence.values()),
        "metadata_le_2pct_baseline_weight_bytes":
            metadata_ratio <= METADATA_TO_BASELINE_WEIGHT_BYTES_MAX,
        "local_same_resource_ratio_of_sums_cycles_ge_1p15":
            speedup >= LOCAL_SAME_RESOURCE_SPEEDUP_MIN,
    }
    return {
        "epsilon": epsilon,
        "epsilon_zero_exact_existing_path": _epsilon_equal(epsilon, 0.0),
        "per_sample": per_sample,
        "per_sequence_aee_delta": per_sequence,
        "overall_aee_delta": overall_delta,
        "cycle_account": {
            "aggregation": "ratio_of_sums",
            "baseline_cycles": baseline_cycle_sum,
            "candidate_cycles": candidate_cycle_sum,
            "local_same_resource_speedup": speedup,
        },
        "metadata_account": {
            "geometry": "16x16",
            "encoding": "one_uint16_bound_per_positive_epsilon_block",
            "metadata_bytes": metadata_bytes,
            "baseline_weight_bytes": baseline_weight_bytes,
            "metadata_to_baseline_weight_bytes": metadata_ratio,
        },
        "internally_derived_savings": {
            "total_blocks": total_blocks,
            "dropped_blocks": dropped_blocks,
            "weight_bytes": saved_weight_bytes,
            "compute_ops": saved_compute_ops,
            "psum_bytes": saved_psum_bytes,
            "decision_precedes_all_credited_resources": True,
        },
        "gates": gates,
        "passes_fixed_gate": all(gates.values()),
    }


def evaluate_paired_document(document):
    """Validate and evaluate one already-decoded, exact paired document."""
    require(type(document) is dict and document.get("schema") == INPUT_SCHEMA,
            "paired input schema drift")
    _verify_capture_identity(document.get("capture"))
    _verify_baseline_identity(document.get("baseline_identity"),
                              document.get("tsbg"))
    baseline, baseline_order = _validate_baseline_samples(
        document.get("baseline_samples"))
    points = document.get("epsilon_points")
    _validate_epsilon_axis(points)
    results = [_evaluate_point(point, baseline, baseline_order)
               for point in points]
    require(results[0]["epsilon_zero_exact_existing_path"] and
            results[0]["metadata_account"]["metadata_bytes"] == 0 and
            results[0]["internally_derived_savings"]["dropped_blocks"] == 0,
            "epsilon-zero exact subset invariant failed")
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "capture_identity": "M1624_40_SAMPLE_REDUCED_BINARY",
        "baseline_mode": document["baseline_identity"]["mode"],
        "tsbg_admitted": document["tsbg"]["admitted"],
        "paired_speedup_only": True,
        "component_speedup_multiplication_allowed": False,
        "epsilon_axis": list(EPSILON_AXIS),
        "points": results,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "paper_admission": False,
        "different_author_review_required": True,
    }


def source_self_check():
    require(WEIGHTS_PER_BLOCK == 256 and METADATA_BYTES_PER_BLOCK == 2,
            "CCBS16 geometry/metadata drift")
    require(EPSILON_AXIS[0] == 0.0 and
            all(EPSILON_AXIS[index] < EPSILON_AXIS[index + 1]
                for index in range(len(EPSILON_AXIS) - 1)),
            "epsilon axis is not zero-rooted and increasing")
    require(CLAIM_BOUNDARY["source_only"] and
            not any(CLAIM_BOUNDARY[key] for key in (
                "actual_payload", "payload_loader", "aee_result",
                "cycle_result", "performance_claim", "paper_result",
                "gpu", "dse", "rtl", "eda", "release")),
            "claim boundary drift")
    return {"schema": SCHEMA, "status": STATUS,
            "geometry": [SOURCE_GROUP, OUTPUT_TILE],
            "metadata_bytes_per_positive_epsilon_block":
                METADATA_BYTES_PER_BLOCK,
            "epsilon_axis": list(EPSILON_AXIS), "samples": SAMPLE_COUNT,
            "claim_boundary": dict(CLAIM_BOUNDARY)}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    require(argv == ["--source-self-check"],
            "only --source-self-check is available; no actual evaluation CLI")
    print(json.dumps(source_self_check(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
