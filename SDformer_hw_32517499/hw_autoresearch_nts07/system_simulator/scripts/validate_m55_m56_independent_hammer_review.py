#!/usr/bin/env python3
"""Validate the independent raw M55/M56 hammer and its bounded review."""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess


HW_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (HW_ROOT / "results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/"
            "manifest.json")
PAYLOAD_RECEIPT = (HW_ROOT /
    "results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/"
    "m51_h67_ep35_binary_input_trace_gpu_payload_validation_receipt_r1.json")
M55_CONTRACT = HW_ROOT / "contracts/m55_h67_full_network_dual_parent_opportunity_contract_r1_20260823.json"
M55_ANALYZER = HW_ROOT / "system_simulator/scripts/analyze_m55_h67_full_network_dual_parent_opportunity.py"
M55_VALIDATOR = HW_ROOT / "system_simulator/scripts/validate_m55_h67_full_network_dual_parent_opportunity.py"
M55_RESULT = (HW_ROOT / "results/m55_h67_full_network_dual_parent_opportunity_r1_20260823/"
              "m55_h67_full_network_dual_parent_opportunity_result_r1.json")
M56_CONTRACT = HW_ROOT / "contracts/m56_prediction_head_lane_fold_dse_contract_r1_20260823.json"
M56_ANALYZER = HW_ROOT / "system_simulator/scripts/analyze_m56_prediction_head_lane_fold_dse.py"
M56_VALIDATOR = HW_ROOT / "system_simulator/scripts/validate_m56_prediction_head_lane_fold_dse.py"
M56_RESULT = (HW_ROOT / "results/m56_prediction_head_lane_fold_dse_r1_20260823/"
              "m56_prediction_head_lane_fold_dse_result_r1.json")
M53_CONTRACT = HW_ROOT / "contracts/m53_adaptive_temporal_parent_k4_ctx16_dse_contract_r1_20260823.json"
RAW_ANALYZER = HW_ROOT / "system_simulator/scripts/reconstruct_m55_m56_raw_payload_independent.py"
REVIEW_DIR = HW_ROOT / "results/m55_m56_independent_hammer_20260823"
RECONSTRUCTION = REVIEW_DIR / "m55_m56_raw_independent_reconstruction.json"
RAW_STDOUT = REVIEW_DIR / "reconstruction.stdout.log"
RAW_STDERR = REVIEW_DIR / "reconstruction.stderr.log"
REVIEW = REVIEW_DIR / "m55_m56_independent_hammer_review.json"

EXPECTED_SHA = {
    "manifest": "2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e",
    "payload_receipt": "d37e26a9e3206229746eb21209603376a4c07c3aa69f7500d0b960f64c580c32",
    "m55_contract": "31df83ef6adf6b1e567deeaa6cce1af8e3b4e6f7f35a092e47133a59f00a5bda",
    "m55_analyzer": "9532e09845956abde97138fc763d704e963c408291bea72675181b67047620c3",
    "m55_validator": "c20bad2b3d511e16106ff909a859e86c0dcfb594bd2a9f925e1591797aae1916",
    "m55_result": "9639903ea82e90b1a8403ff0bee66b01ec732ee6baa11d275ec2725e0a4d531b",
    "m56_contract": "cba82292504bfaa1015f54e254a27719749122c9783ad16d6f0ff6a6cc961263",
    "m56_analyzer": "713705f3550646fe63efaca60493dad4bad4f07cf4d447ba7b9dfd405b68b67e",
    "m56_validator": "63050fc859c59f2f8abbfb821a08dbdf2e45a29006279a47e1bfe46b5ebcd997",
    "m56_result": "1aca6c0d6215f91035434cca45a04dd1d21100f1e5bbd2138851c575188b808a",
    "m53_contract": "e1dd6eb10a4b580115ff8cfe9d28605167256dfe81942ea2e2ea92d5fba88e03",
    "raw_analyzer": "8729f836e887878093adc57ff582748edbf88ac7151e2425c2b26e28f291b871",
    "reconstruction": "2f6e2c738f9e2928a613093328325c6fc346d2308a7c02dda185efcd00fad96d",
    "raw_stdout": "df8ee4d528560351c4e210ae79dc87cee07449f88f9e86c23d12a4da312ff164",
    "raw_stderr": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
EXPECTED_REVIEW_SHA256 = (
    "b5b41ef9e296dde79b4115e365e4caea723aae289114578478db9aae892082cb")
EXPECTED_ATTACKS_SHA256 = (
    "ffdd8914f865b345236e423529d631cef3558f3fc83855234935b0bc803037e4")

PARENTS = ("zero", "left", "up", "previous_timestep")
MODES_M55 = ("zero", "local", "motion", "dual")
MODES_M56 = ("zero", "local", "dual")
WIDTHS = (1, 2, 4, 8, 16, 24, 32, 40, 48)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON: {}".format(raw))

    def pairs(raw_pairs):
        result = {}
        for key, value in raw_pairs:
            require(key not in result, "duplicate JSON key: {}".format(key))
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def close(actual, expected, label):
    require(abs(float(actual) - float(expected)) <=
            1e-12 * max(1.0, abs(float(expected))),
            "float mismatch {}".format(label))


def normalized_shape(record):
    shape = [int(value) for value in record["input_shape"]]
    if record["operator"] == "Linear":
        require(len(shape) == 5, "Linear rank")
        return shape
    require(record["operator"] == "Conv2d", "operator")
    if len(shape) == 5:
        return [shape[0], shape[1], shape[3], shape[4], shape[2]]
    require(len(shape) == 4, "Conv rank")
    return [shape[0], 1, shape[2], shape[3], shape[1]]


def validate_core(manifest, payload_receipt, m55_contract, m55, m56_contract,
                  m56, m53_contract, reconstruction):
    require(manifest["schema"] == "m51_h67_ep35_binary_input_trace_manifest_v1" and
            manifest["status"] ==
            "PASS_EXACT_BINARY_INPUT_TRACE_NO_OUTPUT_OR_PERFORMANCE_CLAIM",
            "manifest identity/status")
    require(payload_receipt["status"] ==
            "PASS_REAL_GPU_ALL310_PAYLOAD_SHA_SIZE_POPCOUNT_PLAN_IDENTITY" and
            all(payload_receipt["checks"].values()) and
            payload_receipt["identity"]["manifest_sha256"] == EXPECTED_SHA["manifest"],
            "payload receipt")
    require(m55_contract["status"] == "FROZEN_EXACT_SOURCE_WORK_ONLY" and
            "speedup" in " ".join(m55_contract["claim_boundary"]["forbidden"]).lower(),
            "M55 boundary")
    require(m55["status"] ==
            "PASS_EXACT_SOURCE_BIT_WORK_NO_CYCLE_SPEEDUP_ENERGY_OR_PPA_CLAIM" and
            m55["contract_sha256"] == EXPECTED_SHA["m55_contract"],
            "M55 identity/status")
    forbidden56 = " ".join(m56_contract["claim_boundary"]["forbidden"]).lower()
    require(m56_contract["status"] == "FROZEN_HEAD_SOURCE_ISSUE_MODEL_ONLY" and
            "system speedup" in forbidden56 and "numerical-equivalence" in forbidden56 and
            "int8" in forbidden56, "M56 boundary")
    require(m56["status"] ==
            "PASS_EXACT_HEAD_SOURCE_ISSUE_DSE_NO_SYSTEM_RTL_PPA_ENERGY_CLAIM" and
            m56["contract_sha256"] == EXPECTED_SHA["m56_contract"],
            "M56 identity/status")
    require(reconstruction["status"] ==
            "PASS_ALL310_RAW_SHA_POPCOUNT_LAYOUT_PARENT_SIGNED_AND_HEAD_DSE",
            "raw reconstruction status")
    require(reconstruction["identity"] == {
        "manifest_sha256": EXPECTED_SHA["manifest"],
        "m55_result_sha256": EXPECTED_SHA["m55_result"],
        "m56_result_sha256": EXPECTED_SHA["m56_result"],
        "reviewer_defined_payload_collection_sha256":
            "61a4036e4d435e3c59b829de0610da9eee08d345edfe93587e73c0bfe9167f6f",
    }, "raw reconstruction identities")
    require(reconstruction["population"] == {
        "records": 310, "samples": 10, "modules": 31,
        "input_elements": 10506240000, "packed_bytes": 1313280000,
        "active_elements": 712894209}, "raw population")

    records = manifest["records"]
    raw_rows = reconstruction["per_record"]
    m55_rows = m55["per_record"]
    require(len(records) == len(raw_rows) == len(m55_rows) == 310,
            "record population")
    identities = set()
    source_sums = dict((mode, 0) for mode in MODES_M55)
    signed_sums = dict((mode, {"positive_0_to_1": 0,
                               "negative_1_to_0": 0}) for mode in MODES_M55)
    choice_sums = dict((parent, 0) for parent in PARENTS)
    modules = [dict((mode, 0) for mode in MODES_M55) for _ in range(31)]
    samples = [dict((mode, 0) for mode in MODES_M55) for _ in range(10)]
    for ordinal, (manifest_row, raw, producer) in enumerate(
            zip(records, raw_rows, m55_rows)):
        identity = (int(manifest_row["sample_id"]), int(manifest_row["module_index"]))
        require(identity not in identities, "duplicate sample/module")
        identities.add(identity)
        require(raw["ordinal"] == producer["ordinal"] == ordinal and
                raw["sample_id"] == producer["sample_id"] == identity[0] and
                raw["module_index"] == producer["module_index"] == identity[1] and
                raw["relative_path"] == producer["relative_path"] ==
                manifest_row["relative_path"] and
                raw["file_sha256"] == producer["file_sha256"] ==
                manifest_row["file_sha256"] and
                raw["packed_bytes"] == manifest_row["packed_bytes"] and
                raw["active_elements"] == producer["active_elements"] ==
                manifest_row["active_elements"] and
                raw["input_elements"] == producer["input_elements"] ==
                manifest_row["input_elements"] and
                raw["normalized_shape_tbhwc"] == normalized_shape(manifest_row) ==
                producer["analysis"]["vector_shape_tbhwc"],
                "record identity/layout {}".format(ordinal))
        require(raw["source_bits"] == producer["analysis"]["source_bits"] and
                dict((mode, raw["choice_counts"][mode])
                     for mode in ("local", "motion", "dual")) ==
                producer["analysis"]["choice_counts"] and
                raw["source_bits_by_timestep"] ==
                producer["analysis"]["source_bits_by_timestep"],
                "record source/choice {}".format(ordinal))
        for mode in MODES_M55:
            signed = raw["signed_source_bits"][mode]
            require(set(signed) == {"positive_0_to_1", "negative_1_to_0"} and
                    signed["positive_0_to_1"] >= 0 and
                    signed["negative_1_to_0"] >= 0 and
                    signed["positive_0_to_1"] + signed["negative_1_to_0"] ==
                    raw["source_bits"][mode],
                    "signed source conservation {} {}".format(ordinal, mode))
            source_sums[mode] += raw["source_bits"][mode]
            modules[identity[1]][mode] += raw["source_bits"][mode]
            samples[identity[0]][mode] += raw["source_bits"][mode]
            for direction in signed_sums[mode]:
                signed_sums[mode][direction] += signed[direction]
        require(raw["signed_source_bits"]["zero"]["negative_1_to_0"] == 0,
                "zero parent negative residual")
        for parent in PARENTS:
            choice_sums[parent] += raw["choice_counts"]["dual"][parent]
    require(identities == set((sample, module) for sample in range(10)
                              for module in range(31)), "Cartesian population")
    aggregate = reconstruction["aggregate"]
    require(aggregate["source_bits"] == source_sums and
            aggregate["signed_source_bits"] == signed_sums and
            aggregate["choice_counts"] == choice_sums and
            aggregate["hook_calls"] == 310 and
            aggregate["input_elements"] == 10506240000 and
            aggregate["vector_count"] == 88500000,
            "raw aggregate reconstruction")
    require(source_sums == {
        "zero": m55["aggregate"]["zero_source_bits"],
        "local": m55["aggregate"]["local_source_bits"],
        "motion": m55["aggregate"]["motion_source_bits"],
        "dual": m55["aggregate"]["dual_source_bits"]} and
            choice_sums == m55["aggregate"]["choice_counts"],
            "M55 aggregate reconciliation")
    for index in range(31):
        producer = m55["per_module"][index]
        require(all(modules[index][mode] == producer[mode + "_source_bits"]
                    for mode in MODES_M55) and
                reconstruction["per_module"][index]["choice_counts"] ==
                producer["choice_counts"],
                "module reconciliation {}".format(index))
    for index in range(10):
        producer = m55["per_sample"][index]
        require(all(samples[index][mode] == producer[mode + "_source_bits"]
                    for mode in MODES_M55) and
                reconstruction["per_sample"][index]["choice_counts"] ==
                producer["choice_counts"],
                "sample reconciliation {}".format(index))

    contribution = reconstruction["module30_marginal_dual_vs_local_contribution"]
    require(contribution["aggregate_saved_source_bits"] == 33996201 and
            contribution["module30_saved_source_bits"] == 28811162,
            "module30 contribution numerator/denominator")
    close(contribution["percent"], 84.74818112765011, "module30 percent")
    require(set(row["sequence_key"] for row in records) == {"zurich_city_09_a"},
            "frozen sequence scope")

    module_name = "sttmultires_unet.preds.3.conv.0"
    weight = manifest["module_identities"][module_name]["weight"]
    bias = manifest["module_identities"][module_name]["bias"]
    require(weight == {"byte_order": "little", "content_bytes": 768,
                       "content_sha256":
                       "ae9d949992c2b7345c4ef4129010be83d3d98ce261ba43c7bfa0c7be7cadf969",
                       "dtype": "torch.float32", "layout": "C_ORDER_CONTIGUOUS",
                       "shape": [2, 96, 1, 1]} and
            bias["dtype"] == "torch.float32" and bias["shape"] == [2],
            "head float32 weight/bias identity")
    require(m56["source_bits"] == {
        "zero": reconstruction["per_module"][30]["source_bits"]["zero"],
        "local": reconstruction["per_module"][30]["source_bits"]["local"],
        "dual": reconstruction["per_module"][30]["source_bits"]["dual"]},
            "M56 source reconciliation")
    raw_widths = reconstruction["module30_lane_fold_widths"]
    producer_widths = m56["widths"]
    require([row["pixels_per_group"] for row in raw_widths] == list(WIDTHS) and
            [row["pixels_per_group"] for row in producer_widths] == list(WIDTHS),
            "width sweep order")
    common = ("allocated_lane_product_slots", "event_cycle_histogram",
              "event_cycles", "event_plus_one_commit_cycle_per_group", "groups",
              "physical_product_slots", "product_updates", "union_source_indices",
              "zero_event_groups")
    for width, raw_width, producer_width in zip(WIDTHS, raw_widths, producer_widths):
        require(raw_width["fixed_dense"] == producer_width["fixed_dense"],
                "dense width{}".format(width))
        expected_groups = 10 * 10 * 240 * ((320 + width - 1) // width)
        require(raw_width["fixed_dense"]["groups"] == expected_groups,
                "row-bounded group count width{}".format(width))
        for mode in MODES_M56:
            raw_mode = raw_width["modes"][mode]
            producer_mode = producer_width["modes"][mode]
            require(all(raw_mode[key] == producer_mode[key] for key in common),
                    "mode width{} {}".format(width, mode))
            require(raw_mode["positive_product_updates"] +
                    raw_mode["negative_product_updates"] ==
                    raw_mode["product_updates"] ==
                    2 * reconstruction["per_module"][30]["source_bits"][mode],
                    "signed product conservation width{} {}".format(width, mode))
            require(sum(raw_mode["event_cycle_histogram"].values()) ==
                    raw_mode["groups"] and
                    sum(int(key) * value
                        for key, value in raw_mode["event_cycle_histogram"].items()) ==
                    raw_mode["event_cycles"] and
                    raw_mode["event_plus_one_commit_cycle_per_group"] ==
                    raw_mode["event_cycles"] + raw_mode["groups"] and
                    raw_mode["physical_product_slots"] ==
                    raw_mode["event_cycles"] * 8 * 96,
                    "cycle/hist/slot width{} {}".format(width, mode))
            close(producer_mode["physical_lane_utilization"],
                  float(raw_mode["product_updates"]) /
                  float(raw_mode["physical_product_slots"]),
                  "physical utilization")
            close(producer_mode["allocated_lane_utilization"],
                  float(raw_mode["product_updates"]) /
                  float(raw_mode["allocated_lane_product_slots"]),
                  "allocated utilization")
        ratios = producer_width["head_kernel_ratios_not_system_speedup"]
        close(ratios["dense_over_dual_event_plus_commit"],
              float(raw_width["fixed_dense"]["event_plus_one_commit_cycle_per_group"]) /
              float(raw_width["modes"]["dual"][
                  "event_plus_one_commit_cycle_per_group"]), "dense/dual+commit")
    require(m56["selected_by_minimum_dual_event_cycles"] == 48 and
            m56["selected_by_minimum_dual_event_plus_commit"] == 48,
            "selected P48")
    p1, p48 = raw_widths[0], raw_widths[-1]
    require(p48["modes"]["dual"]["event_cycles"] == 539614 and
            p48["modes"]["dual"]["event_plus_one_commit_cycle_per_group"] == 707614,
            "P48 events")
    close(float(p1["modes"]["dual"]["event_cycles"]) /
          float(p48["modes"]["dual"]["event_cycles"]),
          19.052976757459962, "P1/P48 event")
    close(p48["ratios_not_system_speedup"]["dense_over_dual_event_plus_commit"],
          3.0864284765422956, "P48 dense/dual+commit")

    storage = m53_contract["temporal_parent_storage_proof"]
    require(storage["existing_two_frame_bytes"] == 136800 and
            storage["bit_tight_frame_bytes"] == 68400 and
            set(storage["mapping"]) == {
                "current_timestep_accumulator",
                "fully_committed_previous_timestep_parent"},
            "M53 scratch identity")
    state = {
        "prediction_head_input_prior_timestep_bytes_1b":
            240 * 320 * 96 // 8,
        "prediction_head_input_prior_row_bytes_1b": 320 * 96 // 8,
        "prediction_head_p48_input_tile_bytes_1b": 48 * 96 // 8,
        "prediction_head_prior_output_frame_bytes_signed19":
            (240 * 320 * 2 * 19 + 7) // 8,
        "m53_two_accumulator_frames_bytes": 136800,
    }
    require(state == {
        "prediction_head_input_prior_timestep_bytes_1b": 921600,
        "prediction_head_input_prior_row_bytes_1b": 3840,
        "prediction_head_p48_input_tile_bytes_1b": 576,
        "prediction_head_prior_output_frame_bytes_signed19": 364800,
        "m53_two_accumulator_frames_bytes": 136800}, "state byte arithmetic")
    return state


def run_attacks(documents):
    names = ("manifest", "payload_receipt", "m55_contract", "m55", "m56_contract",
             "m56", "m53_contract", "reconstruction")
    attacks = []

    def attack(name, mutate):
        docs = dict((key, copy.deepcopy(documents[key])) for key in names)
        mutate(docs)
        rejected = False
        diagnostic = ""
        try:
            validate_core(*(docs[key] for key in names))
        except Exception as exc:
            rejected = True
            diagnostic = str(exc)
        require(rejected, "tamper accepted {}".format(name))
        attacks.append({"name": name, "rejected": True,
                        "diagnostic": diagnostic})

    attack("manifest_payload_sha_flip", lambda d:
           d["manifest"]["records"][0].__setitem__("file_sha256", "0" * 64))
    attack("manifest_popcount_flip", lambda d:
           d["manifest"]["records"][0].__setitem__(
               "active_elements", d["manifest"]["records"][0]["active_elements"] + 1))
    attack("manifest_layout_flip", lambda d:
           d["manifest"]["records"][0].__setitem__("input_shape", [10, 1, 48, 640, 480]))
    attack("manifest_weight_dtype_int8_forgery", lambda d:
           d["manifest"]["module_identities"]["sttmultires_unet.preds.3.conv.0"][
               "weight"].__setitem__("dtype", "torch.int8"))
    attack("raw_record_drop", lambda d: d["reconstruction"]["per_record"].pop())
    attack("raw_signed_negative_flip", lambda d:
           d["reconstruction"]["per_record"][30]["signed_source_bits"]["dual"].__setitem__(
               "negative_1_to_0", d["reconstruction"]["per_record"][30][
                   "signed_source_bits"]["dual"]["negative_1_to_0"] + 1))
    attack("raw_source_bit_flip", lambda d:
           d["reconstruction"]["per_record"][0]["source_bits"].__setitem__(
               "dual", d["reconstruction"]["per_record"][0]["source_bits"]["dual"] + 1))
    attack("m55_aggregate_flip", lambda d:
           d["m55"]["aggregate"].__setitem__(
               "dual_source_bits", d["m55"]["aggregate"]["dual_source_bits"] + 1))
    attack("m55_module30_choice_flip", lambda d:
           d["m55"]["per_module"][30]["choice_counts"].__setitem__(
               "previous_timestep", d["m55"]["per_module"][30]["choice_counts"][
                   "previous_timestep"] + 1))
    attack("module30_contribution_flip", lambda d:
           d["reconstruction"]["module30_marginal_dual_vs_local_contribution"].__setitem__(
               "percent", 50.0))
    attack("m56_width_drop", lambda d: d["m56"]["widths"].pop())
    attack("m56_p48_event_flip", lambda d:
           d["m56"]["widths"][-1]["modes"]["dual"].__setitem__(
               "event_cycles", d["m56"]["widths"][-1]["modes"]["dual"]["event_cycles"] + 1))
    attack("m56_p48_hist_flip", lambda d:
           d["m56"]["widths"][-1]["modes"]["dual"]["event_cycle_histogram"].__setitem__(
               "1", d["m56"]["widths"][-1]["modes"]["dual"][
                   "event_cycle_histogram"]["1"] + 1))
    attack("m56_p48_commit_flip", lambda d:
           d["m56"]["widths"][-1]["modes"]["dual"].__setitem__(
               "event_plus_one_commit_cycle_per_group", 707615))
    attack("m56_p48_physical_slots_flip", lambda d:
           d["m56"]["widths"][-1]["modes"]["dual"].__setitem__(
               "physical_product_slots", 414423553))
    attack("m56_p48_product_updates_flip", lambda d:
           d["m56"]["widths"][-1]["modes"]["dual"].__setitem__(
               "product_updates", 104133695))
    attack("m56_selected_width_flip", lambda d:
           d["m56"].__setitem__("selected_by_minimum_dual_event_cycles", 40))
    attack("m56_claim_boundary_weakening", lambda d:
           d["m56_contract"]["claim_boundary"].__setitem__("forbidden", []))
    return {"schema": "m55_m56_independent_hammer_tamper_attacks_v1",
            "status": "PASS_ALL_TAMPERS_REJECTED",
            "attack_count": len(attacks),
            "rejected_count": sum(1 for row in attacks if row["rejected"]),
            "attacks": attacks}


def validate(attacks_output, receipt_output, rerun_producers):
    paths = {
        "manifest": MANIFEST, "payload_receipt": PAYLOAD_RECEIPT,
        "m55_contract": M55_CONTRACT, "m55_analyzer": M55_ANALYZER,
        "m55_validator": M55_VALIDATOR, "m55_result": M55_RESULT,
        "m56_contract": M56_CONTRACT, "m56_analyzer": M56_ANALYZER,
        "m56_validator": M56_VALIDATOR, "m56_result": M56_RESULT,
        "m53_contract": M53_CONTRACT, "raw_analyzer": RAW_ANALYZER,
        "reconstruction": RECONSTRUCTION, "raw_stdout": RAW_STDOUT,
        "raw_stderr": RAW_STDERR,
    }
    for name, path in paths.items():
        require(path.is_file() and sha256(path) == EXPECTED_SHA[name],
                "exact SHA drift {}".format(name))
    documents = {
        "manifest": strict_json(MANIFEST),
        "payload_receipt": strict_json(PAYLOAD_RECEIPT),
        "m55_contract": strict_json(M55_CONTRACT),
        "m55": strict_json(M55_RESULT),
        "m56_contract": strict_json(M56_CONTRACT),
        "m56": strict_json(M56_RESULT),
        "m53_contract": strict_json(M53_CONTRACT),
        "reconstruction": strict_json(RECONSTRUCTION),
    }
    state = validate_core(*(documents[key] for key in (
        "manifest", "payload_receipt", "m55_contract", "m55", "m56_contract",
        "m56", "m53_contract", "reconstruction")))
    attacks = run_attacks(documents)
    require(attacks["attack_count"] == attacks["rejected_count"] == 18,
            "attack population")
    producer_stdout = {}
    if rerun_producers:
        for name, validator, marker in (
                ("m55", M55_VALIDATOR, "PASS M55 exact source-bit opportunity"),
                ("m56", M56_VALIDATOR, "PASS_M56_EXACT_HEAD_DSE")):
            result = subprocess.run(["/usr/bin/python3.6", str(validator)],
                                    cwd=str(HW_ROOT), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
            require(result.returncode == 0 and marker in result.stdout,
                    "producer validator rerun {}".format(name))
            producer_stdout[name] = result.stdout
    require(not attacks_output.exists() and not receipt_output.exists(),
            "refusing reviewer output overwrite")
    attacks_output.parent.mkdir(parents=True, exist_ok=True)
    attacks_output.write_text(json.dumps(attacks, indent=2, sort_keys=True) + "\n")
    if EXPECTED_ATTACKS_SHA256 != "TO_BE_FROZEN":
        require(sha256(attacks_output) == EXPECTED_ATTACKS_SHA256,
                "attack byte drift")
    if EXPECTED_REVIEW_SHA256 != "TO_BE_FROZEN":
        require(REVIEW.is_file() and sha256(REVIEW) == EXPECTED_REVIEW_SHA256,
                "review SHA drift")
        review = strict_json(REVIEW)
        require(review["scores"] == {"M55": 91, "M56": 74, "joint": 81} and
                review["finding_counts"] == {"P0": 0, "P1": 4, "P2": 4} and
                review["verdict"] ==
                "GO_M55_SOURCE_WORK_AND_M56_ISSUE_OPPORTUNITY_ONLY_NO_GO_HARDWARE_OR_SYSTEM_PERFORMANCE",
                "review conclusion drift")
    receipt = {
        "schema": "m55_m56_independent_hammer_validation_receipt_v1",
        "status": "PASS_M55_M56_INDEPENDENT_HAMMER_VALIDATED",
        "review_sha256": sha256(REVIEW) if REVIEW.is_file() else None,
        "validator_sha256": sha256(Path(__file__)),
        "raw_analyzer_sha256": sha256(RAW_ANALYZER),
        "raw_reconstruction_sha256": sha256(RECONSTRUCTION),
        "attacks_sha256": sha256(attacks_output),
        "producer_validators_rerun": bool(rerun_producers),
        "producer_validator_stdout_sha256": dict(
            (name, hashlib.sha256(value.encode("utf-8")).hexdigest())
            for name, value in sorted(producer_stdout.items())),
        "raw_population": documents["reconstruction"]["population"],
        "signed_source_bits": documents["reconstruction"]["aggregate"][
            "signed_source_bits"],
        "module30_marginal_contribution_percent": 84.74818112765011,
        "p48": {"dual_event_cycles": 539614,
                "dual_event_plus_commit": 707614,
                "dense_over_dual_plus_commit_not_system": 3.0864284765422956,
                "p1_dual_over_p48_dual_event_not_system": 19.052976757459962},
        "state_bytes_challenge": state,
        "tamper_attack_count": 18,
        "tamper_rejected_count": 18,
        "open_source_hdl_tool_used": False,
        "dc_launched": False,
        "system_speedup_admitted": False,
    }
    receipt_output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacks-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--rerun-producers", action="store_true")
    args = parser.parse_args()
    result = validate(args.attacks_output, args.receipt_output,
                      args.rerun_producers)
    print("PASS M55/M56 independent hammer M55=91 M56=74 joint=81 P0=0 P1=4 P2=4 attacks=18/18")
    print("validator_sha256={}".format(result["validator_sha256"]))


if __name__ == "__main__":
    main()
