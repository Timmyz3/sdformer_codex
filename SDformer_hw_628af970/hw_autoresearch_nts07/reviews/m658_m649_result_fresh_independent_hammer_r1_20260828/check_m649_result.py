#!/usr/bin/env python3
"""Independent, read-only arithmetic/identity audit of the sealed M649 result."""

import hashlib
import json
import math
import operator
from functools import reduce
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RESULT = REPO / "hw_autoresearch_nts07/results/m649_h67_ep35_convtranspose_typed_numeric_audit_s10_r1_20260828"
RESULT_JSON = RESULT / "m649_typed_numeric_audit.json"
CONTRACT = REPO / "hw_autoresearch_nts07/contracts/m649_h67_ep35_convtranspose_typed_numeric_audit_contract_r1_20260828.json"
M511_CONTRACT = REPO / "hw_autoresearch_nts07/contracts/m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json"
DOCS359 = REPO / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
M511_CANONICAL = REPO / "hw_autoresearch_nts07/system_handoff/outgoing/m511_h67_ep35_convtranspose_binary_inputs_s10_r1_20260827"


def fail(message):
    raise AssertionError(message)


def require(condition, message):
    if not condition:
        fail(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def strict_load(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key {} in {}".format(key, path))
            out[key] = value
        return out

    def constant(value):
        fail("non-standard JSON number {} in {}".format(value, path))

    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_constant=constant)


def check_regular_identity(item, label):
    path = Path(item["path"])
    require(path.is_absolute(), label + " path is not absolute")
    require(path.exists() and path.is_file() and not path.is_symlink(), label + " is not a regular non-symlink file")
    require(path.stat().st_size == item["bytes"], label + " byte mismatch")
    require(sha256(path) == item["sha256"], label + " sha256 mismatch")


def parse_manifest(directory):
    manifest = directory / "SHA256SUMS"
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require("/" not in name and name not in ("", ".", ".."), "unsafe manifest member")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "manifest member not regular: " + name)
        require(sha256(path) == digest, "manifest digest mismatch: " + name)
        rows.append(name)
    require(rows == ["RUN_COMPLETE.txt", "m649_typed_numeric_audit.json"], "unexpected result manifest population")
    outer = (directory / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8").strip().split()
    require(outer == [sha256(manifest), "SHA256SUMS"], "outer seal mismatch")
    require(sorted(p.name for p in directory.iterdir()) ==
            ["RUN_COMPLETE.txt", "SHA256SUMS", "SHA256SUMS.seal.sha256", "m649_typed_numeric_audit.json"],
            "unexpected canonical population")


COUNT_KEYS = ("elements", "zero_count", "one_count", "exact_binary_count",
              "nonbinary_finite_count", "integer_count", "nonfinite_count")
BOOL_KEYS = ("all_exact_binary", "all_finite", "all_integer")


def validate_summary(summary, label, channels=None):
    for key in COUNT_KEYS:
        require(type(summary[key]) is int and summary[key] >= 0, label + " invalid " + key)
    if channels is not None:
        require(summary["channels"] == channels, label + " channel mismatch")
    require(summary["exact_binary_count"] == summary["zero_count"] + summary["one_count"],
            label + " binary decomposition mismatch")
    require(summary["elements"] == summary["exact_binary_count"] +
            summary["nonbinary_finite_count"] + summary["nonfinite_count"],
            label + " population decomposition mismatch")
    require(summary["integer_count"] <= summary["elements"] - summary["nonfinite_count"],
            label + " integer bound mismatch")
    require(summary["all_exact_binary"] == (summary["exact_binary_count"] == summary["elements"]),
            label + " binary Boolean mismatch")
    require(summary["all_finite"] == (summary["nonfinite_count"] == 0),
            label + " finite Boolean mismatch")
    require(summary["all_integer"] == (summary["integer_count"] == summary["elements"]),
            label + " integer Boolean mismatch")


def aggregate_channels(channels):
    out = {key: sum(row[key] for row in channels) for key in COUNT_KEYS}
    out.update({
        "channels": len(channels),
        "all_exact_binary": out["exact_binary_count"] == out["elements"],
        "all_finite": out["nonfinite_count"] == 0,
        "all_integer": out["integer_count"] == out["elements"],
    })
    return out


def compare_summary(observed, expected, label):
    for key in COUNT_KEYS + ("channels",) + BOOL_KEYS:
        require(observed[key] == expected[key], label + " mismatch: " + key)


def popcount(path):
    total = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += sum(bin(byte).count("1") for byte in block)
    return total


def product(values):
    return reduce(operator.mul, values, 1)


def main():
    checks = []
    check = lambda name: checks.append(name)
    parse_manifest(RESULT)
    check("canonical_double_seal_and_exact_population")
    data = strict_load(RESULT_JSON)
    contract = strict_load(CONTRACT)
    m511 = strict_load(M511_CONTRACT)
    check("strict_json_duplicate_and_nonfinite_token_rejection")

    require(data["schema"] == "m649_h67_ep35_convtranspose_typed_numeric_audit_v1", "schema drift")
    require(data["status"] == "PASS_NUMERIC_AUDIT__NO_GO_EXACT_TYPED_SPLIT", "status drift")
    require((RESULT / "RUN_COMPLETE.txt").read_text(encoding="utf-8").strip() ==
            "PASS_M649_NUMERIC_AUDIT__NO_GO_EXACT_TYPED_SPLIT", "completion receipt drift")
    require(data["identity"]["contract"]["sha256"] == sha256(CONTRACT), "contract identity drift")
    require(data["identity"]["contract"]["sha256"] == "580ddee0e52ef325df5ba73ed799dcd4a6b6fb25e94123428230cf752f405b5b",
            "unexpected contract cut")
    check("schema_status_contract_cut")

    expected_inputs = set(contract["required_input_names"])
    require(set(data["identity"]["inputs"]) == expected_inputs, "result input identity set mismatch")
    for name, item in data["identity"]["inputs"].items():
        check_regular_identity(item, "input:" + name)
        require(item["sha256"] == contract["inputs"][name]["sha256"], "contract/result input SHA mismatch: " + name)
    require(sha256(DOCS359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "docs/359 drift")
    require(data["identity"]["checkpoint_load_audit"]["missing_count"] == 0 and
            data["identity"]["checkpoint_load_audit"]["unexpected_count"] == 0 and
            data["identity"]["checkpoint_load_audit"]["overlay_missing_count"] == 0 and
            data["identity"]["checkpoint_load_audit"]["overlay_unexpected_count"] == 0,
            "checkpoint exact-load audit failed")
    check("24_input_hashes_checkpoint_load_and_docs359")

    raw = data["raw_validation_sources"]
    check_regular_identity(raw["sequence_list"], "raw sequence list")
    require(len(raw["samples"]) == 10, "raw source sample count")
    expected_keys = contract["frozen_workload"]["sample_order"]
    for sample_id, row in enumerate(raw["samples"]):
        require(row["sample_id"] == sample_id and row["sample_key"] == expected_keys[sample_id],
                "raw source sample identity/order mismatch")
        require(set(row["files"]) == {"event", "flow", "mask"}, "raw source file set mismatch")
        for kind, item in row["files"].items():
            check_regular_identity(item, "sample{}:{}".format(sample_id, kind))
    check("strict_s10_raw_sample_source_hashes")

    require(not M511_CANONICAL.exists(), "forbidden M511 canonical exists")
    failed = Path(data["identity"]["failed_m511_state"]["failed_staging_directory"])
    require(sorted(str(p.relative_to(failed)) for p in failed.rglob("*") if p.is_file()) ==
            ["FAILED.json", "calls/s00_d0.activation.le.bitpack"], "M511 failed staging population drift")
    prior = data["identity"]["prior_failed_m649_execution"]
    prior_dir = Path(prior["failed_staging_directory"])
    require(sorted(p.name for p in prior_dir.iterdir()) == ["FAILED.json"], "prior M649 failure population drift")
    require(sha256(prior_dir / "FAILED.json") == prior["failed_receipt_sha256"], "prior M649 failure SHA drift")
    check("failed_m511_and_prior_m649_states_preserved")

    modules = m511["modules"]
    require(len(modules) == 4 and set(data["module_identities"]) == {row["name"] for row in modules},
            "module identity set mismatch")
    for row in modules:
        ident = data["module_identities"][row["name"]]
        require(ident["operator"] == "ConvTranspose2d" and ident["in_channels"] == row["in_channels"] and
                ident["out_channels"] == row["out_channels"] and ident["kernel_size"] == row["kernel_size"] and
                ident["stride"] == row["stride"] and ident["padding"] == row["padding"] and
                ident["output_padding"] == row["output_padding"], "module parameter identity mismatch")
        weight = ident["weight"]
        require(weight["dtype"] == "torch.float32" and weight["layout"] == "C_ORDER_CONTIGUOUS" and
                weight["byte_order"] == "little" and len(weight["content_sha256"]) == 64,
                "module weight identity malformed")
        require(weight["content_bytes"] == product(weight["shape"]) * 4,
                "module weight byte/shape mismatch")
    require(len({v["weight"]["content_sha256"] for v in data["module_identities"].values()}) == 4,
            "module weight identities not unique")
    check("four_convtranspose_module_and_weight_identities")

    records = data["records"]
    require(data["population"] == {"samples": 10, "modules": 4, "records": 40}, "population summary drift")
    require(len(records) == 40, "record count drift")
    expected_lattice = [(sample, module) for sample in range(10) for module in range(4)]
    observed_lattice = [(row["sample_id"], row["module_index"]) for row in records]
    require(observed_lattice == expected_lattice and len(set(observed_lattice)) == 40,
            "record lattice is not exact unique ordered 10x4")
    module_totals = {}
    for record_index, row in enumerate(records):
        sample_id, module_index = expected_lattice[record_index]
        expected_module = modules[module_index]
        require(row["sample_key"] == expected_keys[sample_id] and row["sequence_key"] == "zurich_city_09_a",
                "record sample identity drift")
        require(row["name"] == expected_module["name"] and row["operator"] == "ConvTranspose2d",
                "record module identity drift")
        numeric = row["input_numeric"]
        require(numeric["dtype"] == "torch.float32" and numeric["device_type"] == "cuda" and
                numeric["channel_axis"] == 2 and numeric["is_contiguous"] is True,
                "record runtime numeric identity drift")
        require(numeric["shape"] == expected_module["input_shape"] and row["output_shape"] == expected_module["output_shape"] and
                row["output_dtype"] == "torch.float32", "record shape/dtype drift")
        channels = numeric["per_channel_exactness"]
        require(len(channels) == expected_module["in_channels"], "per-channel population drift")
        per_channel_elements = product(numeric["shape"][:2] + numeric["shape"][3:])
        for channel, channel_row in enumerate(channels):
            require(channel_row["channel"] == channel and channel_row["elements"] == per_channel_elements,
                    "per-channel order/elements drift")
            validate_summary(channel_row, "s{}d{}c{}".format(sample_id, module_index, channel))
        full_expected = aggregate_channels(channels)
        validate_summary(numeric["full_tensor"], "s{}d{} full".format(sample_id, module_index), expected_module["in_channels"])
        compare_summary(numeric["full_tensor"], full_expected, "s{}d{} full aggregate".format(sample_id, module_index))
        typed = numeric["typed_partition"]
        if module_index == 0:
            compare_summary(typed["binary"], full_expected, "s{}d0 typed binary".format(sample_id))
            require(typed["gate_pass"] == full_expected["all_exact_binary"], "d0 gate mismatch")
        else:
            first = typed["first2_flow_hypothesis"]
            last = typed["last2_flow_hypothesis"]
            compare_summary(first["flow_summary"], aggregate_channels(channels[:2]), "first2 flow")
            compare_summary(first["binary_suffix_summary"], aggregate_channels(channels[2:]), "first2 suffix")
            compare_summary(last["binary_prefix_summary"], aggregate_channels(channels[:-2]), "last2 prefix")
            compare_summary(last["flow_summary"], aggregate_channels(channels[-2:]), "last2 flow")
            first_gate = (first["binary_suffix_summary"]["all_exact_binary"] and
                          first["flow_summary"]["all_finite"] and
                          first["flow_summary"]["nonbinary_finite_count"] > 0)
            last_gate = (last["binary_prefix_summary"]["all_exact_binary"] and
                         last["flow_summary"]["all_finite"] and
                         last["flow_summary"]["nonbinary_finite_count"] > 0)
            require(first["gate_pass"] == first_gate and last["gate_pass"] == last_gate,
                    "typed partition gate mismatch")
            for hypothesis in (first, last):
                for stats in hypothesis["flow_channel_safe_stats"]:
                    require(stats["nonfinite_count"] == stats["nan_count"] + stats["positive_infinity_count"] +
                            stats["negative_infinity_count"], "safe-stat nonfinite decomposition mismatch")
                    for key, value in stats.items():
                        if isinstance(value, float):
                            require(math.isfinite(value), "nonfinite serialized safe statistic")
        total = module_totals.setdefault(module_index, {key: 0 for key in COUNT_KEYS})
        for key in COUNT_KEYS:
            total[key] += full_expected[key]
    check("40_record_lattice_per_channel_and_partition_recomputation")

    expected_totals = {
        0: {"elements": 46080000, "zero_count": 37783828, "one_count": 8296172,
            "exact_binary_count": 46080000, "nonbinary_finite_count": 0, "integer_count": 46080000, "nonfinite_count": 0},
        1: {"elements": 92400000, "zero_count": 75314174, "one_count": 0,
            "exact_binary_count": 75314174, "nonbinary_finite_count": 17085826, "integer_count": 75314174, "nonfinite_count": 0},
        2: {"elements": 185280000, "zero_count": 153544434, "one_count": 31735566,
            "exact_binary_count": 185280000, "nonbinary_finite_count": 0, "integer_count": 185280000, "nonfinite_count": 0},
        3: {"elements": 372480000, "zero_count": 267646872, "one_count": 104833128,
            "exact_binary_count": 372480000, "nonbinary_finite_count": 0, "integer_count": 372480000, "nonfinite_count": 0},
    }
    require(module_totals == expected_totals, "independent module totals drift")
    for row in records:
        binary_channels = sum(item["all_exact_binary"] for item in row["input_numeric"]["per_channel_exactness"])
        if row["module_index"] == 1:
            require(binary_channels == 0, "d1 unexpectedly has a full-binary channel")
        else:
            require(binary_channels == len(row["input_numeric"]["per_channel_exactness"]),
                    "d0/d2/d3 unexpectedly has a nonbinary channel")
    check("exact_per_module_numeric_facts")

    partial = Path(data["identity"]["inputs"]["m511_failed_d0_partial_payload"]["path"])
    require(partial.stat().st_size * 8 == records[0]["input_numeric"]["full_tensor"]["elements"],
            "M511 d0 partial bit population mismatch")
    require(popcount(partial) == records[0]["input_numeric"]["full_tensor"]["one_count"] == 839586,
            "M511 partial d0 popcount cross-check failed")
    check("preserved_m511_s00_d0_payload_popcount_crosscheck")

    global_go = all(
        row["input_numeric"]["typed_partition"]["gate_pass"]
        if row["module_index"] == 0 else
        row["input_numeric"]["typed_partition"]["first2_flow_hypothesis"]["gate_pass"]
        for row in records)
    require(global_go is False, "global gate unexpectedly passes")
    decision = data["decision"]
    require(decision["status"] == "NO_GO_EXACT_TYPED_SPLIT" and decision["typed_split_authorized"] is False and
            decision["authorized_layout"] is None and
            decision["first2_flow_hypothesis_all_modules_pass"] is False and
            decision["last2_flow_hypothesis_all_modules_pass"] is False,
            "NO_GO decision mismatch")
    expected_decision_checks = [{
        "id": "POPULATION_10X4", "pass": len(records) == 40,
        "expected": 40, "observed": len(records),
    }]
    for row in records:
        prefix = "S{:02d}_D{}".format(row["sample_id"], row["module_index"])
        expected_decision_checks.append({
            "id": prefix + "_DTYPE_FLOAT32",
            "pass": row["input_numeric"]["dtype"] == "torch.float32",
        })
        if row["module_index"] == 0:
            expected_decision_checks.append({
                "id": prefix + "_ALL_BINARY",
                "pass": row["input_numeric"]["full_tensor"]["all_exact_binary"],
            })
        else:
            expected_decision_checks.append({
                "id": prefix + "_FIRST2_FLOW_SUFFIX_BINARY",
                "pass": row["input_numeric"]["typed_partition"]["first2_flow_hypothesis"]["gate_pass"],
            })
    for module_index in range(4):
        count = sum(row["module_index"] == module_index for row in records)
        expected_decision_checks.append({
            "id": "D{}_S10_POPULATION".format(module_index), "pass": count == 10,
            "expected": 10, "observed": count,
        })
    for module_index in range(1, 4):
        module_rows = [row for row in records if row["module_index"] == module_index]
        expected_decision_checks.append({
            "id": "D{}_S10_FIRST2_FLOW_TYPED_SPLIT".format(module_index),
            "observed_nonbinary_finite": sum(
                row["input_numeric"]["typed_partition"]["first2_flow_hypothesis"]["flow_summary"]
                ["nonbinary_finite_count"] for row in module_rows),
            "pass": all(row["input_numeric"]["typed_partition"]["first2_flow_hypothesis"]["gate_pass"]
                        for row in module_rows),
        })
        expected_decision_checks.append({
            "admission_role": "DIAGNOSTIC_ONLY_NOT_SOURCE_EXPECTED",
            "id": "D{}_S10_LAST2_FLOW_HYPOTHESIS".format(module_index),
            "observed_nonbinary_finite": sum(
                row["input_numeric"]["typed_partition"]["last2_flow_hypothesis"]["flow_summary"]
                ["nonbinary_finite_count"] for row in module_rows),
            "pass": all(row["input_numeric"]["typed_partition"]["last2_flow_hypothesis"]["gate_pass"]
                        for row in module_rows),
        })
    if decision["checks"] != expected_decision_checks:
        mismatch = next((index for index, pair in enumerate(zip(decision["checks"], expected_decision_checks))
                         if pair[0] != pair[1]), min(len(decision["checks"]), len(expected_decision_checks)))
        fail("decision vector mismatch at {}: observed={} expected={}".format(
            mismatch,
            decision["checks"][mismatch] if mismatch < len(decision["checks"]) else "<missing>",
            expected_decision_checks[mismatch] if mismatch < len(expected_decision_checks) else "<missing>"))
    check("global_no_go_decision_recomputed")

    boundary = data["claim_boundary"]
    require(boundary["numeric_audit"] is True and boundary["typed_split_authorized"] is False,
            "numeric claim boundary mismatch")
    for key in ("activation_payload", "cycles", "speedup", "rtl", "vcs", "synopsys", "energy", "ppa",
                "system_speedup", "date_headline"):
        require(boundary[key] is False, "forbidden claim enabled: " + key)
    require(RESULT_JSON.stat().st_size == 11828216 and not any(p.suffix in (".bin", ".bitpack", ".npy", ".npz")
            for p in RESULT.rglob("*") if p.is_file()), "activation payload or result-size drift")
    check("no_payload_cycles_speedup_rtl_or_eda_authority")

    runtime_receipt_present = "runtime" in data["identity"]
    result = {
        "status": "PASS_WITH_P2_RUNTIME_PROVENANCE_GAP",
        "score": 99,
        "p0": 0,
        "p1": 0,
        "p2": 1,
        "checks_passed": checks,
        "runtime_receipt_present": runtime_receipt_present,
        "cuda_evidence": data["cuda_synchronization"],
        "module_totals": expected_totals,
        "d1_exact_binary_fraction": expected_totals[1]["exact_binary_count"] / expected_totals[1]["elements"],
        "d1_nonbinary_fraction": expected_totals[1]["nonbinary_finite_count"] / expected_totals[1]["elements"],
        "decision": "NO_GO_EXACT_TYPED_SPLIT_CORRECT",
        "next_boundary": "NEW_CONTRACT_AND_FRESH_STATIC_HAMMER_FOR_D0_D2_D3_ONLY; NO_CAPTURE_CYCLES_RTL_OR_EDA",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
