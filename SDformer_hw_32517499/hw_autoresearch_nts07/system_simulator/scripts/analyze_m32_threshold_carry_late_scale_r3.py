#!/usr/bin/env python3
"""Close the H67 ep35/s10 real-domain M32 semantic dataflow admission."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = (
    ROOT / "hw_autoresearch_nts07/contracts/"
    "m32_threshold_carry_input_contract_r3_20260822.json"
)
IDENTITY_FIELDS = [
    "same_tensor_object",
    "same_storage_pointer",
    "same_data_pointer",
    "same_storage_offset",
    "same_stride",
    "same_dtype",
    "same_device",
    "same_shape",
    "same_numel",
    "same_logical_nbytes",
    "same_value_digest",
    "identity_admitted",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(raw):
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_contract(contract_path):
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if (
        contract.get("schema") != "m32_threshold_carry_input_contract_v3"
        or contract.get("status")
        != "FROZEN_R2_PLUS_H67_EP35_S10_DYNAMIC_DATAFLOW_IDENTITY"
    ):
        raise ValueError("unexpected M32 r3 input contract")
    paths = {}
    hashes = {}
    for name, spec in sorted(contract["inputs"].items()):
        path = resolve_path(spec["path"])
        if not path.is_file():
            raise ValueError("missing M32 r3 input {}: {}".format(name, path))
        actual = sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(
                "M32 r3 input hash drift for {}: {} != {}".format(
                    name, actual, spec["sha256"]
                )
            )
        paths[name] = path
        hashes[name] = actual
    return contract, paths, hashes


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(
                    "invalid M32 dataflow JSONL line {}: {}".format(
                        line_number, exc
                    )
                )
    return rows


def ordered_sample_digest(workload_rows):
    digest = hashlib.sha256()
    for row in workload_rows:
        digest.update(
            ("{}\t{}\t{}\n".format(
                row["sample_id"], row["sample_key"], row["sequence_key"]
            )).encode("utf-8")
        )
    return digest.hexdigest()


def shape_numel(shape):
    total = 1
    for value in shape:
        total *= int(value)
    return total


def audit_dynamic_identity(r2, manifest, rows, profile, workload_rows, hashes):
    samples = int(r2["identity"]["samples"])
    candidates = r2["candidate_census"]["candidates"]
    if samples != 10 or len(candidates) != 10:
        raise ValueError("M32 r3 requires frozen ten-by-ten population")
    if (
        manifest.get("status")
        != "PASS_EXACT_PRODUCER_CONSUMER_TENSOR_IDENTITY"
        or manifest.get("failures") != []
        or int(manifest.get("records", -1)) != samples * len(candidates)
        or int(manifest.get("identity_admitted_records", -1))
        != samples * len(candidates)
        or int(manifest.get("same_tensor_object_records", -1))
        != samples * len(candidates)
        or int(manifest.get("root_forwards", -1)) != samples
        or manifest["rows"]["sha256"] != hashes["dataflow_rows"]
        or manifest["candidate_report"]["sha256"] != hashes["r2_report"]
        or manifest["run_identity"]["profile_script_sha256"]
        != hashes["runtime_profile_source"]
        or manifest["run_identity"]["trace_contract_sha256"]
        != hashes["trace_contract"]
        or manifest["run_identity"]["writer_script_sha256"]
        != hashes["trace_writer"]
        or manifest["run_identity"]["wrapper_script_sha256"]
        != hashes["trace_wrapper"]
        or not manifest["instrumentation"]["instrumentation_intrusive"]
    ):
        raise ValueError("M32 dynamic manifest admission drift")

    postrun = manifest["postrun_evidence"]
    if (
        postrun.get("status")
        != "PASS_FROZEN_POSTRUN_PROFILE_AND_SAMPLE_IDENTITY"
        or postrun["profile_json"]["sha256"] != hashes["postrun_profile"]
        or postrun["sample_workload"]["sha256"] != hashes["sample_workload"]
    ):
        raise ValueError("M32 postrun manifest binding drift")
    artifact = profile["artifact_identity"]
    load_audit = profile["checkpoint_load_audit"]
    if (
        int(profile["samples"]) != samples
        or artifact["checkpoint_sha256"] != r2["identity"]["checkpoint_sha256"]
        or artifact["checkpoint_sha256"]
        != manifest["run_identity"]["checkpoint_sha256"]
        or artifact["config_sha256"]
        != manifest["run_identity"]["config_sha256"]
        or any(int(load_audit[field]) != 0 for field in (
            "missing_count", "unexpected_count",
            "overlay_missing_count", "overlay_unexpected_count",
        ))
    ):
        raise ValueError("M32 postrun profile/checkpoint identity drift")

    if (
        len(workload_rows) != samples
        or [int(row["sample_id"]) for row in workload_rows]
        != list(range(samples))
        or ordered_sample_digest(workload_rows)
        != postrun["sample_workload"]["ordered_sample_identity_sha256"]
        or [row["sample_key"] for row in workload_rows]
        != postrun["sample_workload"]["sample_keys"]
        or [row["sequence_key"] for row in workload_rows]
        != postrun["sample_workload"]["sequence_keys"]
    ):
        raise ValueError("M32 ordered sample identity drift")

    expected_pairs = {
        (row["producer"], row["name"]): row for row in candidates
    }
    observed_grid = set()
    digests_by_pair = {key: set() for key in expected_pairs}
    for row in rows:
        pair = (row["producer"], row["consumer"])
        if pair not in expected_pairs:
            raise ValueError("unexpected M32 dynamic producer/consumer pair")
        sample_id = int(row["sample_id"])
        grid_key = (sample_id, pair)
        if sample_id < 0 or sample_id >= samples or grid_key in observed_grid:
            raise ValueError("M32 dynamic sample/pair grid drift")
        observed_grid.add(grid_key)
        if (
            int(row["producer_call_index"]) != sample_id
            or int(row["consumer_call_index"]) != sample_id
            or any(row.get(field) is not True for field in IDENTITY_FIELDS)
            or row["producer_raw_value_sha256"]
            != row["consumer_raw_value_sha256"]
            or len(row["producer_raw_value_sha256"]) != 64
            or int(row["consumer_output_numel"])
            != int(expected_pairs[pair]["output_elements_per_sample"])
            or int(row["expected_consumer_output_numel"])
            != int(expected_pairs[pair]["output_elements_per_sample"])
            or shape_numel(row["shape"]) != int(row["numel"])
        ):
            raise ValueError("M32 dynamic identity row admission drift")
        digests_by_pair[pair].add(row["producer_raw_value_sha256"])
    expected_grid = {
        (sample_id, pair)
        for sample_id in range(samples)
        for pair in expected_pairs
    }
    if observed_grid != expected_grid:
        raise ValueError("M32 dynamic grid population mismatch")
    if any(len(values) != samples for values in digests_by_pair.values()):
        raise ValueError("M32 per-pair sample digest diversity drift")
    return {
        "status": "PASS_EXACT_TEN_BY_TEN_RUNTIME_TENSOR_IDENTITY",
        "samples": samples,
        "candidate_pairs": len(expected_pairs),
        "records": len(rows),
        "ordered_sample_identity_sha256": ordered_sample_digest(workload_rows),
        "all_identity_fields_true": True,
        "all_consumer_output_populations_match": True,
        "all_pairs_have_ten_distinct_value_digests": True,
        "instrumentation_intrusive": True,
        "performance_use_forbidden": True,
    }


def build_report(contract_path=DEFAULT_CONTRACT):
    contract, paths, hashes = load_contract(contract_path)
    r2 = json.loads(paths["r2_report"].read_text(encoding="utf-8"))
    if (
        r2.get("schema") != "m32_threshold_carry_late_scale_audit_v2"
        or r2.get("semantic_admission") is not False
        or r2.get("headline_admitted") is not False
    ):
        raise ValueError("unexpected M32 r2 claim boundary")
    manifest = json.loads(paths["dataflow_manifest"].read_text(encoding="utf-8"))
    rows = read_jsonl(paths["dataflow_rows"])
    profile = json.loads(paths["postrun_profile"].read_text(encoding="utf-8"))
    with paths["sample_workload"].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        workload_rows = list(csv.DictReader(handle))
    dynamic_audit = audit_dynamic_identity(
        r2, manifest, rows, profile, workload_rows, hashes
    )

    candidates = []
    for row in r2["candidate_census"]["candidates"]:
        admitted = dict(row)
        admitted["candidate_status"] = (
            "ADMITTED_H67_EP35_S10_EXACT_RUNTIME_TENSOR_REAL_DOMAIN_ONLY"
        )
        admitted["semantic_admission"] = True
        admitted["semantic_scope"] = (
            "frozen H67 ep35 ten-sample workload; exact runtime tensor identity; "
            "real-domain W(theta*b)+bias=theta*(W*b)+bias only"
        )
        candidates.append(admitted)

    product_oracle = dict(r2["signed_product_oracle"])
    product_oracle["status"] = (
        "ALGORITHMIC_FULL_DOMAIN_SIGNED_INTEGER_PRODUCT_IDENTITY_"
        "PLUS_4152_CASE_REGRESSION_PIPELINE_PENDING"
    )
    return {
        "schema": "m32_threshold_carry_late_scale_audit_v3",
        "status": (
            "PASS_H67_EP35_S10_EXACT_RUNTIME_DATAFLOW_REAL_DOMAIN_"
            "SEMANTIC_ADMISSION_ONLY"
        ),
        "identity": {
            "input_contract": str(Path(contract_path).resolve()),
            "input_contract_sha256": sha256(contract_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
            "verified_input_sha256": hashes,
            "checkpoint_sha256": r2["identity"]["checkpoint_sha256"],
            "samples": dynamic_audit["samples"],
        },
        "dynamic_dataflow_audit": dynamic_audit,
        "candidate_census": {
            "semantically_admitted_operators": len(candidates),
            "semantically_admitted_cycles_candidate_population": int(
                r2["candidate_census"]["candidate_factorable_cycles"]
            ),
            "semantically_admitted_outputs_per_sample": int(
                r2["candidate_census"]["candidate_factorable_outputs_per_sample"]
            ),
            "candidates": candidates,
            "continuous_preserved": r2["candidate_census"]["continuous_preserved"],
        },
        "checkpoint_threshold_audit": r2["checkpoint_threshold_audit"],
        "signed_product_oracle": product_oracle,
        "control_charged_cycle_sensitivity": r2[
            "control_charged_cycle_sensitivity"
        ],
        "admission": {
            "semantic_admission": True,
            "semantic_scope": (
                "H67 ep35 frozen ten-sample real-domain factorization only"
            ),
            "semantic_generalization_admitted": False,
            "fixed_point_admitted": False,
            "rtl_admitted": False,
            "system_cycle_admitted": False,
            "performance_admitted": False,
            "ppa_admitted": False,
            "power_energy_admitted": False,
            "headline_admitted": False,
        },
        "claim_boundary": {
            "permitted": [
                "ten H67 ep35 candidate consumers read the exact ATLIF producer tensor",
                "real-domain threshold-carry algebra for the frozen ten-sample workload",
                "constructive signed Acc32-by-signed-Q24 integer decomposition identity",
            ],
            "forbidden": [
                "generalizing semantic admission beyond H67 ep35 s10",
                "calling the checkpoint float32 threshold signed-Q24 bit exact",
                "claiming RNE, saturation, bias, SRAM, or executable RTL closure",
                "claiming any sensitivity row as measured cycle, speedup, FPS, PPA, power, or energy",
            ],
        },
        "semantic_admission": True,
        "headline_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite M32 r3 report: {}".format(args.output))
    report = build_report(args.contract.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
