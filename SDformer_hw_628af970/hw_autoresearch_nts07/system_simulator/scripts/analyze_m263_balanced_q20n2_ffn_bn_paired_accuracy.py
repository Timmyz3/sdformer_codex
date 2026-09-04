#!/usr/bin/env python3
"""Seal the paired M263 balanced-Q20 two-Newton downstream experiment."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RESULT = HW / "results/m263_balanced_q20n2_ffn_bn_paired_s10_r1_20260825"
MILESTONE = RESULT.name
PATHS = {
    "contract": HW / "contracts/m263_balanced_q20n2_ffn_bn_downstream_contract_r1_20260825.json",
    "evaluator": HW / "system_handoff/scripts/eval_m263_balanced_q20n2_ffn_bn_DSEC.py",
    "wrapper": HW / "system_handoff/run_m263_balanced_q20n2_ffn_bn_paired_eval_when_gpu_idle_20260825.sh",
    "precision_model": HW / "system_simulator/scripts/analyze_m263_dynamic_bn_precision_cost_dse.py",
    "precision_dse_seal": HW / "results/m263_dynamic_bn_precision_cost_dse_r1_20260825/SHA256SUMS",
    "m260_negative_seal": HW / "results/m260_m235_ffn_bn_paired_s10_r1_20260825/LOCAL_SHA256SUMS",
    "docs359": HW / "docs/359_DATE终局冻结_20260813.md",
}
EXPECTED = {
    "contract": "3224e696d5eba3083c92c412db761e1d0275a83616ba4cb2e2812bebf51f9ca1",
    "evaluator": "84a4915a81cae289ff909e76c867e350eee82b1d87cfd41770e093b7c488a71e",
    "wrapper": "03eaa696bf1a48c544e874292ec35bf45fb57823debe031c58ab7663ebfc087d",
    "precision_model": "bd5c8587c85f96e93b7dea18e6ca0e9c01898355abceea462fd89e1159737e32",
    "precision_dse_seal": "e5e2811583e99045bb41862b9e2b5a96ccec2ab6a938f31f7c11e8d7b4251094",
    "m260_negative_seal": "9852d3216445e0c9f4ef902706081806d3867cd5951f7842382b9f9cb201b1fe",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(
            handle, object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError("non-finite JSON: " + value)))


def validate_remote_manifest(path):
    checked = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        expected, remote_path = line.split("  ", 1)
        marker = "/" + MILESTONE + "/"
        require(marker in remote_path, "remote manifest path escaped result")
        relative = remote_path.split(marker, 1)[1]
        local_path = RESULT / relative
        observed = sha256(local_path)
        require(observed == expected, "remote payload SHA drift: " + relative)
        checked.append({"path": relative, "sha256": observed,
                        "bytes": local_path.stat().st_size})
    require(len(checked) == 11, "remote manifest population drift")
    return checked


def frames(mode):
    with (RESULT / mode / "per_frame.csv").open(
            "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 10, mode + " sample count drift")
    return rows


def paired_metric(reference, candidate, field):
    before = [float(row[field]) for row in reference]
    after = [float(row[field]) for row in candidate]
    delta = [b - a for a, b in zip(before, after)]
    before_mean = sum(before) / len(before)
    after_mean = sum(after) / len(after)
    return {
        "reference_mean": before_mean,
        "candidate_mean": after_mean,
        "mean_delta": sum(delta) / len(delta),
        "relative_delta_percent": (after_mean / before_mean - 1.0) * 100.0,
        "candidate_wins": sum(b < a for a, b in zip(before, after)),
        "candidate_losses": sum(b > a for a, b in zip(before, after)),
        "candidate_ties": sum(b == a for a, b in zip(before, after)),
        "maximum_absolute_paired_delta": max(abs(value) for value in delta),
    }


def main():
    observed = {name: sha256(path) for name, path in PATHS.items()}
    require(observed == EXPECTED, "M263 frozen identity drift")
    remote_manifest = RESULT / "SHA256SUMS"
    remote_manifest_sha = sha256(remote_manifest)
    require(remote_manifest_sha ==
            "67146fd2e00fcc3d0027a52d11a39de1cdd10bc139cb8739243bd6f07aae8990",
            "M263 remote manifest seal drift")
    payloads = validate_remote_manifest(remote_manifest)

    reference = frames("reference")
    candidate = frames("balanced_q20n2")
    ids = [(row["sequence"], row["file"]) for row in reference]
    require(ids == [(row["sequence"], row["file"]) for row in candidate] and
            len(set(ids)) == 10, "paired sample identity drift")

    ref_receipt = strict_json(RESULT / "reference/m263_runtime_receipt.json")
    can_receipt = strict_json(
        RESULT / "balanced_q20n2/m263_runtime_receipt.json")
    require(ref_receipt["status"] ==
            "PASS_M263_REFERENCE_EVALUATION_RUNTIME" and
            ref_receipt["samples"] == 10 and ref_receipt["hook_calls"] == 0,
            "reference receipt drift")
    require(can_receipt["status"] ==
            "PASS_M263_BALANCED_Q20N2_EVALUATION_RUNTIME" and
            can_receipt["samples"] == 10 and
            can_receipt["hook_calls"] == 240 and
            can_receipt["coefficient_pairs"] == 220800 and
            can_receipt["output_values"] == 4377600000 and
            can_receipt["precision"] == {
                "coefficient_frac": 20,
                "lut_entries": 32,
                "mean_frac": 18,
                "newton_steps": 2,
                "param_frac": 18,
                "per_segment": 16,
                "variance_frac": 20,
                "work_frac": 20,
            } and
            all(value == 0 for value in can_receipt["rail_counts"].values()),
            "candidate receipt drift")

    metrics = {name: paired_metric(reference, candidate, name)
               for name in ("AEE", "DSEC_Fl", "spikes")}
    gates = {
        "maximum_local_bn_output_delta": {
            "limit": 0.00025,
            "observed": can_receipt["maximum_abs_output_delta"],
            "pass": can_receipt["maximum_abs_output_delta"] <= 0.00025,
        },
        "paired_aee_relative_regression_percent": {
            "limit": 0.25,
            "observed": metrics["AEE"]["relative_delta_percent"],
            "pass": metrics["AEE"]["relative_delta_percent"] <= 0.25,
        },
        "paired_dsec_fl_relative_regression_percent": {
            "limit": 1.0,
            "observed": metrics["DSEC_Fl"]["relative_delta_percent"],
            "pass": metrics["DSEC_Fl"]["relative_delta_percent"] <= 1.0,
        },
    }
    require(gates["maximum_local_bn_output_delta"]["pass"] and
            not gates["paired_aee_relative_regression_percent"]["pass"] and
            gates["paired_dsec_fl_relative_regression_percent"]["pass"] and
            metrics["AEE"]["candidate_losses"] == 7,
            "M263 gate observation drift")

    output = {
        "schema": "m263_balanced_q20n2_ffn_bn_paired_accuracy_v1",
        "status": "PASS_PAIRED_S10_EXECUTION_AEE_GATE_FAILED",
        "identity": observed,
        "remote_payload_manifest_sha256": remote_manifest_sha,
        "remote_payloads": payloads,
        "population": {
            "samples": 10,
            "modules": 24,
            "coefficient_pairs": can_receipt["coefficient_pairs"],
            "bn_output_values": can_receipt["output_values"],
        },
        "coefficient_path_output_error": {
            "mean_absolute": can_receipt["mean_abs_output_delta"],
            "rmse": can_receipt["rmse_output_delta"],
            "maximum_absolute": can_receipt["maximum_abs_output_delta"],
            "rail_counts": can_receipt["rail_counts"],
        },
        "paired_metrics": metrics,
        "gates": gates,
        "comparison_to_m235_q16": {
            "q16_aee_relative_regression_percent": 1.7308688627234892,
            "q20n2_aee_relative_regression_percent":
                metrics["AEE"]["relative_delta_percent"],
            "absolute_percentage_point_recovery":
                1.7308688627234892 -
                metrics["AEE"]["relative_delta_percent"],
        },
        "finding": (
            "Balanced Q20 plus two Newton iterations reduces local error and "
            "recovers part of M235 Q16's AEE loss, but uniform approximation "
            "of all 24 FFN BN modules still fails the paired AEE gate."
        ),
        "next_experiment": (
            "Ablate by Swin stage and BN1/BN2, then keep high/exact fallback "
            "only for sensitive modules; uniform further widening is not "
            "justified before locating the threshold-amplifying sources."
        ),
        "admission": {
            "paired_first10": True,
            "local_error_gate": True,
            "dsec_fl_gate": True,
            "aee_gate": False,
            "downstream_accuracy_safe": False,
            "valid825_recommended": False,
            "rtl": False,
            "vcs": False,
            "dc": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    output_path = RESULT / "m263_balanced_q20n2_ffn_bn_paired_accuracy_r1.json"
    require(not output_path.exists(), "refusing to overwrite M263 analysis")
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": output["status"],
        "aee_relative_delta_percent": metrics["AEE"][
            "relative_delta_percent"],
        "dsec_fl_relative_delta_percent": metrics["DSEC_Fl"][
            "relative_delta_percent"],
        "maximum_local_delta": can_receipt["maximum_abs_output_delta"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
