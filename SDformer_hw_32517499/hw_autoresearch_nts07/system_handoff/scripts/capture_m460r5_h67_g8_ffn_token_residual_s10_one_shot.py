#!/usr/bin/env python3
"""One-shot-authorized M460R5 H67 G8 post-compute opportunity capture.

R5 repairs the R4 advisory/base interface by constructing the exact observed
keys consumed by the frozen M460 base.  It keeps code and immutable data roots
separate and emits reduction-only, strictly ordered, double-sealed evidence.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


CODE_REPO = Path(__file__).resolve().parents[3]
HW = CODE_REPO / "hw_autoresearch_nts07"
R3_PATH = (HW / "system_handoff/scripts/"
           "capture_m460r3_h67_g8_ffn_token_residual_s10.py")
RECEIPT_BINDING_FIELDS = (
    "launch_outer_seal_sha256",
    "capture_summary_sha256",
    "capture_inner_manifest_sha256",
    "capture_outer_seal_file_sha256",
)
BASE_OBSERVED_KEYS = (
    "capture_script", "profile_script", "m40_loader", "checkpoint",
    "config", "sample_workload", "docs359", "swin_source",
)


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

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def load_file_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import file " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_capture_bases():
    r3 = load_file_module(R3_PATH, "m460r5_frozen_r3_capture")
    return r3, r3.BASE


def read_workload(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 10 and
            [int(row["sample_id"]) for row in rows] == list(range(10)) and
            all(row["sequence_key"] == "zurich_city_09_a" for row in rows),
            "M460R5 exact S10 workload drift")
    return rows


def roots_for_contract(contract):
    remote = contract["execution_roots"]["remote"]
    if str(CODE_REPO) == remote["code_repo"]:
        return {
            "code_repo": Path(remote["code_repo"]),
            "immutable_data_repo": Path(remote["immutable_data_repo"]),
        }
    return {"code_repo": CODE_REPO, "immutable_data_repo": CODE_REPO}


def resolve_identity(record, roots):
    path = Path(record["path"])
    if record["root"] == "code_repo":
        return roots["code_repo"] / path
    if record["root"] == "code_hw":
        return roots["code_repo"] / "hw_autoresearch_nts07" / path
    if record["root"] == "immutable_data_repo":
        return roots["immutable_data_repo"] / path
    if record["root"] == "absolute":
        require(path.is_absolute(), "absolute identity path is relative")
        return path
    raise RuntimeError("unknown identity root: " + str(record["root"]))


def build_base_observed(identity_observed):
    require(all(name in identity_observed for name in BASE_OBSERVED_KEYS),
            "M460R5 compatibility mapping lacks a frozen base key")
    observed = {name: identity_observed[name]
                for name in BASE_OBSERVED_KEYS}
    require("profile" not in observed and "capture_advisory" not in observed,
            "R4 advisory-only names leaked into base observed mapping")
    require(set(BASE_OBSERVED_KEYS).issubset(set(observed)),
            "base observed compatibility population drift")
    return observed


def validate_contract(contract_path):
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m460r5_h67_g8_one_shot_capture_contract_v1",
            "M460R5 contract schema drift")
    require(contract.get("status") ==
            "ONE_SHOT_CAPTURE_AUTHORIZED_BY_M460R4_INDEPENDENT_HAMMER",
            "M460R5 contract status drift")
    authorization = contract["authorization"]
    require(authorization["one_shot_g8_s10_capture"] is True and
            int(authorization["maximum_capture_attempts"]) == 1 and
            authorization["training"] is False,
            "M460R5 one-shot authorization drift")
    require(tuple(contract["post_capture_receipt_fields"]) ==
            RECEIPT_BINDING_FIELDS,
            "M460R5 post-capture receipt field drift")

    roots = roots_for_contract(contract)
    identity_observed = {}
    for name, record in contract["identity"].items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        path = resolve_identity(record, roots).resolve()
        require(path.is_file(), "M460R5 identity absent: " + name)
        actual = sha256(path)
        require(actual == record["sha256"],
                "M460R5 identity SHA drift: " + name)
        identity_observed[name] = {"path": str(path), "sha256": actual}
    require(sha256(Path(__file__).resolve()) ==
            contract["identity"]["capture_script"]["sha256"],
            "M460R5 capture self SHA drift")

    source = Path(identity_observed["swin_source"]["path"]).read_text(
        encoding="utf-8")
    for fragment in (
            "class MS_Spiking_Mlp(Spiking_Mlp):",
            "x = self.sn1(x)", "x = self.fc1(x)", "x= self.bn1(",
            "x = self.sn2(x)", "x = self.fc2(x)", "x = self.bn2(",
            "self.mlp(x.permute(1,0,2,3,4)).permute(1,0,2,3,4)"):
        require(fragment in source,
                "M460R5 FFN topology source drift: " + fragment)
    workload = read_workload(identity_observed["sample_workload"]["path"])
    observed = build_base_observed(identity_observed)
    return contract, observed, workload, roots, identity_observed


def apply_r5_summary(summary, contract):
    require(summary.get("schema") ==
            "m460_h67_g8_ffn_token_residual_s10_capture_v1",
            "M460R5 frozen base summary schema drift")
    require(isinstance(summary.get("admission"), dict),
            "M460R5 base summary admission absent")
    value = dict(summary)
    value["schema"] = "m460r5_h67_g8_one_shot_capture_v1"
    value["status"] = (
        "PASS_M460R5_H67_EP35_NO_RUNNING_S10_ONE_SHOT_POSTCOMPUTE_ORACLE")
    value["strict_runtime_state_machine"] = {
        "order": ["pre", "sn1", "sn2", "fc2", "full_output"],
        "per_module_per_sample": "exactly once",
        "sn2_fc2_sn1_attack_accepted": False,
    }
    value["one_shot_authorization"] = {
        "review_outer_seal_file_sha256": contract[
            "review_authorization"]["outer_seal_file_sha256"],
        "maximum_capture_attempts": 1,
        "runner_consumed_marker_required": True,
    }
    value["result_sealing"] = {
        "inner_manifest": "manifest.sha256",
        "outer_seal": "manifest.sha256.outer.seal.sha256",
        "receipt_binding_required": list(RECEIPT_BINDING_FIELDS),
    }
    admission = dict(value["admission"])
    admission.update({
        "checkpoint_bound_s10_postcompute_oracle": True,
        "postcompute_opportunity_counts": True,
        "strict_hook_order": True,
        "double_sealed_payload": True,
        "token_skip_rate": False,
        "executable_skip": False,
        "delta_aee": False,
        "valid825_accuracy": False,
        "cycle_speedup": False,
        "energy": False,
        "ppa": False,
        "system_speedup": False,
        "headline": False,
        "training": False,
    })
    value["admission"] = admission
    value["claim_boundary"] = (
        "Frozen H67-ep35/no-running S10 reduction-only post-compute "
        "opportunity/oracle evidence. No executable skip, Delta-AEE, "
        "valid825 accuracy, cycle, energy, PPA, system speedup or headline "
        "is admitted.")
    return value


def finalize_capture_payload(output_dir, contract_path):
    output_dir = Path(output_dir).resolve()
    contract = strict_json(contract_path)
    summary_path = output_dir / "m460_h67_g8_ffn_token_residual_s10_capture.json"
    summary = apply_r5_summary(strict_json(summary_path), contract)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    npz = sorted(output_dir.glob("*.npz"))
    require(len(npz) == 120, "M460R5 requires exactly 120 reduction NPZ")
    evidence = npz + [
        output_dir / "samples.csv",
        output_dir / "per_sample_module_manifest.json",
        summary_path,
    ]
    require(all(path.is_file() for path in evidence),
            "M460R5 payload evidence population incomplete")
    inner = output_dir / "manifest.sha256"
    inner.write_text("".join(
        "{}  {}\n".format(sha256(path), path.name) for path in evidence),
        encoding="utf-8")
    outer = output_dir / "manifest.sha256.outer.seal.sha256"
    outer.write_text("{}  {}\n".format(sha256(inner), inner.name),
                     encoding="utf-8")
    require(tuple(summary["result_sealing"]["receipt_binding_required"]) ==
            tuple(contract["post_capture_receipt_fields"]),
            "M460R5 summary/contract receipt fields diverged")
    return {"summary": summary_path, "inner": inner, "outer": outer}


def execute_with_backend(contract_path, output_dir, validated, r3, base):
    contract, observed, _workload, roots, _identity = validated
    require(set(BASE_OBSERVED_KEYS).issubset(set(observed)),
            "M460R5 base observed compatibility keys missing")
    original_validate = base.validate_contract
    original_class = base.FFNResidualStreamCapture
    original_file = base.__file__
    original_load_module = base.load_module
    original_resolve = base.resolve_path

    def resolve_by_frozen_identity(path_text):
        path = Path(path_text)
        for record in contract["identity"].values():
            if isinstance(record, dict) and record.get("path") == path_text:
                return resolve_identity(record, roots).resolve()
        if path.is_absolute():
            return path
        return original_resolve(path_text)

    def load_with_immutable_data_split(path, name):
        module = original_load_module(path, name)
        if name == "m460_profile":
            original_load_config = module.load_config

            def load_config_with_data_root(config_path):
                config, device = original_load_config(config_path)
                config["data"]["path"] = str(
                    Path(roots["immutable_data_repo"]) /
                    contract["immutable_data_runtime"]["dataset_root"])
                return config, device
            module.load_config = load_config_with_data_root
        return module

    try:
        base.validate_contract = lambda _path: (contract, observed, _workload)
        base.FFNResidualStreamCapture = r3.StrictFFNResidualStreamCapture
        base.__file__ = str(Path(__file__).resolve())
        base.load_module = load_with_immutable_data_split
        base.resolve_path = resolve_by_frozen_identity
        base.execute(contract_path, output_dir)
    finally:
        base.validate_contract = original_validate
        base.FFNResidualStreamCapture = original_class
        base.__file__ = original_file
        base.load_module = original_load_module
        base.resolve_path = original_resolve
    return finalize_capture_payload(output_dir, contract_path)


def dry_run(contract_path):
    contract, observed, workload, roots, identity = validate_contract(
        contract_path)
    print(json.dumps({
        "schema": contract["schema"],
        "status": "PASS_M460R5_STATIC_ONE_SHOT_AND_BASE_COMPATIBILITY_DRY_RUN",
        "identity_inputs": len(identity),
        "base_observed_keys": sorted(observed),
        "samples": len(workload),
        "code_repo": str(roots["code_repo"]),
        "immutable_data_repo": str(roots["immutable_data_repo"]),
        "receipt_binding_required": list(RECEIPT_BINDING_FIELDS),
        "maximum_capture_attempts": 1,
        "gpu_touched": False,
        "model_constructed": False,
        "capture_launched": False,
        "training": False,
        "system_speedup": False,
    }, indent=2, sort_keys=True))


def execute(contract_path, output_dir):
    validated = validate_contract(contract_path)
    r3, base = load_capture_bases()
    finalized = execute_with_backend(
        contract_path, output_dir, validated, r3, base)
    print("PASS M460R5 summary={} inner={} outer={}".format(
        finalized["summary"], sha256(finalized["inner"]),
        sha256(finalized["outer"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    require(args.dry_run != (args.output_dir is not None),
            "choose exactly one of --dry-run or --output-dir")
    if args.dry_run:
        dry_run(args.contract)
    else:
        execute(args.contract, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
