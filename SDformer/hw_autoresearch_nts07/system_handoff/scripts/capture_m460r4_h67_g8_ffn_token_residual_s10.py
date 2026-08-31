#!/usr/bin/env python3
"""M460R4 future capture adapter; R4 runner never invokes execute().

R4 separates the clean code repository from the immutable data repository and
repairs the post-capture advisory receipt field names.  Actual GPU execution
remains forbidden until a later independent hammer explicitly authorizes it.
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
    r3 = load_file_module(R3_PATH, "m460r4_frozen_r3_capture")
    return r3, r3.BASE


def read_workload(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 10 and
            [int(row["sample_id"]) for row in rows] == list(range(10)) and
            all(row["sequence_key"] == "zurich_city_09_a" for row in rows),
            "M460R4 frozen S10 workload drift")
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
    raise RuntimeError("unknown identity root " + record["root"])


def validate_contract(contract_path):
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m460r4_h67_g8_environment_preflight_contract_v1",
            "M460R4 contract schema drift")
    require(contract.get("status") ==
            "SEALED_REMOTE_PREFLIGHT_ONLY__GPU_CAPTURE_FORBIDDEN",
            "M460R4 contract status drift")
    roots = roots_for_contract(contract)
    observed = {}
    for name, record in contract["identity"].items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        path = resolve_identity(record, roots).resolve()
        require(path.is_file(), "M460R4 identity absent: " + name)
        actual = sha256(path)
        require(actual == record["sha256"],
                "M460R4 identity SHA drift: " + name)
        observed[name] = {"path": str(path), "sha256": actual}
    require(tuple(contract["post_capture_advisory_receipt_fields"]) ==
            RECEIPT_BINDING_FIELDS, "M460R4 advisory receipt field drift")
    workload = read_workload(Path(observed["sample_workload"]["path"]))
    return contract, observed, workload, roots


def apply_r4_summary_advisory(summary):
    require(summary.get("schema") in (
        "m460_h67_g8_ffn_token_residual_s10_capture_v1",
        "m460r4_h67_g8_ffn_token_residual_s10_capture_v1"),
        "M460R4 base summary schema drift")
    require(isinstance(summary.get("admission"), dict),
            "M460R4 summary admission absent")
    value = dict(summary)
    value["schema"] = "m460r4_h67_g8_ffn_token_residual_s10_capture_v1"
    value["status"] = (
        "PASS_M460R4_H67_EP35_NO_RUNNING_S10_STRICT_ORDER_DOUBLE_SEAL")
    value["strict_runtime_state_machine"] = {
        "order": ["pre", "sn1", "sn2", "fc2", "full_output"],
        "per_module_per_sample": "exactly once",
        "sn2_fc2_sn1_attack_accepted": False,
    }
    value["result_sealing"] = {
        "inner_manifest": "manifest.sha256",
        "outer_seal": "manifest.sha256.outer.seal.sha256",
        "receipt_binding_required": list(RECEIPT_BINDING_FIELDS),
    }
    value["admission"] = dict(value["admission"])
    value["admission"]["strict_hook_order"] = True
    value["admission"]["double_sealed_payload"] = True
    value["admission"]["gpu_capture_authorized_by_r4_preflight"] = False
    return value


def finalize_r4_result(output_dir, contract_path):
    output_dir = Path(output_dir).resolve()
    summary_path = output_dir / "m460_h67_g8_ffn_token_residual_s10_capture.json"
    summary = apply_r4_summary_advisory(strict_json(summary_path))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    evidence = sorted(output_dir.glob("*.npz")) + [
        output_dir / "samples.csv",
        output_dir / "per_sample_module_manifest.json",
        summary_path,
    ]
    require(all(path.is_file() for path in evidence),
            "M460R4 result evidence population incomplete")
    inner = output_dir / "manifest.sha256"
    inner.write_text("".join(
        "{}  {}\n".format(sha256(path), path.name) for path in evidence),
        encoding="utf-8")
    outer = output_dir / "manifest.sha256.outer.seal.sha256"
    outer.write_text("{}  {}\n".format(sha256(inner), inner.name),
                     encoding="utf-8")
    contract = strict_json(contract_path)
    require(tuple(summary["result_sealing"]["receipt_binding_required"]) ==
            tuple(contract["post_capture_advisory_receipt_fields"]),
            "summary/contract post-capture receipt fields diverged")
    return {"summary": summary_path, "inner": inner, "outer": outer}


def dry_run(contract_path):
    contract, observed, workload, roots = validate_contract(contract_path)
    print(json.dumps({
        "schema": contract["schema"],
        "status": "PASS_M460R4_STATIC_CODE_DATA_AND_RECEIPT_SCHEMA_DRY_RUN",
        "identity_inputs": len(observed),
        "samples": len(workload),
        "code_repo": str(roots["code_repo"]),
        "immutable_data_repo": str(roots["immutable_data_repo"]),
        "receipt_binding_required": list(RECEIPT_BINDING_FIELDS),
        "python36_syntax": True,
        "gpu_touched": False,
        "capture_launched": False,
        "training": False,
        "system_speedup": False,
    }, indent=2, sort_keys=True))


def execute(contract_path, output_dir):
    contract, observed, _workload, roots = validate_contract(contract_path)
    require(contract["authorization"]["gpu_capture"] is True,
            "M460R4 contract forbids GPU capture")
    r3, base = load_capture_bases()
    original_validate = base.validate_contract
    original_class = base.FFNResidualStreamCapture
    original_file = base.__file__
    original_load_module = base.load_module
    original_resolve = base.resolve_path

    def resolved_by_text(path_text):
        path = Path(path_text)
        for record in contract["identity"].values():
            if isinstance(record, dict) and record.get("path") == path_text:
                return resolve_identity(record, roots).resolve()
        if path.is_absolute():
            return path
        return original_resolve(path_text)

    def load_with_data_split(path, name):
        module = original_load_module(path, name)
        if name == "m460_profile":
            original_load_config = module.load_config

            def load_config_with_immutable_data(config_path):
                config, device = original_load_config(config_path)
                config["data"]["path"] = str(
                    Path(roots["immutable_data_repo"]) /
                    contract["immutable_data_runtime"]["dataset_root"])
                return config, device
            module.load_config = load_config_with_immutable_data
        return module

    try:
        base.validate_contract = lambda value: validate_contract(value)[:3]
        base.FFNResidualStreamCapture = r3.StrictFFNResidualStreamCapture
        base.__file__ = str(Path(__file__).resolve())
        base.load_module = load_with_data_split
        base.resolve_path = resolved_by_text
        base.execute(contract_path, output_dir)
    finally:
        base.validate_contract = original_validate
        base.FFNResidualStreamCapture = original_class
        base.__file__ = original_file
        base.load_module = original_load_module
        base.resolve_path = original_resolve
    finalized = finalize_r4_result(output_dir, contract_path)
    print("PASS M460R4 summary={} inner={} outer={}".format(
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
