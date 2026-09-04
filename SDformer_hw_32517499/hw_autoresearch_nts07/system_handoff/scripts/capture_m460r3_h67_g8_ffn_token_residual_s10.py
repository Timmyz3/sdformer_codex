#!/usr/bin/env python3
"""Strict-order, double-sealed M460R3 H67 FFN residual capture.

The numerical reducer and frozen H67 loader are inherited from the locally
verified M460 implementation.  R3 adds a strict per-FFN hook state machine,
R3 contract validation and a detached outer seal over the result manifest.
This file intentionally uses Python-3.6-compatible syntax.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
BASE_PATH = (HW / "system_handoff/scripts/"
             "capture_m460_h67_g8_ffn_token_residual_s10.py")


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


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_PATH, "m460r3_frozen_m460_base")


def resolve_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path_text.startswith(("neuron_experiments/", "third_party/")):
        return ROOT / path
    return HW / path


def validate_contract(contract_path):
    contract_path = Path(contract_path).resolve()
    contract = strict_json(contract_path)
    require(contract.get("schema") ==
            "m460r3_h67_g8_ffn_token_residual_s10_capture_contract_v1",
            "M460R3 contract schema drift")
    require(contract.get("status") ==
            "READY_REVIEW__REMOTE_LAUNCH_FORBIDDEN_UNTIL_OUTER_SEAL_APPROVED",
            "M460R3 contract is not review-frozen")
    observed = {}
    for name, record in contract["identity"].items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        path = resolve_path(record["path"]).resolve()
        require(path.is_file(), "M460R3 missing identity {}: {}".format(
            name, path))
        actual = sha256(path)
        require(actual == record["sha256"],
                "M460R3 SHA drift {} expected={} observed={}".format(
                    name, record["sha256"], actual))
        observed[name] = {"path": str(path), "sha256": actual}
    require(sha256(Path(__file__).resolve()) ==
            contract["identity"]["capture_script"]["sha256"],
            "M460R3 capture self SHA drift")
    require(Path(observed["m460_base_capture"]["path"]).resolve() ==
            BASE_PATH.resolve(), "M460R3 base capture path drift")

    source = Path(observed["swin_source"]["path"]).read_text(
        encoding="utf-8")
    for fragment in (
            "class MS_Spiking_Mlp(Spiking_Mlp):",
            "x = self.sn1(x)", "x = self.fc1(x)", "x= self.bn1(",
            "x = self.sn2(x)", "x = self.fc2(x)", "x = self.bn2(",
            "self.mlp(x.permute(1,0,2,3,4)).permute(1,0,2,3,4)"):
        require(fragment in source,
                "M460R3 FFN topology source drift: " + fragment)
    workload = BASE.read_workload(
        Path(observed["sample_workload"]["path"]))
    return contract, observed, workload


class StrictFFNResidualStreamCapture(BASE.FFNResidualStreamCapture):
    """Require pre -> sn1 -> sn2 -> fc2 -> full-output exactly once."""

    def _make_pre_hook(self, stage, block, name):
        parent = super(StrictFFNResidualStreamCapture, self)._make_pre_hook(
            stage, block, name)

        def hook(module, inputs):
            parent(module, inputs)
            require(name in self.state and "phase" not in self.state[name],
                    "M460R3 pre-hook state initialization drift: " + name)
            self.state[name]["phase"] = "EXPECT_SN1"
        return hook

    def _make_source_hook(self, name, role):
        parent = super(StrictFFNResidualStreamCapture, self)._make_source_hook(
            name, role)
        expected = "EXPECT_SN1" if role == "sn1" else "EXPECT_SN2"
        next_phase = "EXPECT_SN2" if role == "sn1" else "EXPECT_FC2"

        def hook(module, inputs, output):
            require(name in self.state and
                    self.state[name].get("phase") == expected,
                    "M460R3 strict hook order rejected {} at {} expected {}".format(
                        role, name,
                        self.state.get(name, {}).get("phase", "NO_PRE")))
            parent(module, inputs, output)
            self.state[name]["phase"] = next_phase
        return hook

    def _make_fc2_hook(self, name):
        parent = super(StrictFFNResidualStreamCapture, self)._make_fc2_hook(name)

        def hook(module, inputs, output):
            require(name in self.state and
                    self.state[name].get("phase") == "EXPECT_FC2",
                    "M460R3 strict hook order rejected fc2 at {} expected {}".format(
                        name, self.state.get(name, {}).get("phase", "NO_PRE")))
            parent(module, inputs, output)
            self.state[name]["phase"] = "EXPECT_OUTPUT"
        return hook

    def _make_output_hook(self, stage, block, name):
        parent = super(StrictFFNResidualStreamCapture, self)._make_output_hook(
            stage, block, name)

        def hook(module, inputs, output):
            require(name in self.state and
                    self.state[name].get("phase") == "EXPECT_OUTPUT",
                    "M460R3 strict hook order rejected full output at {} expected {}".format(
                        name, self.state.get(name, {}).get("phase", "NO_PRE")))
            parent(module, inputs, output)
        return hook


def dry_run(contract_path):
    contract, observed, workload = validate_contract(contract_path)
    payload = {
        "schema": contract["schema"],
        "status": "PASS_M460R3_STATIC_EXACT_SHA_AND_STRICT_ORDER_DRY_RUN",
        "identity_inputs": len(observed),
        "samples": len(workload),
        "ffn_modules": len(BASE.all_targets()),
        "hooks": 5 * len(BASE.all_targets()),
        "hook_order": ["pre", "sn1", "sn2", "fc2", "full_output"],
        "literal_npz_members": list(contract["streaming_capture"][
            "literal_npz_members"]),
        "result_outer_seal": True,
        "python36_syntax": True,
        "gpu_touched": False,
        "remote_contacted": False,
        "training": False,
        "system_speedup": False,
        "headline": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def execute(contract_path, output_dir):
    # The frozen numerical capture refers to these names in its own module
    # globals.  Rebind only the contract validator, strict state-machine class
    # and self path; no arithmetic, threshold or loader behavior is changed.
    BASE.validate_contract = validate_contract
    BASE.FFNResidualStreamCapture = StrictFFNResidualStreamCapture
    BASE.__file__ = str(Path(__file__).resolve())
    BASE.execute(contract_path, output_dir)

    output_dir = Path(output_dir).resolve()
    summary_path = output_dir / "m460_h67_g8_ffn_token_residual_s10_capture.json"
    require(summary_path.is_file(), "M460R3 base summary absent")
    summary = strict_json(summary_path)
    require(summary.get("schema") ==
            "m460_h67_g8_ffn_token_residual_s10_capture_v1",
            "M460R3 base summary schema drift")
    summary["schema"] = "m460r3_h67_g8_ffn_token_residual_s10_capture_v1"
    summary["status"] = (
        "PASS_M460R3_H67_EP35_NO_RUNNING_S10_STRICT_ORDER_DOUBLE_SEAL")
    summary["strict_runtime_state_machine"] = {
        "order": ["pre", "sn1", "sn2", "fc2", "full_output"],
        "per_module_per_sample": "exactly once",
        "sn2_fc2_sn1_attack_accepted": False,
    }
    summary["result_sealing"] = {
        "inner_manifest": "manifest.sha256",
        "outer_seal": "manifest.sha256.outer.seal.sha256",
        "receipt_binding_required": [
            "summary_sha256", "inner_manifest_sha256",
            "outer_seal_file_sha256"],
    }
    summary["admission"]["strict_hook_order"] = True
    summary["admission"]["double_sealed_payload"] = True
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

    evidence = sorted(output_dir.glob("*.npz")) + [
        output_dir / "samples.csv",
        output_dir / "per_sample_module_manifest.json",
        summary_path,
    ]
    require(all(path.is_file() for path in evidence),
            "M460R3 inner manifest evidence population incomplete")
    inner = output_dir / "manifest.sha256"
    inner.write_text("".join(
        "{}  {}\n".format(sha256(path), path.name) for path in evidence),
        encoding="utf-8")
    outer = output_dir / "manifest.sha256.outer.seal.sha256"
    outer.write_text("{}  {}\n".format(sha256(inner), inner.name),
                     encoding="utf-8")
    require(sha256(Path(__file__).resolve()) ==
            strict_json(contract_path)["identity"]["capture_script"]["sha256"],
            "M460R3 capture changed during execution")
    print("PASS M460R3 {} inner={} outer_seal_file={}".format(
        summary_path, sha256(inner), sha256(outer)), flush=True)


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
