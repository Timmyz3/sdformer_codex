#!/usr/bin/env python3
"""CPU-only M460R4 P1 closure and tamper tests (Python 3.6 syntax)."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


CODE_REPO = Path(__file__).resolve().parents[2]
HW = CODE_REPO / "hw_autoresearch_nts07"
INVENTORY_PATH = (HW / "system_handoff/scripts/"
                  "build_m460r4_package_inventory.py")
PREFLIGHT_PATH = (HW / "system_handoff/scripts/"
                  "preflight_m460r4_code_data_environment.py")
CAPTURE_PATH = (HW / "system_handoff/scripts/"
                "capture_m460r4_h67_g8_ffn_token_residual_s10.py")
CONTRACT_PATH = (HW / "contracts/"
                 "m460r4_h67_g8_environment_preflight_contract_r1_20260826.json")
RUNNER_PATH = (HW / "system_handoff/"
               "run_m460r4_sealed_preflight_no_capture_20260826.sh")
FIELDS = (
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


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_reject(name, function):
    try:
        function()
    except Exception:
        return {"attack": name, "expected": "reject", "observed": "reject",
                "passes": True}
    return {"attack": name, "expected": "reject", "observed": "accept",
            "passes": False}


def check_manifest(directory, name):
    path = Path(directory) / name
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    for line in lines:
        expected, filename = line.split("  ", 1)
        require(sha256(Path(directory) / filename) == expected,
                "manifest mismatch " + filename)
    return True


def synthetic_inventory_pair():
    freeze = {
        "python": {"x": 1},
        "packages": [{"module": "torch", "distribution_version": "2.2.2"}],
        "build": {"torch_cuda": "12.1"},
        "runtime_imports": [],
        "isolation": {"forbidden_sys_path_substrings": ["/forbidden/"]},
    }
    inventory = {
        "schema": "m460r4_live_package_build_inventory_v1",
        "code_repo": "/clean/code",
        "python": {"x": 1},
        "packages": [{"module": "torch", "distribution_version": "2.2.2"}],
        "build": {"torch_cuda": "12.1"},
        "runtime_imports": [],
        "isolation": {"python_isolated": True, "PYTHONNOUSERSITE": "1",
                      "PYTHONPATH": None},
        "final_sys_path": ["/site-packages"],
        "cuda_initialized": False,
    }
    return freeze, inventory


def main():
    inventory_module = load(INVENTORY_PATH, "m460r4_inventory_test")
    preflight_module = load(PREFLIGHT_PATH, "m460r4_preflight_test")
    capture_module = load(CAPTURE_PATH, "m460r4_capture_test")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")

    require(tuple(contract["post_capture_advisory_receipt_fields"]) == FIELDS,
            "contract receipt fields drift")
    require(tuple(capture_module.RECEIPT_BINDING_FIELDS) == FIELDS,
            "capture receipt fields drift")
    require(all(field in runner_source for field in FIELDS),
            "runner advisory receipt fields drift")
    require("--capture)" not in runner_source and
            "--capture|" not in runner_source,
            "R4 runner exposes a capture mode")

    attacks = []
    freeze, inventory = synthetic_inventory_pair()
    require(inventory_module.validate_inventory(inventory, freeze),
            "synthetic valid inventory rejected")

    tampered = json.loads(json.dumps(inventory))
    tampered["packages"][0]["distribution_version"] = "9.9.9"
    attacks.append(expect_reject(
        "package_version_tamper",
        lambda: inventory_module.validate_inventory(tampered, freeze)))
    tampered = json.loads(json.dumps(inventory))
    tampered["cuda_initialized"] = True
    attacks.append(expect_reject(
        "cuda_initialized_tamper",
        lambda: inventory_module.validate_inventory(tampered, freeze)))
    tampered = json.loads(json.dumps(inventory))
    tampered["final_sys_path"].append("/forbidden/original")
    attacks.append(expect_reject(
        "forbidden_sys_path_tamper",
        lambda: inventory_module.validate_inventory(tampered, freeze)))
    tampered = json.loads(json.dumps(inventory))
    tampered["isolation"]["PYTHONPATH"] = "/tmp/shadow"
    attacks.append(expect_reject(
        "pythonpath_tamper",
        lambda: inventory_module.validate_inventory(tampered, freeze)))

    real_freeze = json.loads((HW / "system_handoff/m460r4_launch_bundle_20260826/"
                              "m460r4_remote_environment_freeze.json").read_text(
                                  encoding="utf-8"))
    rejected = preflight_module.shadow_candidates([
        "SDformer/torch.py",
        "SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/"
        "Spiking_STSwinNet.py",
        "SDformer/neuron_experiments/H9_bipolar_self_attention/overlay/"
        "models/STSwinNet_SNN/bsa_attention.py",
        "SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/__init__.py",
    ], real_freeze)
    require(len(rejected) == 4, "critical shadow attacks not all rejected")
    require(preflight_module.shadow_candidates([
        "SDformer/neuron_experiments/H9_bipolar_self_attention/overlay/"
        "models/STSwinNet_SNN/near_match_residual_elision.py",
        "SDformer/hw_autoresearch_nts07/contracts/new_contract.json",
    ], real_freeze) == [], "non-imported R4 files falsely rejected")
    for name in ("top_level_torch_shadow", "tracked_model_shadow",
                 "overlay_bsa_shadow", "namespace_init_shadow"):
        attacks.append({"attack": name, "expected": "reject",
                        "observed": "reject", "passes": True})

    with tempfile.TemporaryDirectory(prefix="m460r4_receipt_test_") as temp:
        root = Path(temp)
        summary_path = root / "m460_h67_g8_ffn_token_residual_s10_capture.json"
        summary_path.write_text(json.dumps({
            "schema": "m460_h67_g8_ffn_token_residual_s10_capture_v1",
            "admission": {"system_speedup": False, "training": False},
        }) + "\n", encoding="utf-8")
        (root / "samples.csv").write_text("sample_id\n0\n", encoding="utf-8")
        (root / "per_sample_module_manifest.json").write_text(
            "{}\n", encoding="utf-8")
        (root / "s00_stage0_block0_ffn_metrics.npz").write_bytes(b"fixture")
        sealed = capture_module.finalize_r4_result(root, CONTRACT_PATH)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(tuple(summary["result_sealing"]["receipt_binding_required"]) ==
                FIELDS, "summary advisory receipt fields drift")
        require(check_manifest(root, "manifest.sha256"),
                "inner manifest rejected")
        require(check_manifest(root, "manifest.sha256.outer.seal.sha256"),
                "outer seal rejected")
        summary_path.write_text("tamper\n", encoding="utf-8")
        attacks.append(expect_reject(
            "post_seal_summary_tamper",
            lambda: check_manifest(root, sealed["inner"].name)))

    require(all(row["passes"] for row in attacks),
            "M460R4 tamper suite failed")
    result = {
        "status": "PASS_M460R4_CPU_P1_CLOSURE_AND_TAMPER_TESTS",
        "attack_total": len(attacks),
        "attack_passes": sum(row["passes"] for row in attacks),
        "attacks": attacks,
        "post_capture_receipt_fields": list(FIELDS),
        "runner_capture_mode_exposed": False,
        "python36_syntax": True,
        "gpu_touched": False,
        "remote_contacted": False,
        "capture_launched": False,
        "training": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
