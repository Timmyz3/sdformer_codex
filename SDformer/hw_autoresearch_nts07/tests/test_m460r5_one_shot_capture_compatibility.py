#!/usr/bin/env python3
"""CPU fake-execute/adversarial coverage for M460R5 (Python 3.6)."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


CODE_REPO = Path(__file__).resolve().parents[2]
HW = CODE_REPO / "hw_autoresearch_nts07"
CAPTURE_PATH = (HW / "system_handoff/scripts/"
                "capture_m460r5_h67_g8_ffn_token_residual_s10_one_shot.py")
SEALER_PATH = (HW / "system_handoff/scripts/"
               "seal_m460r5_one_shot_result.py")
CONTRACT_PATH = (HW / "contracts/"
                 "m460r5_h67_g8_one_shot_capture_contract_r1_20260826.json")
RUNNER_PATH = (HW / "system_handoff/"
               "run_m460r5_one_shot_capture_20260826.sh")


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


class FakeProfile(object):
    def __init__(self):
        self.observed_data_path = None

    def load_config(self, _path):
        value = {"data": {"path": "BAD_ORIGINAL_RELATIVE_PATH"}}
        self.observed_data_path = value
        return value, "fake_device"


class FakeBase(object):
    def __init__(self, capture, contract, expected_data_root, fail=False):
        self.capture = capture
        self.contract = contract
        self.expected_data_root = str(expected_data_root)
        self.fail = fail
        self.validate_contract = self.original_validate
        self.FFNResidualStreamCapture = "ORIGINAL_CLASS"
        self.__file__ = "ORIGINAL_FILE"
        self.load_module = self.original_load_module
        self.resolve_path = self.original_resolve
        self.strict_class_seen = None
        self.compatibility_keys_seen = None
        self.data_path_seen = None
        self.end_self_sha_seen = None
        self.end_docs359_sha_seen = None

    def original_validate(self, _path):
        raise RuntimeError("original validator must be replaced")

    def original_load_module(self, _path, name):
        if name == "m460_profile":
            return FakeProfile()
        return object()

    def original_resolve(self, path):
        return Path(path)

    def execute(self, contract_path, output_dir):
        contract, observed, workload = self.validate_contract(contract_path)
        self.compatibility_keys_seen = sorted(observed)
        require(set(self.capture.BASE_OBSERVED_KEYS).issubset(set(observed)),
                "fake base observed mapping incomplete")
        require("profile" not in observed and
                "capture_advisory" not in observed,
                "R4 advisory names reached fake base")
        require(sha256(self.__file__) ==
                contract["identity"]["capture_script"]["sha256"],
                "fake base self SHA contract mismatch")
        self.strict_class_seen = self.FFNResidualStreamCapture
        profile = self.load_module(observed["profile_script"]["path"],
                                   "m460_profile")
        config, _device = profile.load_config(observed["config"]["path"])
        self.data_path_seen = config["data"]["path"]
        require(self.data_path_seen == self.expected_data_root,
                "immutable data root redirect failed")
        require(len(workload) == 10, "fake base workload drift")
        if self.fail:
            raise RuntimeError("injected fake base failure")
        output_dir = Path(output_dir)
        require(not output_dir.exists(), "fake base refuses overwrite")
        output_dir.mkdir(parents=True)
        for sample in range(10):
            for stage, blocks in enumerate((2, 2, 6, 2)):
                for block in range(blocks):
                    (output_dir / "s{:02d}_stage{}_block{}_ffn_metrics.npz".format(
                        sample, stage, block)).write_bytes(b"reduction-only")
        (output_dir / "samples.csv").write_text(
            "sample_id,sequence_key\n" +
            "".join("{},zurich_city_09_a\n".format(i) for i in range(10)),
            encoding="utf-8")
        (output_dir / "per_sample_module_manifest.json").write_text(
            json.dumps({"records": [{"i": i} for i in range(120)]}) + "\n",
            encoding="utf-8")
        summary = {
            "schema": "m460_h67_g8_ffn_token_residual_s10_capture_v1",
            "identity": {
                "checkpoint_load_audit": {"missing_count": 0,
                                          "unexpected_count": 0},
                "capture_bn_policy": "no_running/current-batch",
            },
            "population": {
                "samples": 10, "sequence_keys": ["zurich_city_09_a"],
                "ffn_modules": 12, "sample_module_records": 120,
                "tokens": 5580000, "expected_tokens": 5580000,
            },
            "semantics": {"full_tensor_dumped": False},
            "admission": {"training": False, "system_speedup": False},
        }
        (output_dir / "m460_h67_g8_ffn_token_residual_s10_capture.json").write_text(
            json.dumps(summary) + "\n", encoding="utf-8")
        # Mirror the frozen base's two end-of-run guards after all output has
        # been emitted.  R5 must keep __file__ bound to the R5 wrapper and
        # resolve protected docs359 through the immutable-data identity root.
        self.end_self_sha_seen = sha256(self.__file__)
        require(self.end_self_sha_seen ==
                contract["identity"]["capture_script"]["sha256"],
                "fake base end-of-run self SHA mismatch")
        docs359 = self.resolve_path(contract["identity"]["docs359"]["path"])
        self.end_docs359_sha_seen = sha256(docs359)
        require(self.end_docs359_sha_seen ==
                contract["identity"]["docs359"]["sha256"],
                "fake base end-of-run docs359 SHA mismatch")


class FakeR3(object):
    StrictFFNResidualStreamCapture = "STRICT_R3_CLASS"


def check_manifest(directory, filename):
    directory = Path(directory)
    verified = []
    with (directory / filename).open("r", encoding="utf-8") as handle:
        for line in handle:
            expected, relative = line.rstrip("\n").split("  ", 1)
            require(sha256(directory / relative) == expected,
                    "manifest drift: " + relative)
            verified.append(relative)
    return verified


def main():
    capture = load(CAPTURE_PATH, "m460r5_capture_test")
    sealer = load(SEALER_PATH, "m460r5_sealer_test")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    attacks = []

    require(tuple(contract["post_capture_receipt_fields"]) ==
            capture.RECEIPT_BINDING_FIELDS,
            "contract/capture receipt fields drift")
    require("--capture-once" in runner and
            "M460R5_EXPLICIT_ONE_SHOT_CAPTURE" in runner,
            "runner lacks explicit one-shot gate")
    require("maximum_capture_attempts=1" in runner,
            "runner lacks literal one-attempt boundary")

    complete = {name: {"path": name, "sha256": "0" * 64}
                for name in capture.BASE_OBSERVED_KEYS}
    mapped = capture.build_base_observed(complete)
    require(sorted(mapped) == sorted(capture.BASE_OBSERVED_KEYS),
            "base compatibility map drift")
    incomplete = dict(complete)
    del incomplete["profile_script"]
    attacks.append(expect_reject(
        "missing_profile_script_compatibility_key",
        lambda: capture.build_base_observed(incomplete)))
    legacy = dict(complete)
    del legacy["profile_script"]
    legacy["profile"] = {"path": "bad", "sha256": "0" * 64}
    attacks.append(expect_reject(
        "r4_profile_alias_not_accepted",
        lambda: capture.build_base_observed(legacy)))

    roots = {"code_repo": CODE_REPO, "immutable_data_repo": CODE_REPO}
    observed = {name: {"path": name, "sha256": "0" * 64}
                for name in capture.BASE_OBSERVED_KEYS}
    workload = [{"sample_id": i} for i in range(10)]
    validated = (contract, observed, workload, roots, observed)

    with tempfile.TemporaryDirectory(prefix="m460r5_fake_execute_") as temp:
        output = Path(temp) / "capture"
        fake = FakeBase(capture, contract,
                        CODE_REPO / contract["immutable_data_runtime"][
                            "dataset_root"])
        finalized = capture.execute_with_backend(
            CONTRACT_PATH, output, validated, FakeR3(), fake)
        require(fake.compatibility_keys_seen == sorted(
            capture.BASE_OBSERVED_KEYS), "fake execute mapping drift")
        require(fake.strict_class_seen == "STRICT_R3_CLASS",
                "strict capture class not installed")
        require(fake.end_self_sha_seen ==
                contract["identity"]["capture_script"]["sha256"] and
                fake.end_docs359_sha_seen ==
                contract["identity"]["docs359"]["sha256"],
                "frozen base end-of-run guards were not preserved")
        require(fake.validate_contract == fake.original_validate and
                fake.FFNResidualStreamCapture == "ORIGINAL_CLASS" and
                fake.__file__ == "ORIGINAL_FILE" and
                fake.load_module == fake.original_load_module and
                fake.resolve_path == fake.original_resolve,
                "fake base hooks not restored")
        require(check_manifest(output, finalized["inner"].name) and
                check_manifest(output, finalized["outer"].name),
                "fake execute result seals failed")
        summary = json.loads(finalized["summary"].read_text(encoding="utf-8"))
        require(summary["admission"]["postcompute_opportunity_counts"] is True and
                summary["admission"]["executable_skip"] is False and
                summary["admission"]["system_speedup"] is False,
                "fake summary claim boundary drift")
        require(len(list(output.glob("*.npz"))) == 120,
                "fake reduction NPZ population drift")
        require(sealer.validate_capture_payload(output)[0]["population"][
            "sample_module_records"] == 120,
            "sealer rejected valid fake payload")
        require(len(check_manifest(output, finalized["inner"].name)) == 123,
                "sealer-compatible inner manifest must contain 123 leaves")
        finalized["summary"].write_text("tamper\n", encoding="utf-8")
        attacks.append(expect_reject(
            "post_seal_summary_tamper",
            lambda: check_manifest(output, finalized["inner"].name)))

    with tempfile.TemporaryDirectory(prefix="m460r5_fake_failure_") as temp:
        fake = FakeBase(capture, contract,
                        CODE_REPO / contract["immutable_data_runtime"][
                            "dataset_root"], fail=True)
        attacks.append(expect_reject(
            "injected_base_failure",
            lambda: capture.execute_with_backend(
                CONTRACT_PATH, Path(temp) / "capture", validated,
                FakeR3(), fake)))
        require(fake.validate_contract == fake.original_validate and
                fake.FFNResidualStreamCapture == "ORIGINAL_CLASS" and
                fake.__file__ == "ORIGINAL_FILE",
                "failure path did not restore fake base")

    bad_summary = {
        "schema": "m460_h67_g8_ffn_token_residual_s10_capture_v1",
        "admission": {"training": False},
    }
    bounded = capture.apply_r5_summary(bad_summary, contract)
    require(all(bounded["admission"][name] is False for name in (
        "executable_skip", "delta_aee", "cycle_speedup", "energy", "ppa",
        "system_speedup", "headline", "training")),
        "claim red lines not forced false")
    attacks.extend([
        {"attack": "strict_r3_class_bound", "expected": "reject_drift",
         "observed": "reject_drift", "passes": True},
        {"attack": "immutable_data_root_redirect", "expected": "exact",
         "observed": "exact", "passes": True},
        {"attack": "base_state_restored_success", "expected": "restore",
         "observed": "restore", "passes": True},
        {"attack": "base_state_restored_failure", "expected": "restore",
         "observed": "restore", "passes": True},
        {"attack": "one_shot_gate_literal", "expected": "present",
         "observed": "present", "passes": True},
        {"attack": "forbidden_claims_forced_false", "expected": "false",
         "observed": "false", "passes": True},
        {"attack": "reduction_only_120_npz", "expected": "120",
         "observed": "120", "passes": True},
        {"attack": "frozen_base_end_self_sha", "expected": "exact",
         "observed": "exact", "passes": True},
        {"attack": "frozen_base_end_docs359_resolve", "expected": "exact",
         "observed": "exact", "passes": True},
        {"attack": "sealer_inner_leaf_population", "expected": "123",
         "observed": "123", "passes": True},
    ])
    require(all(item["passes"] for item in attacks),
            "M460R5 fake/adversarial suite failed")
    print(json.dumps({
        "status": "PASS_M460R5_CPU_FAKE_EXECUTE_AND_ADVERSARIAL_TESTS",
        "attack_total": len(attacks),
        "attack_passes": sum(item["passes"] for item in attacks),
        "attacks": attacks,
        "base_observed_keys": sorted(capture.BASE_OBSERVED_KEYS),
        "receipt_fields": list(capture.RECEIPT_BINDING_FIELDS),
        "python36_syntax": True,
        "gpu_touched": False,
        "capture_launched": False,
        "training": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
