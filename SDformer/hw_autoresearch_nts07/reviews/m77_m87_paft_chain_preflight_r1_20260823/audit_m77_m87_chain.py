#!/usr/bin/env python3
"""Independent, CPU-only preflight for the delayed M77/M87 PAFT chain.

This audit deliberately does not import the production builder, materializer,
trainer, or PAFT loader.  It checks their frozen text/YAML contracts and emits
only review evidence.  It never constructs a model or touches CUDA.
"""

import hashlib
import json
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
EXP = ROOT / "neuron_experiments/H9_bipolar_self_attention"

PATHS = {
    "m73_tracer": HW / "system_simulator/scripts/trace_m73_train_calibration_bottleneck_sources.py",
    "m73_queue": HW / "system_handoff/run_m73_train_capture_when_gpu_idle_20260823.sh",
    "m77_builder": HW / "system_simulator/scripts/build_m77_train_only_phi_kmeans_paft_catalog.py",
    "m87_materializer": EXP / "entrypoints/materialize_m87_h67_trainonly_paft_configs.py",
    "m87_chain": HW / "system_handoff/run_m77_m87_paft_after_m73_20260823.sh",
    "pattern_paft": EXP / "overlay/models/STSwinNet_SNN/pattern_paft.py",
    "train_entry": EXP / "entrypoints/train.py",
    "capture_config": EXP / "configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml",
    "training_source_config": EXP / "configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml",
    "m75_r6_receipt": HW / "results/m75_pattern_paft_hard_support_ste_unit_dev_r6_20260823/m75_pattern_paft_hard_support_ste_unit_receipt.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(config: dict, *keys: str):
    value = config
    for key in keys:
        value = value[key]
    return value


def main() -> None:
    require(all(path.is_file() for path in PATHS.values()), "review input missing")
    identity = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for name, path in PATHS.items()
    }

    capture = yaml.safe_load(PATHS["capture_config"].read_text(encoding="utf-8"))
    training = yaml.safe_load(
        PATHS["training_source_config"].read_text(encoding="utf-8"))
    keys = (
        ("bsa_attention", "alpha0"),
        ("bsa_attention", "hardware_quant_enabled"),
        ("bsa_attention", "hardware_rtl_shiftmax_enabled"),
        ("bsa_attention", "hardware_score_step"),
        ("bsa_attention", "hardware_gate_step"),
    )
    forward_differences = []
    for key_path in keys:
        capture_value = capture
        training_value = training
        for key in key_path:
            capture_value = capture_value.get(key) if isinstance(capture_value, dict) else None
            training_value = training_value.get(key) if isinstance(training_value, dict) else None
        if capture_value != training_value:
            forward_differences.append({
                "field": ".".join(key_path),
                "m73_capture": capture_value,
                "m87_training_source": training_value,
            })

    m73_text = PATHS["m73_queue"].read_text(encoding="utf-8")
    chain_text = PATHS["m87_chain"].read_text(encoding="utf-8")
    materializer_text = PATHS["m87_materializer"].read_text(encoding="utf-8")
    builder_text = PATHS["m77_builder"].read_text(encoding="utf-8")
    loader_text = PATHS["pattern_paft"].read_text(encoding="utf-8")
    train_text = PATHS["train_entry"].read_text(encoding="utf-8")

    delayed_launch_pin_tokens = {
        "m77_builder_sha": identity["m77_builder"]["sha256"],
        "m87_materializer_sha": identity["m87_materializer"]["sha256"],
        "pattern_paft_sha": identity["pattern_paft"]["sha256"],
        "train_entry_sha": identity["train_entry"]["sha256"],
        "training_source_config_sha": identity["training_source_config"]["sha256"],
    }
    delayed_launch_pins_present = {
        name: token in chain_text for name, token in delayed_launch_pin_tokens.items()
    }
    materializer_source_sha_pinned = (
        identity["training_source_config"]["sha256"] in materializer_text
        and "sha256(SOURCE)" in materializer_text
    )
    m73_receipt_hash_verified_by_successor = bool(re.search(
        r"sha256sum\s+(?:--check|-c)", chain_text))
    failure_receipt_or_trap = (
        "trap " in chain_text and re.search(r"FAIL|failure|status=", chain_text) is not None
    )
    paired_baseline_present = bool(re.search(
        r"no[-_]?paft|paired[_ -]?baseline", chain_text, flags=re.IGNORECASE))
    second_idle_gate_present = chain_text.count("consecutive_idle") >= 2

    m75_receipt = json.loads(PATHS["m75_r6_receipt"].read_text(encoding="utf-8"))
    positive_mechanism = {
        "status": m75_receipt.get("status"),
        "pattern_paft_sha_matches": (
            m75_receipt["identity"]["pattern_paft_sha256"] ==
            identity["pattern_paft"]["sha256"]),
        "revoked_catalog_rejected": m75_receipt[
            "catalog_pin_attacks"][
                "revoked_sha_even_with_old_config_override"]["rejected"],
        "directed_support_proxy_speedup_fixture": m75_receipt[
            "directed_cost"]["directed_speedup"],
        "formal_training_launch_admitted": m75_receipt[
            "claim_boundary"]["formal_training_launch_admitted"],
    }

    expected_schema_tokens = {
        "builder_emits_m77_schema": (
            "m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1" in builder_text),
        "loader_requires_m77_schema": (
            "m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1" in loader_text),
        "train_installs_loader": "install_pattern_paft" in train_text,
        "m73_uses_hardware_order_capture": "hardware_order_q7q17_deploy.yml" in m73_text,
        "materializer_uses_float_training_source": (
            "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml" in materializer_text),
    }
    require(all(expected_schema_tokens.values()), "static route identity drift")
    require(forward_differences, "expected capture/training semantic difference disappeared")

    findings = [
        {
            "severity": "P0",
            "id": "CAPTURE_TRAIN_FORWARD_MISMATCH",
            "observed": forward_differences,
            "impact": (
                "M73 freezes patterns from the hardware-order/quantized attention "
                "forward while M87 trains the floating H67 forward.  The four PAFT "
                "Conv inputs are downstream of attention; without a bit-exact support "
                "equivalence receipt, the catalog is not admitted for this training run."
            ),
        },
        {
            "severity": "P0",
            "id": "DELAYED_MUTABLE_CODE_WITHOUT_SHA_GATE",
            "observed": delayed_launch_pins_present,
            "impact": (
                "The already-running successor waits before invoking mutable Python/YAML "
                "paths.  It does not fail closed if another session edits the builder, "
                "materializer, loader, trainer, or source config during the wait."
            ),
        },
        {
            "severity": "P1",
            "id": "M73_RECEIPT_CONTENT_NOT_HASH_VERIFIED",
            "observed": {"sha256sum_check_present": m73_receipt_hash_verified_by_successor},
            "impact": "The successor greps PASS but does not verify the receipt's manifest SHA line.",
        },
        {
            "severity": "P1",
            "id": "FAILURE_LEAVES_NO_RECEIPT_AND_NONRESTARTABLE_OUTPUTS",
            "observed": {"failure_trap_or_receipt": failure_receipt_or_trap},
            "impact": (
                "set -e exits without a failure receipt; partially created M77/config/run "
                "paths then trigger the top-level refusal on relaunch."
            ),
        },
        {
            "severity": "P1",
            "id": "NO_PAIRED_NO_PAFT_BASELINE",
            "observed": {"paired_baseline_present": paired_baseline_present},
            "impact": (
                "A five-epoch candidate can be produced, but PAFT-specific benefit cannot "
                "be separated from ordinary continuation or activity collapse."
            ),
        },
        {
            "severity": "P1",
            "id": "GPU_IDLE_NOT_RESERVED",
            "observed": {"idle_gate_before_training": second_idle_gate_present},
            "impact": (
                "The idle probes do not reserve the GPU; the chain has no fail-closed "
                "ownership recheck between smoke and full5."
            ),
        },
        {
            "severity": "P2",
            "id": "MATERIALIZER_SOURCE_CONFIG_NOT_PINNED",
            "observed": {"source_sha_pinned": materializer_source_sha_pinned},
            "impact": "Standalone materialization can silently inherit source-config drift.",
        },
    ]

    payload = {
        "schema": "m77_m87_paft_chain_independent_preflight_v1",
        "status": "NO_GO_M77_M87_AUTOMATIC_TRAINING_TWO_P0",
        "scope": "CPU_ONLY_STATIC_AND_FROZEN_RECEIPT_REPLAY_NO_GPU_NO_TRAINING",
        "identity": identity,
        "static_route_checks": expected_schema_tokens,
        "mechanism_evidence_retained": positive_mechanism,
        "findings": findings,
        "verdict": {
            "automatic_successor_launch": "NO_GO",
            "m73_current_capture_configuration": "NO_GO_FOR_M87_FLOAT_TRAINING_CATALOG",
            "m77_builder_arithmetic_after_matching_trace": "CONDITIONAL_GO",
            "m87_candidate_training_after_patches": "CONDITIONAL_GO",
            "paft_accuracy_or_speedup_claim": "NO_GO_UNTIL_PAIRED_VALID825_AND_CYCLE_REPLAY",
        },
        "minimum_relaunch_gates": [
            "Capture M73 with the exact PAFT training forward config, or provide a bit-exact four-operator support-equivalence receipt.",
            "Pin and verify SHA256 for M73 tracer/manifest receipt, M77 builder, M87 materializer, source YAML, pattern_paft.py, train.py, and checkpoint before delayed execution.",
            "Write an atomic failure receipt and a safe resume policy for partial outputs.",
            "Run one-step real positive smoke, then paired PAFT/no-PAFT five-epoch runs from the same checkpoint, seed, sample order, and optimizer policy.",
            "Keep valid825 out of catalog/training; use it only after checkpoints exist for accuracy/activity guardrails.",
        ],
    }
    output = Path(__file__).resolve().parent / "m77_m87_paft_chain_preflight.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS independent preflight: verdict={} findings={}".format(
        payload["verdict"]["automatic_successor_launch"], len(findings)))


if __name__ == "__main__":
    main()
