#!/usr/bin/env python3
"""Independent CPU/static hammer review for the repaired M73/M77/M87 chain.

The review does not import the production PAFT/training stack and never touches
CUDA.  It checks exact identities and shell control flow, invokes only the
CPU-only config materializer on a synthetic admitted fixture, and independently
compares the three generated YAML arms.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import yaml


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
EXP = ROOT / "neuron_experiments/H9_bipolar_self_attention"

PATHS = {
    "m73_queue": HW / "system_handoff/run_m73_train_capture_when_gpu_idle_20260823.sh",
    "m87_chain": HW / "system_handoff/run_m77_m87_paft_after_m73_20260823.sh",
    "m73_tracer": HW / "system_simulator/scripts/trace_m73_train_calibration_bottleneck_sources.py",
    "m40_writer": HW / "system_simulator/scripts/trace_m40_bottleneck_packed_sources.py",
    "profile": EXP / "entrypoints/profile_nts11_hardware_p0.py",
    "m77_builder": HW / "system_simulator/scripts/build_m77_train_only_phi_kmeans_paft_catalog.py",
    "materializer": EXP / "entrypoints/materialize_m87_h67_trainonly_paft_configs.py",
    "pattern_paft": EXP / "overlay/models/STSwinNet_SNN/pattern_paft.py",
    "train_entry": EXP / "entrypoints/train.py",
    "forward_config": EXP / "configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml",
}

EXPECTED = {
    "m73_tracer": "9d79f7198ba1ac221f6e58428480c9d59e3deafff0775d2ae3aaa0da75f693bb",
    "m40_writer": "b02ac10fb95e68fa2871b74330d6f39d7d3d8cbfa6440990d43ec832e943bf19",
    "profile": "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684",
    "m77_builder": "c760e21eac16c4e7d5112b1335c0b121762f47175f48b92e9393391b1b33e6c6",
    "materializer": "d6f80180de911edf0a13a55f2ca2a96b474956d15c6441fd50168cf5eb71375f",
    "pattern_paft": "47e6d80fa5fd50604f0d9adce1fb7ac34a741da492ac19f2ef945cfba46c7bd2",
    "train_entry": "fccd1d05bbf73aac0061e604a9d199cf9c3fd4ba8e9cea175231a3ebc14e44ac",
    "forward_config": "86db3960c7d12ce5c7365e82e24b1f3aef6961b79d12317da32fc41b15d1cbcc",
    "checkpoint": "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
    "train_list": "919c79c61535eb499364ffe28fad3000441e25d1bddbf4fa9a0c27a78d4fdc10",
    "valid_list": "7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0",
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


def json_sha(payload):
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), raw


def flatten_diff(left, right, prefix=""):
    differences = []
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            child = "{}.{}".format(prefix, key) if prefix else str(key)
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(flatten_diff(left[key], right[key], child))
    elif isinstance(left, list):
        if left != right:
            differences.append(prefix)
    elif left != right:
        differences.append(prefix)
    return differences


def synthetic_materializer_test():
    operators = [
        "sttmultires_unet.resblocks.0.conv1.0",
        "sttmultires_unet.resblocks.0.conv2.0",
        "sttmultires_unet.resblocks.1.conv1.0",
        "sttmultires_unet.resblocks.1.conv2.0",
    ]
    trace = {
        "schema": "m73_h67_ep35_train_calibration_packed_source_trace_v1",
        "identity": {
            "config_sha256": EXPECTED["forward_config"],
            "paft_forward_base_config_sha256": EXPECTED["forward_config"],
        },
        "split_audit": {
            "full_train_valid825_key_overlap": 0,
            "selected_valid825_key_overlap": 0,
        },
    }
    trace_sha, trace_raw = json_sha(trace)
    catalog = {
        "schema": "m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1",
        "identity": {"forward_base_config_sha256": EXPECTED["forward_config"]},
        "split": {
            "role": "DSEC_TRAIN_ONLY_PAFT_CALIBRATION",
            "train_catalog_eligible": True,
            "test_or_validation_data_used": False,
        },
        "operators": [{"operator": name, "partitions": []} for name in operators],
    }
    catalog_sha, catalog_raw = json_sha(catalog)
    contract = {
        "schema": "m77_pattern_paft_catalog_admission_contract_v1",
        "unit_test_only": False,
        "train_only_admitted": True,
        "catalog_sha256": catalog_sha,
        "train_trace_manifest_sha256": trace_sha,
        "forward_base_config_sha256": EXPECTED["forward_config"],
    }
    _, contract_raw = json_sha(contract)

    with tempfile.TemporaryDirectory(prefix="materializer_", dir=str(REVIEW)) as raw_temp:
        temp = Path(raw_temp)
        trace_path = temp / "trace.json"
        catalog_path = temp / "catalog.json"
        contract_path = temp / "contract.json"
        trace_path.write_text(trace_raw, encoding="utf-8")
        catalog_path.write_text(catalog_raw, encoding="utf-8")
        contract_path.write_text(contract_raw, encoding="utf-8")
        outputs = {
            "full": temp / "paft_full5.yml",
            "smoke": temp / "paft_smoke1.yml",
            "control": temp / "no_paft_control_full5.yml",
        }
        command = [
            "python3", str(PATHS["materializer"]),
            "--catalog", str(catalog_path),
            "--admission-contract", str(contract_path),
            "--train-trace-manifest", str(trace_path),
            "--full-output", str(outputs["full"]),
            "--smoke-output", str(outputs["smoke"]),
            "--control-output", str(outputs["control"]),
        ]
        completed = subprocess.run(
            command, cwd=str(ROOT), universal_newlines=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        require(completed.returncode == 0, "materializer synthetic positive path failed")
        configs = {
            name: yaml.safe_load(path.read_text(encoding="utf-8"))
            for name, path in outputs.items()
        }
        full = configs["full"]
        control = configs["control"]
        smoke = configs["smoke"]
        paired_diff = flatten_diff(full, control)
        smoke_diff = flatten_diff(full, smoke)
        require(set(item for item in paired_diff if not item.startswith(
            "pattern_paft.")) == {
                "experiment", "note", "runtime.paired_arm"}
            and any(item.startswith("pattern_paft.") for item in paired_diff),
            "paired PAFT/control config differs outside declared arm fields: " +
            repr(paired_diff))
        require(control["pattern_paft"]["enabled"] is False,
                "control PAFT is not disabled")
        require(full["pattern_paft"]["enabled"] is True,
                "candidate PAFT is not enabled")
        require(full["loader"]["n_epochs"] == control["loader"]["n_epochs"] == 5,
                "paired epoch budget drift")
        require(full["runtime"]["seed"] == control["runtime"]["seed"],
                "paired seed drift")
        require(full["optimizer"] == control["optimizer"],
                "paired optimizer policy drift")
        require(full["pattern_paft"]["expected_forward_base_config_sha256"] ==
                EXPECTED["forward_config"], "materialized forward identity drift")
        require(full["pattern_paft"]["expected_checkpoint_sha256"] ==
                EXPECTED["checkpoint"], "materialized checkpoint identity drift")
        require(set(smoke_diff) == {
            "experiment", "loader.n_epochs", "note", "runtime.force_save_epochs",
            "runtime.max_train_steps", "runtime.skip_save",
            "runtime.skip_state_save", "runtime.state_save_epochs"},
            "smoke config differs outside declared smoke fields")
        require(smoke["loader"]["n_epochs"] == 1 and
                smoke["runtime"]["max_train_steps"] == 1,
                "smoke budget is not one step")
        return {
            "status": "PASS_CPU_SYNTHETIC_MATERIALIZATION",
            "stdout": completed.stdout.strip(),
            "paired_diff_paths": paired_diff,
            "smoke_diff_paths": smoke_diff,
            "paired_epoch_budget": full["loader"]["n_epochs"],
            "paired_seed": full["runtime"]["seed"],
            "paired_optimizer_equal": full["optimizer"] == control["optimizer"],
            "control_paft_enabled": control["pattern_paft"]["enabled"],
        }


def main():
    require(all(path.is_file() for path in PATHS.values()), "review input missing")
    identity = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in PATHS.items()
    }
    for name, expected in EXPECTED.items():
        if name in identity:
            require(identity[name]["sha256"] == expected,
                    "reviewed input SHA drift: " + name)

    m73 = PATHS["m73_queue"].read_text(encoding="utf-8")
    chain = PATHS["m87_chain"].read_text(encoding="utf-8")
    tracer = PATHS["m73_tracer"].read_text(encoding="utf-8")
    builder = PATHS["m77_builder"].read_text(encoding="utf-8")
    materializer = PATHS["materializer"].read_text(encoding="utf-8")
    loader = PATHS["pattern_paft"].read_text(encoding="utf-8")

    same_forward = {
        "m73_queue_uses_float_source": (
            "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml" in m73 and
            "hardware_order_q7q17_deploy.yml" not in m73),
        "m73_tracer_expected_sha": EXPECTED["forward_config"] in tracer,
        "m77_builder_expected_sha": EXPECTED["forward_config"] in builder,
        "materializer_expected_sha": EXPECTED["forward_config"] in materializer,
        "runtime_loader_expected_sha": EXPECTED["forward_config"] in loader,
        "successor_receipt_gate": (
            "forward_base_config_sha256=" + EXPECTED["forward_config"] in chain),
    }
    require(all(same_forward.values()), "original forward-config P0 not closed")

    m73_pins = {
        name: expected in m73 for name, expected in EXPECTED.items()
        if name in {"m73_tracer", "m40_writer", "profile", "forward_config",
                    "checkpoint", "train_list", "valid_list"}
    }
    chain_pins = {
        name: expected in chain for name, expected in EXPECTED.items()
        if name in {"m73_tracer", "m77_builder", "materializer", "pattern_paft",
                    "train_entry", "forward_config", "checkpoint"}
    }
    wait_gates = {
        "m73_all_direct_pins_present": all(m73_pins.values()),
        "m73_pre_and_post_wait_gate": m73.count("check_all_pins") >= 3,
        "m87_all_direct_pins_present": all(chain_pins.values()),
        "m87_pre_and_post_m73_wait_gate": chain.count("check_static_pins") >= 5,
        "m87_manifest_hash_from_m73_receipt_checked": (
            'm73_expected="$(awk' in chain and
            'check_sha "$m73_manifest" "$m73_expected"' in chain),
        "runtime_loader_cross_binds_trace_config_checkpoint": all(token in loader for token in (
            "M71 PAFT runtime train-trace SHA mismatch",
            "M71 PAFT capture/training forward-config mismatch",
            "M71 PAFT runtime checkpoint SHA mismatch")),
    }
    require(all(wait_gates.values()), "original delayed mutable-input P0 not closed")

    publication = {
        "m73_unique_partial": "output_stage=" in m73,
        "m73_atomic_directory_publish": 'mv "$output_stage" "$output"' in m73,
        "m73_atomic_receipt_publish": 'mv "$receipt_tmp" "$receipt"' in m73,
        "m73_failure_trap": "trap on_exit EXIT" in m73 and
                            "FAILED_M73_DO_NOT_USE" in m73,
        "m77_unique_partial": "m77_stage=" in chain,
        "m77_atomic_directory_publish": 'mv "$m77_stage" "$m77_dir"' in chain,
        "config_unique_partial": "config_stage=" in chain,
        "config_atomic_directory_publish": 'mv "$config_stage" "$config_bundle"' in chain,
        "arm_unique_partial": 'local partial="${final_dir}.partial.' in chain,
        "arm_atomic_directory_publish": 'mv "$partial" "$final_dir"' in chain,
        "chain_atomic_receipt_publish": 'mv "$receipt_tmp" "$chain_receipt"' in chain,
        "chain_failure_trap": "trap on_exit EXIT" in chain and
                              "FAILED_M87_CHAIN_DO_NOT_CITE" in chain,
        "restart_existing_arm_checks": 'if [[ -d "$final_dir" ]]' in chain,
        "restart_complete_chain_check": 'if [[ -f "$chain_receipt" ]]' in chain,
    }
    require(all(publication.values()), "atomic/failure/restart mechanism missing")

    paired = {
        "control_config_route": "no_paft_control_full5.yml" in chain,
        "control_runs_before_candidate": (
            chain.index("run_arm no_paft_control_full5") <
            chain.index("run_arm paft_full5")),
        "same_checkpoint_argument": chain.count('--prev_runid "$checkpoint"') == 1,
        "both_epoch4_checkpoints_required": (
            "control_checkpoint_epoch4_sha256=" in chain and
            "paft_checkpoint_epoch4_sha256=" in chain),
        "claim_flags_false": all(token in chain for token in (
            "valid825_accuracy=false", "cycle_speedup=false",
            "system_speedup=false", "headline=false")),
    }
    require(all(paired.values()), "paired-control or claim-boundary route missing")

    idle = {
        "four_probe_gate_per_arm": "while (( idle < 4 ))" in chain,
        "gate_before_each_arm": "wait_for_idle" in chain,
        "never_kills_or_preempts": not re.search(
            r"\b(kill|pkill|killall|fuser\s+-k|nvidia-smi\s+--gpu-reset)\b", chain),
        "nvidia_smi_fail_closed": "nvidia-smi --query-compute-apps" in chain and
                                  "|| true" not in chain,
        "exclusive_chain_lock": bool(re.search(r"\bflock\b|mkdir\s+[^\n]*lock", chain)),
    }

    materialization = synthetic_materializer_test()

    findings = [
        {
            "severity": "P1",
            "id": "GPU_IDLE_PROBE_FAILS_OPEN_AND_HAS_TOCTOU",
            "evidence": {
                "nvidia_smi_fail_closed": idle["nvidia_smi_fail_closed"],
                "four_probe_gate_per_arm": idle["four_probe_gate_per_arm"],
            },
            "impact": (
                "nvidia-smi failure is converted to an empty process list by `|| true`; "
                "four such failures launch training.  A real user process can also start "
                "after the last probe because no GPU reservation is acquired."
            ),
        },
        {
            "severity": "P1",
            "id": "NO_SINGLETON_LOCK_FOR_CHAIN_OR_ARMS",
            "evidence": {"exclusive_chain_lock": idle["exclusive_chain_lock"]},
            "impact": (
                "Two successor instances can both pass existence checks, run duplicate "
                "GPU arms, and race directory mv/receipt publication."
            ),
        },
        {
            "severity": "P1",
            "id": "EXISTING_FINAL_RECEIPT_IS_ONLY_STATUS_CHECKED",
            "evidence": {
                "restart_status_only": (
                    "grep -qx 'status=PASS_M87_H67_TRAINONLY_PAFT_PAIRED_FULL5'" in chain),
                "restart_rehashes_checkpoints": False,
            },
            "impact": (
                "On restart, a pre-existing PASS receipt returns before re-hashing the "
                "manifest, catalog, contract, configs, logs, or paired checkpoints."
            ),
        },
        {
            "severity": "P1",
            "id": "POST_CAPTURE_SOURCE_GATE_AND_TRANSITIVE_SOURCE_MANIFEST_ABSENT",
            "evidence": {
                "m73_post_capture_check_all_pins": (
                    m73.rfind("check_all_pins") > m73.find("M73_CAPTURE_START_UTC")),
                "transitive_source_manifest": False,
            },
            "impact": (
                "The repaired delayed-wait gate covers every direct source named by the "
                "prior admission checklist, but M73 does not repeat it after capture and "
                "neither launcher freezes the dynamically imported baseline/overlay tree."
            ),
        },
        {
            "severity": "P2",
            "id": "M73_CRASH_AFTER_DIRECTORY_MV_NEEDS_MANUAL_RECOVERY",
            "evidence": {
                "final_directory_refused_on_restart": '[[ -e "$output" || -e "$receipt"' in m73,
                "directory_published_before_receipt": (
                    m73.index('mv "$output_stage" "$output"') <
                    m73.index('mv "$receipt_tmp" "$receipt"')),
            },
            "impact": (
                "A crash in the narrow directory-mv/receipt-mv window leaves a complete "
                "directory that the queue refuses on restart; recovery is safe but manual."
            ),
        },
        {
            "severity": "P2",
            "id": "CONTROL_RESUME_CHECK_IS_WEAKER_THAN_PAFT_ARMS",
            "evidence": {
                "paft_log_markers_required": True,
                "control_has_arm_specific_completion_marker": False,
            },
            "impact": (
                "An existing control arm is admitted from only non-empty train.log plus "
                "checkpoint_epoch4.pth; it has no atomic arm receipt or negative assertion "
                "that PAFT hooks were absent."
            ),
        },
    ]

    payload = {
        "schema": "m73_m77_m87_paired_chain_r2_independent_hammer_v1",
        "status": "SCOPED_DEPLOYMENT_GO_ORIGINAL_P0_CLOSED_PERFORMANCE_NO_GO",
        "scope": "CPU_ONLY_STATIC_AND_SYNTHETIC_CONFIG_TEST_NO_GPU_NO_TRAINING",
        "identity": identity,
        "original_p0_closure": {
            "same_86db_forward_config": same_forward,
            "delayed_direct_source_config_checkpoint_sha_gate": wait_gates,
            "closed_p0_count": 2,
            "open_p0_count": 0,
        },
        "atomic_failure_restart_checks": publication,
        "paired_route_checks": paired,
        "gpu_idle_checks": idle,
        "cpu_materializer_test": materialization,
        "findings": findings,
        "severity_counts": {
            "P0": sum(item["severity"] == "P0" for item in findings),
            "P1": sum(item["severity"] == "P1" for item in findings),
            "P2": sum(item["severity"] == "P2" for item in findings),
        },
        "verdict": {
            "existing_single_known_successor_with_working_nvidia_smi": "SCOPED_GO",
            "generic_duplicate_or_unattended_relaunch": "NO_GO_UNTIL_LOCK_AND_STRONG_RESTART_REVALIDATION",
            "m73_train_only_capture": "GO_AFTER_RUNTIME_GPU_IDLE",
            "m77_catalog_build": "GO_AFTER_REAL_M73_ADMISSION",
            "paired_smoke_control_paft_training": "SCOPED_GO_WITH_SINGLETON_OPERATOR_GUARD",
            "valid825_accuracy_claim": "NO_GO",
            "cycle_or_system_speedup_claim": "NO_GO",
            "date_headline_or_performance_claim": "NO_GO",
        },
        "required_operator_guardrails": [
            "Confirm exactly one M87 successor PID before GPU becomes idle.",
            "Treat any nvidia-smi error as busy/unknown and do not launch an arm.",
            "Do not relaunch from an existing PASS receipt without re-hashing all receipt-bound artifacts.",
            "After both ep4 checkpoints exist, run paired valid825 and clean hardware-heldout replay before any benefit claim.",
        ],
    }
    output = REVIEW / "m73_m77_m87_r2_independent_hammer.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS R2 hammer open_p0={} p1={} p2={} verdict={}".format(
        payload["severity_counts"]["P0"], payload["severity_counts"]["P1"],
        payload["severity_counts"]["P2"],
        payload["verdict"]["existing_single_known_successor_with_working_nvidia_smi"]))


if __name__ == "__main__":
    main()
