#!/usr/bin/env python3
"""Fail-closed report for checkpoint-bound ATLIF DP-TME component replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_RE = re.compile(
    r"ATLIF_DPTME_RESULT commands=(\d+) hidden=(\d+) hidden_mismatches=(\d+) "
    r"events=(\d+) event_mismatches=(\d+) sampled_protocol_errors=(\d+)"
)
PROTOCOL_RE = re.compile(
    r"DPTME_PROTOCOL_RESULT sampled_protocol_errors=(\d+) tag_reject=(\d+) "
    r"early_last_reject=(\d+) single_step_reject=(\d+) state_advance_errors=(\d+)"
)
EXPECTED_RTL_COUNTS = {
    "commands": 81,
    "hidden": 25_920,
    "events": 25_920,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sim(path: Path, simulator: str) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    if f"SIMULATOR={simulator}" not in text:
        raise RuntimeError(f"simulator identity mismatch: {path}")
    if simulator == "verilator" and "ASSERTIONS=enabled" not in text:
        raise RuntimeError(f"Verilator SVA runtime is not proven enabled: {path}")
    match = RESULT_RE.search(text)
    if match is None or "PASS: checkpoint-bound ATLIF DP-TME RTL exact" not in text:
        raise RuntimeError(f"incomplete simulation log: {path}")
    keys = (
        "commands",
        "hidden",
        "hidden_mismatches",
        "events",
        "event_mismatches",
        "sampled_protocol_errors",
    )
    result = dict(zip(keys, (int(value) for value in match.groups())))
    if (
        result["hidden_mismatches"]
        or result["event_mismatches"]
        or result["sampled_protocol_errors"]
    ):
        raise RuntimeError(f"RTL mismatch: {path}: {result}")
    for key, expected in EXPECTED_RTL_COUNTS.items():
        if result[key] != expected:
            raise RuntimeError(
                f"RTL comparison count mismatch: {path}: {key}={result[key]} expected={expected}"
            )
    return result


def parse_protocol_sim(
    path: Path, simulator: str, require_assertions: bool = False
) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    if f"SIMULATOR={simulator}" not in text:
        raise RuntimeError(f"directed simulator identity mismatch: {path}")
    if require_assertions and "ASSERTIONS=enabled" not in text:
        raise RuntimeError(f"directed Verilator SVA runtime is not proven enabled: {path}")
    match = PROTOCOL_RE.search(text)
    if match is None or "PASS: HIT-Flow DP-TME array" not in text:
        raise RuntimeError(f"incomplete directed protocol log: {path}")
    keys = (
        "sampled_protocol_errors",
        "tag_reject",
        "early_last_reject",
        "single_step_reject",
        "state_advance_errors",
    )
    result = dict(zip(keys, (int(value) for value in match.groups())))
    expected = {
        "sampled_protocol_errors": 3,
        "tag_reject": 1,
        "early_last_reject": 1,
        "single_step_reject": 1,
        "state_advance_errors": 0,
    }
    if result != expected:
        raise RuntimeError(f"directed protocol coverage mismatch: {path}: {result}")
    return result


def validate_commands(manifest: dict) -> None:
    commands = manifest.get("commands")
    summary = manifest.get("summary", {})
    if not isinstance(commands, list) or len(commands) != 81:
        raise RuntimeError("ATLIF manifest must contain exactly 81 commands")

    names: set[str] = set()
    temporal_counts = {2: 0, 10: 0}
    captured_events = 0
    fixed_float_mismatches = 0
    model_mismatches = 0
    for expected_tag, command in enumerate(commands, start=1):
        name = command.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise RuntimeError(f"invalid or duplicate ATLIF site name: {name!r}")
        names.add(name)
        if int(command.get("tag", -1)) != expected_tag:
            raise RuntimeError(f"non-contiguous ATLIF command tag: {command}")

        temporal = int(command.get("temporal_steps", -1))
        if temporal not in temporal_counts:
            raise RuntimeError(f"unsupported ATLIF temporal length: {name}: {temporal}")
        temporal_counts[temporal] += 1
        expected_lanes = (
            {"ordinary": 10, "near_threshold": 10, "max_amplitude": 12}
            if temporal == 10
            else {"ordinary": 53, "near_threshold": 53, "max_amplitude": 54}
        )
        if (
            command.get("scenario") != "mixed_ordinary_near_threshold_max_amplitude"
            or command.get("scenario_lane_counts") != expected_lanes
        ):
            raise RuntimeError(f"invalid ATLIF lane scenarios: {name}")

        clip_counts = command.get("clip_counts")
        if (
            not isinstance(clip_counts, dict)
            or set(clip_counts) != {"input", "weight", "bias", "threshold"}
            or any(int(value) != 0 for value in clip_counts.values())
            or int(command.get("accumulator_overflow_count", -1)) != 0
        ):
            raise RuntimeError(f"ATLIF clipping/overflow is not zero: {name}")
        if int(command.get("hidden_min", -(1 << 24))) < -(1 << 23) or int(
            command.get("hidden_max", 1 << 24)
        ) > (1 << 23) - 1:
            raise RuntimeError(f"ATLIF hidden value exceeds Acc24: {name}")
        for scale_name in ("x_scale", "weight_scale", "accumulator_scale"):
            scale = float(command.get(scale_name, float("nan")))
            if not math.isfinite(scale) or scale <= 0.0:
                raise RuntimeError(f"invalid ATLIF scale {scale_name}: {name}: {scale}")
        if command.get("output_contract") != "one_bit_event_plus_checkpoint_static_threshold_scale":
            raise RuntimeError(f"invalid ATLIF output contract: {name}")

        expected_events = 320
        if int(command.get("captured_events", -1)) != expected_events:
            raise RuntimeError(f"invalid ATLIF event count: {name}")
        captured_events += expected_events
        fixed_float_mismatches += int(command.get("fixed_vs_float_event_mismatches", -1))
        model_mismatches += int(command.get("model_reference_mismatches", -1))

    if temporal_counts != {2: 36, 10: 45}:
        raise RuntimeError(f"invalid ATLIF temporal site split: {temporal_counts}")
    if (
        captured_events != 25_920
        or captured_events != int(summary.get("captured_events", -1))
        or fixed_float_mismatches != int(summary.get("fixed_vs_float_event_mismatches", -1))
        or model_mismatches != 0
        or model_mismatches != int(summary.get("model_reference_mismatches", -1))
    ):
        raise RuntimeError("ATLIF command totals do not match manifest summary")
    expected_ratio = fixed_float_mismatches / captured_events
    if not math.isclose(
        expected_ratio,
        float(summary.get("fixed_vs_float_event_mismatch_ratio", float("nan"))),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("ATLIF fixed/float mismatch ratio is inconsistent")


def validate_site_coverage(manifest: dict) -> None:
    coverage = manifest.get("site_coverage") or {}
    expected_counts = {"installed": 105, "called": 93, "dead_called": 12, "replayed": 81}
    sets = {}
    for key, expected_count in expected_counts.items():
        row = coverage.get(key) or {}
        names = row.get("names") or []
        if (
            int(row.get("count", -1)) != expected_count
            or len(names) != expected_count
            or len(set(names)) != expected_count
            or names != sorted(names)
        ):
            raise RuntimeError(f"invalid ATLIF {key} site coverage")
        payload = json.dumps(names, ensure_ascii=True, separators=(",", ":")).encode()
        if row.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise RuntimeError(f"ATLIF {key} site coverage SHA mismatch")
        sets[key] = set(names)
    if (
        not sets["called"].issubset(sets["installed"])
        or not sets["dead_called"].issubset(sets["called"])
        or sets["called"] - sets["dead_called"] != sets["replayed"]
        or any(
            not name.endswith(".attn.attn_sn.spiking_neuron")
            for name in sets["dead_called"]
        )
    ):
        raise RuntimeError("ATLIF installed/called/dead/replayed set relation failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    vector_dir = args.vector_dir.resolve()
    result_dir = args.result_dir.resolve()
    manifest_path = vector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "checkpoint_atlif_dptme_vectors_v1":
        raise RuntimeError("unexpected vector schema")
    summary = manifest["summary"]
    if (
        int(summary.get("live_sites", 0)) != 81
        or int(summary.get("live_t10_sites", 0)) != 45
        or int(summary.get("live_t2_sites", 0)) != 36
        or summary.get("selection_scenarios") != ["ordinary", "near_threshold", "max_amplitude"]
        or int(summary.get("commands", 0)) != 81
        or int(summary.get("model_reference_mismatches", -1)) != 0
    ):
        raise RuntimeError(f"invalid checkpoint ATLIF capture: {summary}")
    validate_commands(manifest)
    validate_site_coverage(manifest)
    identity = manifest.get("identity", {})
    audit = identity.get("checkpoint_load_audit") or {}
    if (
        identity.get("resolution") != [480, 640]
        or identity.get("crop") is not None
        or identity.get("window_size") != [2, 15, 15]
        or identity.get("bn_policy") != "no_running"
        or identity.get("module_counts") != {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12}
        or int(audit.get("checkpoint_overlay_keys", -1)) != 210
        or int(audit.get("model_overlay_keys", -1)) != 210
        or int(audit.get("missing_count", -1)) != 0
        or int(audit.get("unexpected_count", -1)) != 0
    ):
        raise RuntimeError(f"invalid checkpoint/protocol identity: {identity}")
    config_path = Path(str(identity.get("config_path", "")))
    checkpoint_path = Path(str(identity.get("checkpoint_path", "")))
    if (
        not config_path.is_file()
        or not checkpoint_path.is_file()
        or sha256(config_path) != identity.get("config_sha256")
        or sha256(checkpoint_path) != identity.get("checkpoint_sha256")
        or checkpoint_path.stat().st_size != int(identity.get("checkpoint_size", -1))
    ):
        raise RuntimeError("config/checkpoint artifact identity mismatch")
    for source, expected in manifest.get("source_sha256", {}).items():
        if sha256(Path(source)) != expected:
            raise RuntimeError(f"vector source hash mismatch: {source}")
    icarus = parse_sim(result_dir / "icarus.log", "icarus")
    verilator = parse_sim(result_dir / "verilator.log", "verilator")
    if icarus != verilator or icarus["commands"] != int(summary["commands"]):
        raise RuntimeError("Icarus/Verilator/manifest transaction mismatch")
    directed_icarus = parse_protocol_sim(
        result_dir / "directed_icarus.log", "icarus"
    )
    directed_verilator = parse_protocol_sim(
        result_dir / "directed_verilator.log",
        "verilator",
        require_assertions=True,
    )
    if directed_icarus != directed_verilator:
        raise RuntimeError("Icarus/Verilator directed protocol coverage mismatch")
    lint = (result_dir / "verilator_lint.log").read_text(encoding="utf-8")
    yosys = (result_dir / "yosys.log").read_text(encoding="utf-8")
    if "%Error" in lint or "ERROR:" in yosys or "Found and reported" not in yosys:
        raise RuntimeError("lint or Yosys did not pass fail-closed checks")

    sources = [
        ROOT / "rtl_hitflow/hitflow_dptme_array.sv",
        ROOT / "tb_hitflow/tb_checkpoint_atlif_dptme.sv",
        ROOT / "tb_hitflow/tb_hitflow_dptme_array.sv",
        ROOT / "sim_hitflow/run_checkpoint_atlif_dptme_checks.sh",
        ROOT / "verif_hitflow/hitflow_dptme_assertions.sv",
        ROOT / "verif_hitflow/bind_hitflow_dptme_assertions.sv",
        ROOT / "scripts/generate_checkpoint_atlif_dptme_vectors.py",
        Path(__file__).resolve(),
        manifest_path,
    ]
    report = {
        "status": "PASS",
        "evidence_scope": (
            "checkpoint_bound_atlif_temporal_matrix_int8_acc24_component_rtl_exact_"
            "not_output_scale_folding_bn_requant_residual_full_encoder_or_full_network"
        ),
        "checkpoint_identity": manifest["identity"],
        "numeric_contract": manifest["numeric_contract"],
        "numeric_bridge": {
            "captured_events": int(summary["captured_events"]),
            "fixed_vs_float_event_mismatches": int(summary["fixed_vs_float_event_mismatches"]),
            "fixed_vs_float_event_mismatch_ratio": float(summary["fixed_vs_float_event_mismatch_ratio"]),
            "deployment_accuracy_signoff": False,
            "reason": (
                "component vectors quantify local event flips only; valid825 must be rerun with "
                "static site scales and downstream event-times-threshold folding"
            ),
        },
        "rtl": icarus,
        "directed_protocol": directed_icarus,
        "checks": {
            "icarus_zero_mismatch": True,
            "verilator_zero_mismatch": True,
            "verilator_lint": True,
            "yosys_check_assert": True,
            "checkpoint_overlay210_load_audit": True,
            "atlif105_shiftmax12": True,
            "live81_t10_45_t2_36": True,
            "installed105_called93_dead12_replayed81": True,
            "three_lane_scenarios_per_site": True,
            "input_output_backpressure": True,
            "zero_sampled_protocol_errors": True,
            "tag_early_last_single_step_rejects": True,
            "directed_state_nonadvance": True,
            "sva": True,
        },
        "source_sha256": {str(path): sha256(path) for path in sources},
        "evidence_artifact_sha256": {
            path.name: sha256(path)
            for path in (
                result_dir / "icarus.log",
                result_dir / "verilator.log",
                result_dir / "directed_icarus.log",
                result_dir / "directed_verilator.log",
                result_dir / "verilator_lint.log",
                result_dir / "yosys.log",
            )
        },
    }
    (result_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "rtl": report["rtl"], "numeric_bridge": report["numeric_bridge"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
