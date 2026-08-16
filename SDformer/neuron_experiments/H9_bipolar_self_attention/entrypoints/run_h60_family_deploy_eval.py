"""Evaluate each H60-family full30 winner under the frozen dyadic INT8 deployment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path(sys.executable)
SUMMARY_JSON = RESULTS / "h60_family_dyadic_int8_deploy_valid825.json"
SUMMARY_MD = RESULTS / "h60_family_dyadic_int8_deploy_valid825.md"


def quant_grid(minimum: float, maximum: float, step: float) -> dict[str, int | float]:
    levels = int(round((maximum - minimum) / step)) + 1
    return {
        "min": minimum,
        "max": maximum,
        "step": step,
        "levels": levels,
        "minimum_code_bits": int(math.ceil(math.log2(levels))),
    }


def fixed_cases() -> list[tuple[str, Path, Path, int | None]]:
    cases = [
        (
            "TTX frozen mainline",
            GEN / "date11full_ttx_dyadic_txonly_all12_deploy_int8.yml",
            RESULTS / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid",
            2,
        ),
        (
            "H67 Motion-XOR TTX",
            GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml",
            RESULTS / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid",
            None,
        ),
        (
            "H68 Castling-inspired deploy",
            GEN / "h68_allbinary_all12_castling_ttx_deploy_full30.yml",
            RESULTS / "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid",
            None,
        ),
        (
            "H70 Event-Selective TTX",
            GEN / "h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30.yml",
            RESULTS / "h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30_bs8_full30_20260711_setsid",
            None,
        ),
        (
            "H71 Window-Context TTX",
            GEN / "h71_allbinary_all12_window_context_ttx_w720_fastlr_full30.yml",
            RESULTS / "h71_allbinary_all12_window_context_ttx_w720_fastlr_full30_bs8_full30_20260711_setsid",
            None,
        ),
        (
            "H66a Alpha-XNOR Matrix",
            GEN / "h66a_allbinary_all12_axnor_matrix_shiftmax_w720_fastlr_full30.yml",
            RESULTS / "h66a_allbinary_all12_axnor_matrix_shiftmax_w720_fastlr_full30_bs8_full30_20260712_setsid",
            None,
        ),
        (
            "H66b Hamming Linear",
            GEN / "h66b_allbinary_all12_hamming_linear_w720_fastlr_full30.yml",
            RESULTS / "h66b_allbinary_all12_hamming_linear_w720_fastlr_full30_bs8_full30_20260712_setsid",
            None,
        ),
        (
            "H66c Temporal-Pair TTX",
            GEN / "h66c_allbinary_all12_tp_ttx_w720_fastlr_full30.yml",
            RESULTS / "h66c_allbinary_all12_tp_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid",
            None,
        ),
        (
            "H66d Local-5 TTX",
            GEN / "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30.yml",
            RESULTS / "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid",
            None,
        ),
        (
            "H66e Temporal-Pair Self-Bias",
            GEN / "h66e_allbinary_all12_tp_ttx_selfbias1_w720_fastlr_full30.yml",
            RESULTS / "h66e_allbinary_all12_tp_ttx_selfbias1_w720_fastlr_full30_bs8_full30_20260712_setsid",
            None,
        ),
        (
            "H73 DE9 Match-Code",
            GEN / "h73_allbinary_all12_de9_match_code_w720_fastlr_full30.yml",
            RESULTS / "h73_allbinary_all12_de9_match_code_w720_fastlr_full30_bs8_full30_20260712_setsid",
            None,
        ),
        (
            "H74 MC49 Match-Code",
            GEN / "h74_allbinary_all12_mc49_match_code_w720_fastlr_full30.yml",
            RESULTS / "h74_allbinary_all12_mc49_match_code_w720_fastlr_full30_bs8_full30_20260712_setsid",
            None,
        ),
        (
            "H75 AX17 Match-Code",
            GEN / "h75_allbinary_all12_ax17_match_code_w720_fastlr_full30.yml",
            RESULTS / "h75_allbinary_all12_ax17_match_code_w720_fastlr_full30_bs8_full30_20260712_setsid",
            None,
        ),
        (
            "H76 PC9 Patch-Consistent Match-Code",
            GEN / "h76_allbinary_all12_pc9_patch_match_code_w720_fastlr_full30.yml",
            RESULTS / "h76_allbinary_all12_pc9_patch_match_code_w720_fastlr_full30_bs8_full30_20260713_setsid",
            None,
        ),
        (
            "H77 LC4 Contingency Match-Code",
            GEN / "h77_allbinary_all12_lc4_match_code_w720_fastlr_full30.yml",
            RESULTS / "h77_allbinary_all12_lc4_match_code_w720_fastlr_full30_bs8_full30_20260713_setsid",
            None,
        ),
        (
            "H78 G4 Grouped Match-Code",
            GEN / "h78_allbinary_all12_g4_match_code_w720_fastlr_full30.yml",
            RESULTS / "h78_allbinary_all12_g4_match_code_w720_fastlr_full30_bs8_full30_20260713_setsid",
            None,
        ),
        (
            "H79 CF10 Null-Assignment Match-Code",
            GEN / "h79_allbinary_all12_cf10_match_code_w720_fastlr_full30.yml",
            RESULTS / "h79_allbinary_all12_cf10_match_code_w720_fastlr_full30_bs8_full30_20260713_setsid",
            None,
        ),
        (
            "H80 DN9 Dual-Normalized Match-Code",
            GEN / "h80_allbinary_all12_dn9_match_code_w720_fastlr_full30.yml",
            RESULTS / "h80_allbinary_all12_dn9_match_code_w720_fastlr_full30_bs8_full30_20260713_setsid",
            None,
        ),
    ]
    h69_runs = sorted(RESULTS.glob("h69_allbinary_all12_dyadic_temperature_ttx_x*_w720_fastlr_full30_bs8_full30_20260711_setsid"))
    if len(h69_runs) != 1:
        raise RuntimeError(f"expected one promoted H69 full30 run, found {h69_runs}")
    h69_run = h69_runs[0]
    name = h69_run.name.split("_bs8_full30_", 1)[0]
    cases.insert(3, ("H69 Dyadic-Temperature TTX", GEN / f"{name}.yml", h69_run, None))
    return cases


def best_epoch(ranking: Path) -> int:
    for line in ranking.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*1\s*\|\s*(\d+)\s*\|", line)
        if match:
            return int(match.group(1))
    raise RuntimeError(f"cannot parse rank-1 epoch from {ranking}")


def make_deploy_config(source: Path) -> Path:
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    deploy = deepcopy(config)
    deploy["experiment"] = source.stem + "_dyadic_int8_deploy"
    deploy["bsa_attention"].update({
        "alpha0": 1.0 / 64.0,
        "castling_matrix_aux_weight": 0.0,
        "castling_matrix_aux_end_step": 0,
        "hardware_quant_enabled": True,
        "hardware_mu_pow2_shift": 0,
        "hardware_score_step": 1.0 / 128.0,
        "hardware_score_min": -2.0,
        "hardware_score_max": 2.0,
        "hardware_gate_step": 1.0 / 128.0,
        "hardware_gate_min": 0.0,
        "hardware_gate_max": 2.0,
        "match_code_weight_quant_enabled": True,
        "match_code_weight_step": 1.0 / 128.0,
        "match_code_weight_min": -1.0,
        "match_code_weight_max": 127.0 / 128.0,
    })
    deploy["note"] = (
        "Frozen DATE deployment audit: alpha0=1/64, INT8 score/gate, candidate-specific "
        "all12 attention mechanism preserved, Castling auxiliary disabled, and Match-Code "
        "static weights quantized to a signed 2^-7 grid when present."
    )
    path = GEN / f"{deploy['experiment']}.yml"
    path.write_text(yaml.safe_dump(deploy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def profile_artifact_status(
    profile: Path,
    config: Path,
    checkpoint: Path,
) -> str:
    """Return ``match``, ``legacy``, or ``mismatch`` for an existing profile."""
    raw = json.loads(profile.read_text(encoding="utf-8"))
    identity = raw.get("artifact_identity")
    if not isinstance(identity, dict):
        return "legacy"
    stat = checkpoint.stat()
    expected = {
        "config_path": str(config.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
    }
    return (
        "match"
        if all(identity.get(key) == value for key, value in expected.items())
        else "mismatch"
    )


def run_eval(config: Path, checkpoint: Path, output: Path) -> None:
    profile = output / "spike_profile.json"
    if profile.exists():
        status = profile_artifact_status(profile, config, checkpoint)
        if status == "mismatch":
            raise RuntimeError(
                f"refusing to reuse stale deploy profile: {profile}; "
                "use an artifact-fingerprinted output directory"
            )
        if status == "legacy":
            print(
                f"[deploy] reusing legacy profile without artifact identity: {profile}",
                flush=True,
            )
        return
    env = os.environ.copy()
    env.update({
        "SDFORMER_USE_MLFLOW": "0",
        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
        "SDFORMER_SNN_BACKEND": "cupy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(PY), "-u", "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
        "--config", str(config), "--checkpoint", str(checkpoint),
        "--path_results", str(output), "--mode", "valid",
    ]
    with (output / "eval.log").open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"deploy eval failed: {output / 'eval.log'}")


def parse_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    return {
        "AEE": float(metrics["AEE"]),
        "AAE": float(metrics["AAE"]),
        "AAE_Benchmark": float(metrics.get("AAE_Benchmark", "nan")),
        "DSEC_Fl": float(metrics.get("DSEC_Fl", "nan")),
        "PE1": float(metrics["AEE_PE1"]),
        "PE2": float(metrics["AEE_PE2"]),
        "outlier": float(metrics["AEE_outliers"]),
        "outlier_scope": "legacy_prediction_magnitude_fraction",
        "total_spikes_g": float(data["total_spikes"]) / 1e9,
        "firing": float(data["global_firing_rate"]),
        "spike_energy_proxy_uj": float(data["energy_uj"]),
        "samples": int(data["samples"]),
        "energy_scope": data.get("energy_scope", "legacy_unlabeled_spike_proxy"),
    }


def write(rows: list[dict[str, Any]]) -> None:
    quantization = {
        "score": quant_grid(-2.0, 2.0, 1.0 / 128.0),
        "gate": quant_grid(0.0, 2.0, 1.0 / 128.0),
        "warning": "historical INT8 label means a 2^-7 grid; exact inclusive endpoints require 10/9-bit codes",
    }
    SUMMARY_JSON.write_text(json.dumps({"quantization": quantization, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# H60-family Dyadic INT8 Deploy Valid825", "",
        "Each row evaluates the float-valid825 winner with alpha0=1/64 and INT8 score/gate.", "",
        "Historical naming caveat: score [-2,2] at 1/128 has 513 levels (minimum 10 code bits); gate [0,2] has 257 levels (minimum 9 bits).", "",
        "| candidate | epoch | AEE | AAE | PE1 | PE2 | outlier | spikes(G) | firing | spike-energy proxy(uJ) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
            f"{row['PE1']:.4f} | {row['PE2']:.4f} | {row['outlier']:.4f} | "
            f"{row['total_spikes_g']:.4f} | {row['firing']*100:.4f}% | {row['spike_energy_proxy_uj']:.2f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    marker = "H60_FAMILY_DYADIC_INT8_DEPLOY_VALID825"
    if marker not in REDESIGN.read_text(encoding="utf-8"):
        with REDESIGN.open("a", encoding="utf-8") as handle:
            handle.write("\n\n### H60-family dyadic INT8 deploy valid825 自动结果\n\n")
            handle.write(f"<!-- {marker} -->\n")
            handle.write(f"- summary: `{SUMMARY_MD.relative_to(REPO)}`\n\n")
            for line in lines[6:]:
                handle.write(line + "\n")


def main() -> int:
    rows = []
    for candidate, source_config, run_dir, forced_epoch in fixed_cases():
        epoch = forced_epoch if forced_epoch is not None else best_epoch(run_dir / "profile_ranking_valid825.md")
        config = make_deploy_config(source_config)
        checkpoint = run_dir / f"checkpoint_epoch{epoch}.pth"
        output = run_dir / "standard_dyadic_int8_valid825" / f"epoch{epoch}"
        run_eval(config, checkpoint, output)
        rows.append({
            "candidate": candidate,
            "epoch": epoch,
            "config": str(config),
            "checkpoint": str(checkpoint),
            "profile": str(output / "spike_profile.json"),
            **parse_profile(output / "spike_profile.json"),
        })
    write(rows)
    print(SUMMARY_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
