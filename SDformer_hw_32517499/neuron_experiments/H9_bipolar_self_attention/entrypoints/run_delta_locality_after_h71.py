"""Run the exact Delta-TTX locality audit after the software full30 queue."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RESULTS = EXP / "results"
PY = Path(sys.executable)
PREV_STATUS = RESULTS / "ttb_cycle_profile_v2_after_round3_status.log"
STATUS = RESULTS / "delta_locality_after_h71_status.log"
CONFIG = EXP / "configs/generated/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
CHECKPOINT = RESULTS / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth"
OUTPUT = RESULTS / "date11_ttx_ep2_delta_locality_profile100_v2_20260711"
OPS_JSON = OUTPUT / "attention_candidate_ops.json"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
HARDWARE_DOC = REPO / "hw_autoresearch_nts07/docs/42_H67运动XOR与有界TTX硬件增量.md"


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def wait_previous() -> None:
    while True:
        if PREV_STATUS.exists() and "ALL COMPLETE TTB/DELTA CYCLE V2:" in PREV_STATUS.read_text(encoding="utf-8", errors="ignore"):
            return
        record(f"WAIT TTB/Delta cycle-v2 profile100: {PREV_STATUS}")
        time.sleep(600)


def run_profile() -> None:
    env = os.environ.copy()
    env.update({
        "SDFORMER_USE_MLFLOW": "0",
        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
        "SDFORMER_SNN_BACKEND": "cupy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    command = [
        str(PY), "-u", str(EXP / "entrypoints/profile_nts11_hardware_p0.py"),
        "--config", str(CONFIG), "--checkpoint", str(CHECKPOINT),
        "--output-dir", str(OUTPUT), "--samples", "100", "--num-workers", "0",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    record(f"START Delta-locality profile100: {' '.join(command)}")
    with (OUTPUT / "profile.log").open("a", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END Delta-locality profile100: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"Delta locality profile failed: {OUTPUT / 'profile.log'}")


def run_deploy_eval() -> None:
    command = [str(PY), "-u", str(EXP / "entrypoints/run_h60_family_deploy_eval.py")]
    record(f"START H60-family dyadic INT8 deploy valid825: {' '.join(command)}")
    with (RESULTS / "h60_family_dyadic_int8_deploy_valid825.log").open("a", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END H60-family dyadic INT8 deploy valid825: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError("H60-family deploy valid825 failed")


def run_temperature_clip_audit() -> None:
    command = [str(PY), "-u", str(EXP / "entrypoints/run_temperature_score_clip_audit.py")]
    log = RESULTS / "temperature_score_clip_profile20_launcher.log"
    record(f"START H69/H70 temperature score-clip audit: {' '.join(command)}")
    with log.open("a", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END H69/H70 temperature score-clip audit: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"temperature score-clip audit failed: {log}")


def deploy_summary_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    candidates = {str(row.get("candidate", "")) for row in rows}
    required = {
        "TTX frozen mainline",
        "H69 Dyadic-Temperature TTX",
        "H70 Event-Selective TTX",
        "H76 PC9 Patch-Consistent Match-Code",
        "H77 LC4 Contingency Match-Code",
        "H78 G4 Grouped Match-Code",
        "H79 CF10 Null-Assignment Match-Code",
        "H80 DN9 Dual-Normalized Match-Code",
    }
    return len(rows) == 19 and required <= candidates and all(int(row.get("samples", 0)) == 825 for row in rows)


def run_attention_audit(profile_json: Path) -> None:
    configs = [
        CONFIG,
        EXP / "configs/generated/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml",
        EXP / "configs/generated/h68_allbinary_all12_castling_ttx_deploy_full30.yml",
        EXP / "configs/generated/h69_allbinary_all12_dyadic_temperature_ttx_x4.yml",
        EXP / "configs/generated/h69_allbinary_all12_dyadic_temperature_ttx_x8.yml",
        EXP / "configs/generated/h69_allbinary_all12_dyadic_temperature_ttx_x16.yml",
        EXP / "configs/generated/h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30.yml",
        EXP / "configs/generated/h71_allbinary_all12_window_context_ttx_w720_fastlr_full30.yml",
        EXP / "configs/generated/h66a_allbinary_all12_axnor_matrix_shiftmax_w720_fastlr_full30.yml",
        EXP / "configs/generated/h66b_allbinary_all12_hamming_linear_w720_fastlr_full30.yml",
        EXP / "configs/generated/h66c_allbinary_all12_tp_ttx_w720_fastlr_full30.yml",
        EXP / "configs/generated/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30.yml",
        EXP / "configs/generated/h66e_allbinary_all12_tp_ttx_selfbias1_w720_fastlr_full30.yml",
        EXP / "configs/generated/h73_allbinary_all12_de9_match_code_w720_fastlr_full30.yml",
        EXP / "configs/generated/h74_allbinary_all12_mc49_match_code_w720_fastlr_full30.yml",
        EXP / "configs/generated/h75_allbinary_all12_ax17_match_code_w720_fastlr_full30.yml",
        EXP / "configs/generated/h76_allbinary_all12_pc9_patch_match_code_w720_fastlr_full30.yml",
        EXP / "configs/generated/h77_allbinary_all12_lc4_match_code_w720_fastlr_full30.yml",
        EXP / "configs/generated/h78_allbinary_all12_g4_match_code_w720_fastlr_full30.yml",
        EXP / "configs/generated/h79_allbinary_all12_cf10_match_code_w720_fastlr_full30.yml",
        EXP / "configs/generated/h80_allbinary_all12_dn9_match_code_w720_fastlr_full30.yml",
    ]
    command = [
        str(PY), str(EXP / "entrypoints/audit_attention_candidate_ops.py"),
        "--profile-json", str(profile_json), "--output", str(OPS_JSON),
    ]
    for config in configs:
        command.extend(["--config", str(config)])
    record(f"START attention operation audit: {' '.join(command)}")
    proc = subprocess.run(command, cwd=REPO)
    record(f"END attention operation audit: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError("attention operation audit failed")


def append_document(path: Path, marker: str, title: str, delta: dict) -> None:
    current = path.read_text(encoding="utf-8")
    if marker in current:
        return
    histogram = [
        ("0", delta["delta_update_count_0"]),
        ("1", delta["delta_update_count_1"]),
        ("2", delta["delta_update_count_2"]),
        ("3--4", delta["delta_update_count_3_4"]),
        ("5--8", delta["delta_update_count_5_8"]),
        ("9--16", delta["delta_update_count_9_16"]),
        ("17+", delta["delta_update_count_17_plus"]),
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n## {title}\n\n<!-- {marker} -->\n")
        handle.write(f"- profile: `{OUTPUT.relative_to(REPO)}`\n")
        handle.write(f"- zero-update token/head: `{delta['delta_zero_update_token_ratio']:.6%}`\n")
        handle.write(f"- mean changed-token run length: `{delta['delta_mean_changed_run_length']:.4f}`\n")
        handle.write(f"- empty 4-token bundle: `{delta['delta_bundle4_empty_ratio']:.6%}`\n")
        handle.write(f"- empty 8-token bundle: `{delta['delta_bundle8_empty_ratio']:.6%}`\n\n")
        handle.write("| updated lanes/token/head | count |\n|---|---:|\n")
        for label, count in histogram:
            handle.write(f"| {label} | {count} |\n")


def append_ops(path: Path, rows: list[dict]) -> None:
    marker = "ATTENTION_CANDIDATE_OP_AUDIT_V2"
    if marker in path.read_text(encoding="utf-8"):
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n## H60-family attention operation audit v2\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- artifact: `{OPS_JSON.relative_to(REPO)}`\n")
        handle.write("- standard `energy_uj` excludes these incremental attention operations.\n\n")
        handle.write("| candidate | incremental logic/sample | add/sample | fixed MAC/sample | proxy uJ | vs base TX-score |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            counts = row["counts_per_sample"]
            handle.write(
                f"| {row['experiment']} | {counts['incremental_logic']:.0f} | "
                f"{counts['incremental_add']:.0f} | {counts['incremental_mac']:.0f} | "
                f"{row['incremental_attention_proxy_uj']:.3f} | "
                f"{row['incremental_vs_base_tx_score_pct']:.2f}% |\n"
            )
        handle.write("\nProxy 使用 45 nm logic/add/MAC 常数，只用于统一操作审计，不替代 post-layout 或 SRAM/NoC 能耗。\n")


def main() -> int:
    wait_previous()
    run_temperature_clip_audit()
    deploy_summary = RESULTS / "h60_family_dyadic_int8_deploy_valid825.json"
    if not deploy_summary_complete(deploy_summary):
        run_deploy_eval()
    profile_json = OUTPUT / "nts11_hardware_p0_profile.json"
    if not profile_json.exists():
        run_profile()
    if not OPS_JSON.exists():
        run_attention_audit(profile_json)
    result = json.loads(profile_json.read_text(encoding="utf-8"))
    delta = result["summary"]["delta_ttx"]
    ops = json.loads(OPS_JSON.read_text(encoding="utf-8"))["rows"]
    append_document(REDESIGN, "DELTA_LOCALITY_PROFILE100_V2", "43.14 Exact Delta-TTX locality profile100 v2", delta)
    append_document(HARDWARE_DOC, "DELTA_LOCALITY_PROFILE100_V2", "Delta-Locality profile100 v2 自动结果", delta)
    append_ops(REDESIGN, ops)
    append_ops(HARDWARE_DOC, ops)
    record(f"ALL COMPLETE Delta-locality v2: {profile_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
