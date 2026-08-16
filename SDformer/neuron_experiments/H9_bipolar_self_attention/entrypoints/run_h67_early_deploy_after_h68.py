"""Evaluate the H67 rank-1 checkpoint on the frozen dyadic deployment after H68."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from run_h60_family_deploy_eval import make_deploy_config, parse_profile, run_eval


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
QUEUE_STATUS = RESULTS / "h67_h68_full30_queue_status.log"
STATUS = RESULTS / "h67_early_deploy_after_h68_status.log"
SOURCE = GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml"
RUN = RESULTS / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid"
EPOCH = 19
OUTPUT = RUN / "standard_dyadic_int8_valid825" / f"epoch{EPOCH}"
SUMMARY = RUN / "h67_epoch19_dyadic_int8_valid825.json"


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def wait_h68() -> None:
    marker = "ALL COMPLETE H67 -> H68 full30 queue"
    while not QUEUE_STATUS.exists() or marker not in QUEUE_STATUS.read_text(encoding="utf-8", errors="ignore"):
        record(f"WAIT H68 full30+valid825: {QUEUE_STATUS}")
        time.sleep(600)


def append_result(row: dict) -> None:
    marker = "H67_EPOCH19_EARLY_DYADIC_DEPLOY_VALID825"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### H67 epoch19 early dyadic deploy valid825 自动结果\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- artifact: `{SUMMARY.relative_to(REPO)}`\n")
        handle.write(f"- AEE: `{row['AEE']:.4f}`; AAE: `{row['AAE']:.4f}`\n")
        handle.write(f"- total_spikes: `{row['total_spikes_g']:.4f}G`; firing: `{row['firing']*100:.4f}%`\n")
        handle.write(f"- spike-energy proxy: `{row['spike_energy_proxy_uj']:.2f}uJ`\n")
        handle.write("- deployment: alpha0=1/64, score/gate step=1/128; attention operation cost remains separate.\n")


def main() -> int:
    wait_h68()
    deploy = make_deploy_config(SOURCE)
    checkpoint = RUN / f"checkpoint_epoch{EPOCH}.pth"
    record(f"START H67 epoch{EPOCH} dyadic deploy: config={deploy} checkpoint={checkpoint}")
    run_eval(deploy, checkpoint, OUTPUT)
    row = {
        "candidate": "H67 Motion-XOR TTX",
        "epoch": EPOCH,
        "config": str(deploy),
        "checkpoint": str(checkpoint),
        "profile": str(OUTPUT / "spike_profile.json"),
        **parse_profile(OUTPUT / "spike_profile.json"),
    }
    SUMMARY.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    append_result(row)
    record(f"ALL COMPLETE H67 EARLY DEPLOY: {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
