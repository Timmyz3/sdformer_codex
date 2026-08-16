"""Evaluate the H67 full-resolution ep30 winner on frozen deploy numerics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from run_dsec_fullres_paper_w15_deploy_followup import (
    make_deploy_configs,
    parse_profile,
    protocol_from_profile,
    run_or_reuse_eval,
)


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep30.yml"
RUN = EXP / "results/dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804"
CHECKPOINT = RUN / "checkpoint_epoch30.pth"
FLOAT_PROFILE = RUN / "standard_valid825/epoch30/spike_profile.json"
STATUS = RUN / "ep30_deploy_eval_status.log"
SUMMARY_JSON = RUN / "ep30_deploy_summary.json"
SUMMARY_MD = RUN / "ep30_deploy_summary.md"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def append_result() -> None:
    marker = "H67_FULLRES_EP30_DEPLOY_NUMERIC_20260805"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### H67 fullres ep30 部署数值评估（2026-08-05）\n\n")
        handle.write(f"<!-- {marker} -->\n\n")
        handle.write(SUMMARY_MD.read_text(encoding="utf-8"))


def main() -> int:
    for required in (CONFIG, CHECKPOINT, FLOAT_PROFILE):
        if not required.is_file():
            raise FileNotFoundError(required)
    protocol_from_profile(FLOAT_PROFILE)

    dyadic_config, hardware_config = make_deploy_configs("H67", CONFIG)
    dyadic_profile, dyadic = run_or_reuse_eval(
        "H67 ep30 dyadic Q7/Q1.7 fullres valid825",
        dyadic_config,
        CHECKPOINT,
        RUN / "deploy_valid825/dyadic_q7q17/epoch30",
    )
    hardware_profile, hardware = run_or_reuse_eval(
        "H67 ep30 hardware-order Q7/Q1.7 fullres valid825",
        hardware_config,
        CHECKPOINT,
        RUN / "deploy_valid825/hardware_order_q7q17/epoch30",
    )
    floating = parse_profile(FLOAT_PROFILE)
    result = {
        "scope": "attention_core_hardware_order_numeric_not_full_network_rtl_exact",
        "checkpoint": str(CHECKPOINT),
        "float_profile": str(FLOAT_PROFILE),
        "dyadic_profile": str(dyadic_profile),
        "hardware_order_profile": str(hardware_profile),
        "float": floating,
        "dyadic": dyadic,
        "hardware_order": hardware,
    }
    SUMMARY_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# H67 fullres ep30 deploy numeric summary",
        "",
        "Scope: attention-core hardware-order numeric; this is not full-network RTL-exact or T450 SV sign-off.",
        "",
        "| path | AEE | AAE legacy | AAE benchmark | spikes(G) | energy proxy(uJ) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in (("float", floating), ("dyadic Q7/Q1.7", dyadic), ("hardware-order", hardware)):
        lines.append(
            f"| {label} | {metrics['AEE']:.4f} | {metrics['AAE']:.4f} | "
            f"{metrics['AAE_Benchmark']:.4f} | {metrics['total_spikes_g']:.4f} | "
            f"{metrics['spike_energy_proxy_uj']:.2f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_result()
    record("ALL COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
