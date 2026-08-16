#!/usr/bin/env python3
"""RTL-calibrated multi-sample Fixed2S vs RQTB2S cycle/slot model for Motion DATE evidence.

Evidence level: [prof]+[rtl校准模型]
- Calibrates linear cycle/exp models on sample0/window0 138-row Fixed2S RTL report.
- Applies to all 100-sample ordered count traces in H67 fullres T450 profile.
- Does NOT claim real multi-sample RTL cycles or ASIC energy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.profile_h67_zkqi_multisample_ordered import (
        BLOCK_RE,
        EXPECTED_DEPTHS,
        EXPECTED_HEADS,
        EXPECTED_NAMES,
        EXPECTED_WINDOWS,
        decode_trace,
        h67_score_from_counts,
        receipt,
    )
except ModuleNotFoundError:
    from profile_h67_zkqi_multisample_ordered import (
        BLOCK_RE,
        EXPECTED_DEPTHS,
        EXPECTED_HEADS,
        EXPECTED_NAMES,
        EXPECTED_WINDOWS,
        decode_trace,
        h67_score_from_counts,
        receipt,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT
    / "results/h67_fullres_ep30_t450_profile100_20260805"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_RTL = (
    ROOT / "results/h67_rqtb_strong_baseline_2s_t450_20260809/report.json"
)
DEFAULT_OUT = ROOT / "results/h67_rqtb_fixed2s_multisample_model_20260811"
PAIRS = 225
FIXED_SLOTS = 450


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_linear(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ coef
    resid = y - pred
    ss_res = float(resid @ resid)
    ss_tot = float((y - y.mean()) @ (y - y.mean()))
    metrics = {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "mae": float(np.mean(np.abs(resid))),
        "max_abs_resid": float(np.max(np.abs(resid))),
        "n": int(y.size),
    }
    return coef.astype(np.float64), metrics


def apply_linear(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return Xb @ coef


def dist(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty distribution")
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
        "n": int(arr.size),
    }


def calibrate(rtl_report: dict[str, Any]) -> dict[str, Any]:
    rows = rtl_report["rows_2s"]
    active = np.array([r["active"] for r in rows], dtype=np.float64)
    equal = np.array([r["equal"] for r in rows], dtype=np.float64)
    f_slots = np.array([r["fixed_slots"] for r in rows], dtype=np.float64)
    r_slots = np.array([r["rqtb_slots"] for r in rows], dtype=np.float64)
    f_exp = np.array([r["fixed_exp"] for r in rows], dtype=np.float64)
    r_exp = np.array([r["rqtb_exp"] for r in rows], dtype=np.float64)
    f_cyc = np.array([r["fixed_cycles"] for r in rows], dtype=np.float64)
    r_cyc = np.array([r["rqtb_cycles"] for r in rows], dtype=np.float64)

    # Contract checks on calibration set
    if not np.all(f_slots == FIXED_SLOTS):
        raise ValueError("calibration Fixed slots != 450")
    if not np.allclose(r_slots, FIXED_SLOTS - equal):
        raise ValueError("calibration RQTB slots != 450-equal")

    exp_x = np.column_stack([active, equal])
    f_exp_coef, f_exp_m = fit_linear(exp_x, f_exp)
    r_exp_coef, r_exp_m = fit_linear(exp_x, r_exp)

    f_pred_exp = apply_linear(f_exp_coef, exp_x)
    r_pred_exp = apply_linear(r_exp_coef, exp_x)

    f_cyc_coef, f_cyc_m = fit_linear(
        np.column_stack([active, equal, f_slots, f_exp]), f_cyc
    )
    r_cyc_coef, r_cyc_m = fit_linear(
        np.column_stack([active, equal, r_slots, r_exp]), r_cyc
    )

    # Closed-loop residual using predicted exp (deployment path)
    f_cyc_dep = apply_linear(
        f_cyc_coef,
        np.column_stack([active, equal, f_slots, f_pred_exp]),
    )
    r_cyc_dep = apply_linear(
        r_cyc_coef,
        np.column_stack([active, equal, r_slots, r_pred_exp]),
    )
    dep_metrics = {
        "fixed_mae": float(np.mean(np.abs(f_cyc - f_cyc_dep))),
        "rqtb_mae": float(np.mean(np.abs(r_cyc - r_cyc_dep))),
        "fixed_max_abs": float(np.max(np.abs(f_cyc - f_cyc_dep))),
        "rqtb_max_abs": float(np.max(np.abs(r_cyc - r_cyc_dep))),
        "speedup_true": float(f_cyc.sum() / r_cyc.sum()),
        "speedup_model": float(f_cyc_dep.sum() / r_cyc_dep.sum()),
    }

    return {
        "n_rows": len(rows),
        "fixed_slots_contract": FIXED_SLOTS,
        "pairs": PAIRS,
        "fixed_exp_coef": f_exp_coef.tolist(),
        "rqtb_exp_coef": r_exp_coef.tolist(),
        "fixed_cycle_coef": f_cyc_coef.tolist(),
        "rqtb_cycle_coef": r_cyc_coef.tolist(),
        "fit_metrics": {
            "fixed_exp": f_exp_m,
            "rqtb_exp": r_exp_m,
            "fixed_cycle_oracle_exp": f_cyc_m,
            "rqtb_cycle_oracle_exp": r_cyc_m,
            "deployment_path": dep_metrics,
        },
        "feature_order_exp": ["bias", "active", "equal"],
        "feature_order_cycle": ["bias", "active", "equal", "slots", "exp"],
    }


def row_stats_from_traces(
    q_count: np.ndarray,
    k_count: np.ndarray,
    overlap: np.ndarray,
    motion: np.ndarray,
) -> list[dict[str, float]]:
    """q/k/overlap: [2,B,H,N], motion: [B,H,N] with N=225 pairs."""
    scores = h67_score_from_counts(q_count, k_count, overlap, motion)
    # scores shape [2,B,H,N]
    s0, s1 = scores[0], scores[1]
    k0 = k_count[0]
    k1 = k_count[1]
    # active pair: at least one K non-zero (matches RTL active accounting closely)
    active_mask = (k0 > 0) | (k1 > 0)
    equal_mask = s0 == s1
    # RQTB merges when scores equal (including both-zero class merge)
    # RTL: rqtb_slots = 450 - equal, equal counts pairs with equal scores
    B, H, N = s0.shape
    rows: list[dict[str, float]] = []
    for b in range(B):
        for h in range(H):
            equal = int(np.count_nonzero(equal_mask[b, h]))
            active = int(np.count_nonzero(active_mask[b, h]))
            rqtb_slots = FIXED_SLOTS - equal
            rows.append(
                {
                    "active": float(active),
                    "equal": float(equal),
                    "fixed_slots": float(FIXED_SLOTS),
                    "rqtb_slots": float(rqtb_slots),
                }
            )
    return rows


def predict_rows(calib: dict[str, Any], rows: list[dict[str, float]]) -> dict[str, Any]:
    active = np.array([r["active"] for r in rows], dtype=np.float64)
    equal = np.array([r["equal"] for r in rows], dtype=np.float64)
    f_slots = np.array([r["fixed_slots"] for r in rows], dtype=np.float64)
    r_slots = np.array([r["rqtb_slots"] for r in rows], dtype=np.float64)
    exp_x = np.column_stack([active, equal])
    f_exp = apply_linear(np.asarray(calib["fixed_exp_coef"]), exp_x)
    r_exp = apply_linear(np.asarray(calib["rqtb_exp_coef"]), exp_x)
    f_cyc = apply_linear(
        np.asarray(calib["fixed_cycle_coef"]),
        np.column_stack([active, equal, f_slots, f_exp]),
    )
    r_cyc = apply_linear(
        np.asarray(calib["rqtb_cycle_coef"]),
        np.column_stack([active, equal, r_slots, r_exp]),
    )
    # physical non-negativity floor
    f_cyc = np.maximum(f_cyc, 1.0)
    r_cyc = np.maximum(r_cyc, 1.0)
    speedups = f_cyc / r_cyc
    return {
        "fixed_cycles_total": float(f_cyc.sum()),
        "rqtb_cycles_total": float(r_cyc.sum()),
        "fixed_slots_total": float(f_slots.sum()),
        "rqtb_slots_total": float(r_slots.sum()),
        "speedup": float(f_cyc.sum() / r_cyc.sum()),
        "slot_reduction": float(1.0 - r_slots.sum() / f_slots.sum()),
        "row_speedup_dist": dist(speedups.tolist()),
        "fixed_cycle_dist": dist(f_cyc.tolist()),
        "rqtb_cycle_dist": dist(r_cyc.tolist()),
        "n_rows": int(len(rows)),
        "rqtb_faster_rows": int(np.count_nonzero(speedups > 1.0)),
        "rqtb_slower_rows": int(np.count_nonzero(speedups < 1.0)),
        "equal_total": float(equal.sum()),
        "active_total": float(active.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--rtl-report", type=Path, default=DEFAULT_RTL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rtl = json.loads(args.rtl_report.read_text(encoding="utf-8"))
    calib = calibrate(rtl)

    print("loading profile...", flush=True)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    records = profile["summary"]["h60_records"]
    if len(records) != 1200:
        raise ValueError(f"expected 1200 h60 records, got {len(records)}")

    by_sample: dict[int, list[dict[str, float]]] = defaultdict(list)
    by_stage: dict[int, list[dict[str, float]]] = defaultdict(list)
    all_rows: list[dict[str, float]] = []

    for rec in records:
        name = str(rec["name"])
        match = BLOCK_RE.match(name)
        if not match:
            raise ValueError(f"bad block name {name}")
        stage = int(match.group("stage"))
        sample_id = int(rec["sample_id"])
        q = decode_trace(rec["pair_q_count_ordered_trace"])
        k = decode_trace(rec["pair_k_count_ordered_trace"])
        ov = decode_trace(rec["pair_overlap_ordered_trace"])
        mo = decode_trace(rec["pair_motion_ordered_trace"])
        # shapes [2,B,H,N]
        rows = row_stats_from_traces(q, k, ov, mo)
        for row in rows:
            row["stage"] = float(stage)
            row["sample_id"] = float(sample_id)
            row["name"] = name
        by_sample[sample_id].extend(rows)
        by_stage[stage].extend(rows)
        all_rows.extend(rows)

    global_stats = predict_rows(calib, all_rows)
    sample_speedups = []
    sample_details = {}
    for sid in sorted(by_sample):
        st = predict_rows(calib, by_sample[sid])
        sample_speedups.append(st["speedup"])
        sample_details[str(sid)] = {
            "speedup": st["speedup"],
            "slot_reduction": st["slot_reduction"],
            "fixed_cycles_total": st["fixed_cycles_total"],
            "rqtb_cycles_total": st["rqtb_cycles_total"],
            "n_rows": st["n_rows"],
            "rqtb_faster_rows": st["rqtb_faster_rows"],
            "rqtb_slower_rows": st["rqtb_slower_rows"],
        }
    stage_details = {}
    for stage in sorted(by_stage):
        stage_details[str(stage)] = predict_rows(calib, by_stage[stage])

    # Profile partition stability only. Neither half participates in model fitting;
    # the cycle coefficients are calibrated solely on sample0/window0 RTL rows.
    cal_rows = [r for r in all_rows if int(r["sample_id"]) < 50]
    hold_rows = [r for r in all_rows if int(r["sample_id"]) >= 50]
    split = {
        "profile_samples_0_49": predict_rows(calib, cal_rows),
        "profile_samples_50_99": predict_rows(calib, hold_rows),
    }

    report = {
        "schema": "h67_rqtb_fixed2s_multisample_rtl_calibrated_model_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_level": "[prof]+[rtl校准模型]",
        "status": "PASS",
        "scope": (
            "H67 ep30 fullres T450 ordered profile100; cycle model calibrated on "
            "sample0/window0 Fixed2S/RQTB2S RTL 138 head-rows; not multi-sample RTL"
        ),
        "calibration": calib,
        "global": global_stats,
        "sample_speedup_distribution": dist(sample_speedups),
        "samples_detail": sample_details,
        "stages": stage_details,
        "profile_partition_stability": split,
        "source_receipts": {
            "profile": receipt(args.profile),
            "rtl_report": receipt(args.rtl_report),
        },
        "claim_boundary": [
            "Multi-sample speedups are model estimates, not Icarus/Verilator cycles",
            "Calibration residual is small on 138 RTL rows but deployment uses predicted exp",
            "Samples 50-99 are a profile partition, not a training/calibration holdout",
            "The exact aggregate speedup match on the calibration rows follows from intercept-fitted least squares and is not independent validation",
            "Do not write full-encoder or ASIC energy claims from this report",
            "Primary paper RTL cycle claim remains Fixed2S->RQTB2S 1.185x on sample0/window0",
        ],
        "admission_hint": {
            "profile_samples_50_99_speedup": split["profile_samples_50_99"]["speedup"],
            "global_speedup": global_stats["speedup"],
            "global_slot_reduction": global_stats["slot_reduction"],
            "min_sample_speedup": dist(sample_speedups)["min"],
            "p95_sample_speedup": dist(sample_speedups)["p95"],
        },
    }

    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    g = global_stats
    sdist = report["sample_speedup_distribution"]
    md = f"""# Motion Fixed2S vs RQTB2S 多样本周期模型（RTL 校准）

## 结论

- 状态：**PASS**；证据等级：**`[prof]+[rtl校准模型]`**（不是多样本真实 RTL，不是 SAIF/PPA）。
- 校准集：sample0/window0 共 {calib['n_rows']} 个 head-row 的 Fixed2S/RQTB2S 实测周期；部署路径（先预测 exp 再预测周期）在校准集上 speedup true={calib['fit_metrics']['deployment_path']['speedup_true']:.4f} / model={calib['fit_metrics']['deployment_path']['speedup_model']:.4f}。
- 全量 100 sample / {g['n_rows']} head-row 模型：Fixed2S→RQTB2S **{g['speedup']:.4f}×**，slot 减少 **{g['slot_reduction']:.2%}**。
- 逐 sample 加速 mean/p95/p99/min/max = {sdist['mean']:.4f}/{sdist['p95']:.4f}/{sdist['p99']:.4f}/{sdist['min']:.4f}/{sdist['max']:.4f}。
- profile 后半 sample50–99 模型加速 **{split['profile_samples_50_99']['speedup']:.4f}×**（前半 sample0–49 为 {split['profile_samples_0_49']['speedup']:.4f}×）。两半都未参与周期模型拟合，该划分只检查 workload 分区稳定性，不是训练留出验证。

## 校准质量

| 模型 | R² / MAE | 备注 |
|---|---|---|
| fixed exp | R²={calib['fit_metrics']['fixed_exp']['r2']:.6f}, MAE={calib['fit_metrics']['fixed_exp']['mae']:.3f} | active,equal → exp |
| rqtb exp | R²={calib['fit_metrics']['rqtb_exp']['r2']:.6f}, MAE={calib['fit_metrics']['rqtb_exp']['mae']:.3f} | |
| fixed cycle (oracle exp) | R²={calib['fit_metrics']['fixed_cycle_oracle_exp']['r2']:.6f}, MAE={calib['fit_metrics']['fixed_cycle_oracle_exp']['mae']:.3f} | |
| rqtb cycle (oracle exp) | R²={calib['fit_metrics']['rqtb_cycle_oracle_exp']['r2']:.6f}, MAE={calib['fit_metrics']['rqtb_cycle_oracle_exp']['mae']:.3f} | |
| deployment fixed MAE | {calib['fit_metrics']['deployment_path']['fixed_mae']:.3f} | 用预测 exp |
| deployment rqtb MAE | {calib['fit_metrics']['deployment_path']['rqtb_mae']:.3f} | |

## 全局账本

| 指标 | Fixed2S | RQTB2S | 变化 |
|---|---:|---:|---:|
| 模型周期 | {g['fixed_cycles_total']:.0f} | {g['rqtb_cycles_total']:.0f} | {g['speedup']:.4f}× |
| slot | {g['fixed_slots_total']:.0f} | {g['rqtb_slots_total']:.0f} | -{g['slot_reduction']*100:.2f}% |
| 更快行 / 更慢行 | — | {g['rqtb_faster_rows']} / {g['rqtb_slower_rows']} | |

## 分 stage

| Stage | rows | speedup | slot reduction |
|---|---:|---:|---:|
"""
    for stage, st in stage_details.items():
        md += f"| {stage} | {st['n_rows']} | {st['speedup']:.4f}× | {st['slot_reduction']*100:.2f}% |\n"

    md += f"""

## 论文口径

- **可写：** 在 RTL 校准的周期模型下，100-sample fullres T450 上 Fixed2S→RQTB2S 的模型加速约 {g['speedup']:.2f}×，逐 sample 最差约 {sdist['min']:.2f}×，profile 后半集合约 {split['profile_samples_50_99']['speedup']:.2f}×；slot 模型减少约 {g['slot_reduction']*100:.1f}%。
- **不可写：** 把 sample50–99 称为模型 held-out，或把校准集总加速完全相等当作独立健康证据。
- **必须保留主 RTL 数字：** sample0/window0 真实仿真 Fixed2S→RQTB2S = 1.185×。
- **不可写：** 多样本真实 RTL 周期、full encoder 加速、ASIC 功耗/EDP。

## 源

- profile: `{args.profile}`
- rtl calibration: `{args.rtl_report}`
- machine report: `report.json`
"""
    (args.out / "report.md").write_text(md, encoding="utf-8")
    print(json.dumps({"status": "PASS", "out": str(args.out), "speedup": g["speedup"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
