"""Summarize hardware-facing stats for the NTS DATE mainline discussion.

The script is intentionally read-only with respect to experiment artifacts. It
collects already-generated valid825 spike profiles, configs, and logs into a
compact markdown/JSON report so NTS07/09 and NTS11 can be compared on the same
hardware audit axes.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "neuron_experiments/H9_bipolar_self_attention/results/hardware_stats_nts11_mainline"
TOTAL_ENCODER_BLOCKS = 12


@dataclass(frozen=True)
class Case:
    name: str
    epoch: int | None
    run_dir: Path | None
    config: Path | None
    profile: Path
    notes: str = ""


CASES = [
    Case(
        name="NB0_ep59",
        epoch=59,
        run_dir=None,
        config=None,
        profile=REPO_ROOT / "results_inference/nb0_baseline_epoch59_valid825_fixed_eval_20260601_140852/spike_profile.json",
        notes="baseline",
    ),
    Case(
        name="NTS07b_ep29",
        epoch=29,
        run_dir=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/results/"
        / "nts07b_hw_h60_ffn_update0_act0_s1224_steps1224_auto_full_bs6_20260608_042113_setsid",
        config=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/configs/"
        / "nts07b_hw_h60_ffn_update0_act0_s1224_steps1224_auto_full_20260608_042113.yml",
        profile=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/results/"
        / "nts07b_hw_h60_ffn_update0_act0_s1224_steps1224_auto_full_bs6_20260608_042113_setsid/"
        / "standard_valid825/epoch29/spike_profile.json",
        notes="S2-only H60",
    ),
    Case(
        name="NTS09e_ep29",
        epoch=29,
        run_dir=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/results/"
        / "nts09e_hw_h60_freeze1224_s1224_steps1224_auto_full_bs6_20260610_001833_setsid",
        config=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/configs/"
        / "nts09e_hw_h60_freeze1224_s1224_steps1224_auto_full_20260610_001833.yml",
        profile=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/results/"
        / "nts09e_hw_h60_freeze1224_s1224_steps1224_auto_full_bs6_20260610_001833_setsid/"
        / "standard_valid825/epoch29/spike_profile.json",
        notes="S2-only H60, qk freeze1224",
    ),
    Case(
        name="NTS11bd_ep19",
        epoch=19,
        run_dir=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/results/"
        / "nts11bd_u12_ds_w720_fastlr_full30_20260613_223042_bs8_20260613_223042_setsid",
        config=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/configs/"
        / "nts11bd_u12_ds_w720_fastlr_full30_20260613_223042.yml",
        profile=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/results/"
        / "nts11bd_u12_ds_w720_fastlr_full30_20260613_223042_bs8_20260613_223042_setsid/"
        / "standard_valid825/epoch19/spike_profile.json",
        notes="all-12 H60, bd best",
    ),
    Case(
        name="NTS11bj_ep2",
        epoch=2,
        run_dir=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/results/"
        / "nts11bj_u12_ds_w720_stdlr_ftbd19_ft5_bs8_20260614_233224_setsid",
        config=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
        / "nts11bj_u12_ds_w720_stdlr_ftbd19_ft5.yml",
        profile=REPO_ROOT
        / "neuron_experiments/H9_bipolar_self_attention/results/"
        / "nts11bj_u12_ds_w720_stdlr_ftbd19_ft5_bs8_20260614_233224_setsid/"
        / "standard_valid825/epoch2/spike_profile.json",
        notes="all-12 H60, bj fine-tune known checkpoint",
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def metric_float(profile: dict[str, Any], key: str) -> float:
    value = profile.get("metrics", {}).get(key, float("nan"))
    return float(value)


def read_log(run_dir: Path | None, epoch: int | None) -> str:
    if run_dir is None:
        return ""
    chunks: list[str] = []
    train_log = run_dir / "train.log"
    if train_log.exists():
        chunks.append(train_log.read_text(encoding="utf-8", errors="replace"))
    if epoch is not None:
        eval_log = run_dir / "standard_valid825" / f"epoch{epoch}" / "eval.log"
        if eval_log.exists():
            chunks.append(eval_log.read_text(encoding="utf-8", errors="replace"))
        profile_glob = (
            REPO_ROOT
            / "neuron_experiments/H9_bipolar_self_attention/results"
        ).glob(f"profile_{run_dir.name}_checkpoint_epoch{epoch}_valid825/profile.log")
        for profile_log in profile_glob:
            chunks.append(profile_log.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def parse_first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def parse_overlay_audit(text: str) -> dict[str, int | None]:
    matches = re.findall(r"checkpoint_overlay_keys=(\d+), missing=(\d+), unexpected=(\d+)", text)
    if not matches:
        return {"checkpoint_overlay_keys": None, "missing": None, "unexpected": None}
    keys, missing, unexpected = matches[-1]
    return {
        "checkpoint_overlay_keys": int(keys),
        "missing": int(missing),
        "unexpected": int(unexpected),
    }


def parse_summary_dict(text: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for marker in ("[H9] neuron summary after install:", "[H9] ATLIFTernaryPSN summary:"):
        for line in text.splitlines():
            if marker in line:
                payload = line.split(marker, 1)[1].strip()
                try:
                    summary.update(ast.literal_eval(payload))
                except (SyntaxError, ValueError):
                    pass
    return summary


def stage_from_name(name: str) -> str:
    match = re.search(r"layers\.(\d+)", name)
    if match:
        return f"S{match.group(1)}"
    if ".decoders." in name:
        return "decoder"
    if ".resblocks." in name:
        return "resblock"
    return "other"


def category_from_name(name: str) -> str:
    if ".attn.sn_q" in name:
        return "q_event"
    if ".attn.sn_k" in name:
        return "k_event"
    if ".attn.sn2_q" in name:
        return "q2_event"
    if ".attn.attn_sn" in name:
        return "attn_output_event"
    if ".attn.proj_sn" in name:
        return "attn_proj_event"
    if ".mlp.sn" in name:
        return "mlp_event"
    if ".downsample.sn" in name:
        return "downsample_event"
    if ".decoders." in name:
        return "decoder_event"
    if ".resblocks." in name:
        return "resblock_event"
    return "other_event"


def add_bucket(bucket: dict[str, dict[str, float]], key: str, spikes: float, elements: float) -> None:
    row = bucket.setdefault(key, {"spikes": 0.0, "elements": 0.0})
    row["spikes"] += spikes
    row["elements"] += elements


def summarize_layers(profile: dict[str, Any]) -> dict[str, Any]:
    layers = profile.get("layer_firing_rates", {}) or {}
    by_stage: dict[str, dict[str, float]] = {}
    by_category: dict[str, dict[str, float]] = {}
    zero_layers: list[str] = []
    top_layers: list[dict[str, Any]] = []

    for name, row in layers.items():
        spikes = float(row.get("spikes", 0.0) or 0.0)
        elements = float(row.get("elements", 0.0) or 0.0)
        firing = float(row.get("firing_rate", 0.0) or 0.0)
        add_bucket(by_stage, stage_from_name(name), spikes, elements)
        add_bucket(by_category, category_from_name(name), spikes, elements)
        if elements > 0 and spikes == 0:
            zero_layers.append(name)
        top_layers.append({"name": name, "spikes": spikes, "firing": firing})

    def finalize(bucket: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key, row in sorted(bucket.items()):
            elements = row["elements"]
            out[key] = {
                "spikes": row["spikes"],
                "spikes_g": row["spikes"] / 1e9,
                "elements": elements,
                "firing_rate": row["spikes"] / elements if elements else 0.0,
            }
        return out

    return {
        "profiled_layers": int(profile.get("profiled_layers", len(layers)) or len(layers)),
        "zero_firing_layers": zero_layers,
        "zero_firing_layer_count": len(zero_layers),
        "top_spike_layers": sorted(top_layers, key=lambda item: item["spikes"], reverse=True)[:10],
        "by_stage": finalize(by_stage),
        "by_category": finalize(by_category),
    }


def summarize_config(config: dict[str, Any]) -> dict[str, Any]:
    bsa = config.get("bsa_attention", {}) or {}
    atlif = config.get("atlif_ternary_psn", {}) or {}
    target_blocks = [str(item) for item in bsa.get("target_blocks", []) or []]
    h60_blocks = len(target_blocks)
    target_groups = atlif.get("target_groups", []) or []
    path_selection = [group.get("path_selection") for group in target_groups if isinstance(group, dict)]
    explicit_group_paths = sum(len(group.get("paths", []) or []) for group in target_groups if isinstance(group, dict))
    explicit_ternary_group_paths = sum(
        len(group.get("paths", []) or [])
        for group in target_groups
        if isinstance(group, dict) and group.get("output_mode") == "ternary"
    )
    explicit_binary_group_paths = sum(
        len(group.get("paths", []) or [])
        for group in target_groups
        if isinstance(group, dict) and group.get("output_mode") == "binary"
    )
    qk_ternary_modules_est = 2 * h60_blocks if atlif.get("target") == "qk" else 0
    return {
        "bsa_enabled": bool(bsa.get("enabled", False)),
        "mode": bsa.get("mode"),
        "target_blocks": target_blocks,
        "h60_blocks": h60_blocks,
        "native_attention_blocks_est": max(TOTAL_ENCODER_BLOCKS - h60_blocks, 0),
        "full_encoder_h60": h60_blocks == TOTAL_ENCODER_BLOCKS,
        "value_mode": bsa.get("value_mode"),
        "k_magnitude_alpha": float(bsa.get("k_magnitude_alpha", 0.0) or 0.0),
        "target_rate": bsa.get("target_rate"),
        "mismatch_penalty": float(bsa.get("mismatch_penalty", 0.0) or 0.0),
        "single_active_penalty": float(bsa.get("single_active_penalty", 0.0) or 0.0),
        "bipolar_mu": float(bsa.get("bipolar_mu", 0.0) or 0.0),
        "atlif_enabled": bool(atlif.get("enabled", False)),
        "atlif_target": atlif.get("target"),
        "atlif_output_mode": atlif.get("output_mode"),
        "atlif_threshold_mode": atlif.get("threshold_mode"),
        "atlif_target_rate": atlif.get("target_rate"),
        "threshold_freeze_after_step": atlif.get("threshold_freeze_after_step"),
        "target_group_count": len(target_groups),
        "explicit_target_group_paths": explicit_group_paths,
        "explicit_ternary_group_paths": explicit_ternary_group_paths,
        "explicit_binary_group_paths": explicit_binary_group_paths,
        "ternary_modules_est_from_config": qk_ternary_modules_est + explicit_ternary_group_paths,
        "has_all_non_qk_path_selection": "all_non_qk" in path_selection,
    }


def fill_missing_atlif_counts(row: dict[str, Any]) -> None:
    summary = row["atlif_summary"]
    cfg = row["config"]
    audit = row["install_audit"]
    total = audit.get("atlif_modules")
    if summary.get("symmetric_bsa_tsn_modules") is None:
        inferred_ternary = cfg.get("ternary_modules_est_from_config", 0)
        if inferred_ternary:
            summary["symmetric_bsa_tsn_modules"] = inferred_ternary
    if summary.get("official_atlif_modules") is None and total is not None:
        ternary = int(summary.get("symmetric_bsa_tsn_modules", 0) or 0)
        summary["official_atlif_modules"] = max(int(total) - ternary, 0)


def profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    dense = float(profile.get("dense_flops", 0.0) or 0.0)
    effective = float(profile.get("effective_flops", 0.0) or 0.0)
    total_spikes = float(profile.get("total_spikes", 0.0) or 0.0)
    return {
        "AEE": metric_float(profile, "AEE"),
        "AAE": metric_float(profile, "AAE"),
        "PE1": metric_float(profile, "AEE_PE1"),
        "PE2": metric_float(profile, "AEE_PE2"),
        "outlier": metric_float(profile, "AEE_outliers"),
        "total_spikes": total_spikes,
        "total_spikes_g": total_spikes / 1e9,
        "global_firing_rate": float(profile.get("global_firing_rate", 0.0) or 0.0),
        "dense_flops_g": dense / 1e9,
        "effective_flops_g": effective / 1e9,
        "sparsity_ratio": float(profile.get("sparsity_ratio", 0.0) or 0.0),
        "synops_mac_g": float(profile.get("synops_mac", 0.0) or 0.0) / 1e9,
        "synops_logic_g": float(profile.get("synops_logic", 0.0) or 0.0) / 1e9,
        "energy_uj": float(profile.get("energy_uj", 0.0) or 0.0),
        "samples": int(profile.get("samples", 0) or 0),
    }


def pct_delta(value: float, baseline: float) -> float:
    if not baseline or math.isnan(value) or math.isnan(baseline):
        return float("nan")
    return (value / baseline - 1.0) * 100.0


def collect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        profile = load_json(case.profile)
        config = load_yaml(case.config)
        log_text = read_log(case.run_dir, case.epoch)
        install_summary = parse_summary_dict(log_text)
        row = {
            "name": case.name,
            "epoch": case.epoch,
            "notes": case.notes,
            "paths": {
                "profile": str(case.profile.relative_to(REPO_ROOT)),
                "config": str(case.config.relative_to(REPO_ROOT)) if case.config else None,
                "run_dir": str(case.run_dir.relative_to(REPO_ROOT)) if case.run_dir else None,
            },
            "metrics": profile_summary(profile),
            "config": summarize_config(config),
            "install_audit": {
                "atlif_modules": parse_first_int(r"eval installed ATLIFTernaryPSN: (\d+) modules", log_text)
                or parse_first_int(r"installed ATLIFTernaryPSN: (\d+) modules", log_text),
                "shiftmax_attention_modules": parse_first_int(r"eval installed Shiftmax attention: (\d+) modules", log_text)
                or parse_first_int(r"installed Shiftmax attention: (\d+) modules", log_text),
                **parse_overlay_audit(log_text),
            },
            "atlif_summary": install_summary,
            "layers": summarize_layers(profile),
        }
        fill_missing_atlif_counts(row)
        rows.append(row)
    baseline = rows[0]["metrics"]
    for row in rows:
        metrics = row["metrics"]
        row["vs_baseline"] = {
            "AEE_pct": pct_delta(metrics["AEE"], baseline["AEE"]),
            "AAE_pct": pct_delta(metrics["AAE"], baseline["AAE"]),
            "spikes_pct": pct_delta(metrics["total_spikes"], baseline["total_spikes"]),
            "energy_pct": pct_delta(metrics["energy_uj"], baseline["energy_uj"]),
        }
    return rows


def fmt_float(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    if value is None or math.isnan(float(value)):
        return "n/a"
    return f"{float(value):+.{digits}f}%"


def fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    lines.append("# NTS 硬件统计汇总")
    lines.append("")
    lines.append("本报告由已落盘的 valid825 `spike_profile.json`、配置文件、训练日志和评估日志汇总生成。")
    lines.append("")
    lines.append("## 主指标")
    lines.append("")
    lines.append(
        "| 方案 | AEE | AAE | total_spikes | 相对 NB0 spikes | energy_uj | 相对 NB0 energy | firing | samples |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        m = row["metrics"]
        v = row["vs_baseline"]
        lines.append(
            f"| {row['name']} | {m['AEE']:.4f} | {m['AAE']:.4f} | "
            f"{m['total_spikes_g']:.4f}G | {fmt_pct(v['spikes_pct'])} | "
            f"{m['energy_uj']:.2f} | {fmt_pct(v['energy_pct'])} | "
            f"{m['global_firing_rate'] * 100:.4f}% | {m['samples']} |"
        )
    lines.append("")
    lines.append("## 覆盖范围与混合数据流审计")
    lines.append("")
    lines.append(
        "| 方案 | H60 blocks | 估计原生 attention | 全 encoder H60 | Shiftmax 模块 | ATLIF 模块 | ternary ATLIF | binary ATLIF | Kmag | target-rate | all-non-QK wrapper | overlay 加载 |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|")
    for row in rows:
        cfg = row["config"]
        audit = row["install_audit"]
        atlif = row["atlif_summary"]
        overlay = audit.get("checkpoint_overlay_keys")
        missing = audit.get("missing")
        unexpected = audit.get("unexpected")
        overlay_text = "n/a" if overlay is None else f"{overlay}/{missing}/{unexpected}"
        lines.append(
            f"| {row['name']} | {cfg['h60_blocks']} | {cfg['native_attention_blocks_est']} | "
            f"{fmt_bool(cfg['full_encoder_h60'])} | {audit.get('shiftmax_attention_modules') or 0} | "
            f"{audit.get('atlif_modules') or 0} | {atlif.get('symmetric_bsa_tsn_modules', 0)} | "
            f"{atlif.get('official_atlif_modules', 0)} | {cfg['k_magnitude_alpha']:.3g} | "
            f"{cfg['target_rate']} | {fmt_bool(cfg['has_all_non_qk_path_selection'])} | {overlay_text} |"
        )
    lines.append("")
    lines.append("overlay 加载格式为 `checkpoint_overlay_keys/missing/unexpected`；baseline 没有 overlay 审计。")
    lines.append("")
    lines.append("## ATLIF 活性快照")
    lines.append("")
    lines.append(
        "| 方案 | threshold_mean | threshold_max | ternary_activity | binary_activity | pos/neg ratio | zero pos modules | zero neg modules | frozen |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        a = row["atlif_summary"]
        lines.append(
            f"| {row['name']} | {fmt_float(a.get('threshold_mean', float('nan')), 6)} | "
            f"{fmt_float(a.get('threshold_max', float('nan')), 6)} | "
            f"{fmt_float(a.get('ternary_activity_mean', float('nan')), 6)} | "
            f"{fmt_float(a.get('binary_activity_mean', float('nan')), 6)} | "
            f"{fmt_float(a.get('ternary_pos_neg_ratio', float('nan')), 4)} | "
            f"{a.get('ternary_zero_pos_modules', 'n/a')} | {a.get('ternary_zero_neg_modules', 'n/a')} | "
            f"{a.get('threshold_updates_frozen', 'n/a')} |"
        )
    lines.append("")
    lines.append("## 按层类型统计 spikes")
    lines.append("")
    category_keys = [
        "q_event",
        "k_event",
        "q2_event",
        "attn_output_event",
        "attn_proj_event",
        "mlp_event",
        "downsample_event",
        "decoder_event",
        "resblock_event",
    ]
    lines.append("| 方案 | " + " | ".join(category_keys) + " | zero firing layers | profiled layers |")
    lines.append("|---|" + "|".join(["---:"] * (len(category_keys) + 2)) + "|")
    for row in rows:
        by_cat = row["layers"]["by_category"]
        values = [f"{by_cat.get(key, {}).get('spikes_g', 0.0):.3f}G" for key in category_keys]
        lines.append(
            f"| {row['name']} | "
            + " | ".join(values)
            + f" | {row['layers']['zero_firing_layer_count']} | {row['layers']['profiled_layers']} |"
        )
    lines.append("")
    lines.append("## 按 stage 统计 spikes")
    lines.append("")
    stage_keys = ["S0", "S1", "S2", "S3", "decoder", "resblock", "other"]
    lines.append("| 方案 | " + " | ".join(stage_keys) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(stage_keys)) + "|")
    for row in rows:
        by_stage = row["layers"]["by_stage"]
        values = [f"{by_stage.get(key, {}).get('spikes_g', 0.0):.3f}G" for key in stage_keys]
        lines.append(f"| {row['name']} | " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## spike 热点层")
    for row in rows:
        lines.append("")
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append("| rank | 层名 | spikes | firing |")
        lines.append("|---:|---|---:|---:|")
        for idx, layer in enumerate(row["layers"]["top_spike_layers"][:8], 1):
            lines.append(
                f"| {idx} | `{layer['name']}` | {layer['spikes'] / 1e9:.4f}G | "
                f"{layer['firing'] * 100:.4f}% |"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect()
    json_path = OUT_DIR / "hardware_stats_summary.json"
    md_path = OUT_DIR / "hardware_stats_summary.md"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(rows, md_path)
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
