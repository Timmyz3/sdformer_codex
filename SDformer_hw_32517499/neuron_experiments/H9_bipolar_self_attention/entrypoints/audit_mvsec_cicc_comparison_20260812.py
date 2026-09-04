#!/usr/bin/env python3
"""Fail-closed audit and DATE-facing table for direct-MVSEC experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RESULTS = EXP / "results"
COMPARISON = RESULTS / "mvsec_cicc_nb0_h67_local5_comparison_20260811.json"
OUTPUT = RESULTS / "mvsec_cicc_nb0_h67_local5_audit_20260812.json"
OUTPUT_MD = RESULTS / "mvsec_cicc_nb0_h67_local5_audit_20260812.md"
MANIFEST = EXP / "manifests/mvsec_cicc_dt1_v1.json"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
HW_BOARD = REPO / (
    "hw_autoresearch_nts07/docs/"
    "312_DATE贡献冻结_评判指标与验证链路看板_20260810.md"
)
SEQUENCES = {
    "outdoor_day1": 2755,
    "indoor_flying1": 1883,
    "indoor_flying2": 1885,
    "indoor_flying3": 1885,
}
ROUTES = {
    "nb0": {
        "config": EXP / "configs/generated/mvsec_cicc_nb0_w8_seed0.yml",
        "train": RESULTS / "mvsec_cicc_nb0_w8_seed0_v4_20260811",
        "fixed800": RESULTS / "mvsec_cicc_nb0_w8_seed0_v4_fixed800_20260811/mvsec_summary.json",
        "full": RESULTS / "mvsec_cicc_nb0_w8_seed0_v4_full_20260811/mvsec_summary.json",
    },
    "h67": {
        "config": EXP / "configs/generated/mvsec_cicc_h67_motion_w8_seed0.yml",
        "train": RESULTS / "mvsec_cicc_h67_motion_w8_seed0_v4_20260811",
        "smoke": RESULTS / "mvsec_cicc_h67_motion_w8_seed0_v4_load_smoke_20260811",
        "fixed800": RESULTS / "mvsec_cicc_h67_motion_w8_seed0_v4_fixed800_20260811/mvsec_summary.json",
        "full": RESULTS / "mvsec_cicc_h67_motion_w8_seed0_v4_full_20260811/mvsec_summary.json",
    },
    "local5": {
        "config": EXP / "configs/generated/mvsec_cicc_local5_w8_seed0.yml",
        "train": RESULTS / "mvsec_cicc_local5_w8_seed0_v4_20260811",
        "smoke": RESULTS / "mvsec_cicc_local5_w8_seed0_v4_load_smoke_20260811",
        "fixed800": RESULTS / "mvsec_cicc_local5_w8_seed0_v4_fixed800_20260811/mvsec_summary.json",
        "full": RESULTS / "mvsec_cicc_local5_w8_seed0_v4_full_20260811/mvsec_summary.json",
    },
}
EPOCH_RE = re.compile(r"^Epoch (\d+)\s*$")
VALID_RE = re.compile(r"Epoch loss \(Validation\): ([0-9.eE+-]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def validation_losses(path: Path) -> dict[int, float]:
    current_epoch: int | None = None
    losses: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = EPOCH_RE.match(line.strip())
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        valid_match = VALID_RE.search(line)
        if valid_match and current_epoch is not None:
            losses[current_epoch] = float(valid_match.group(1))
    if not losses:
        raise RuntimeError(f"no validation losses found: {path}")
    return losses


def load_contract(route: str, nb0_checkpoint: Path) -> dict[str, object]:
    smoke = ROUTES[route]["smoke"]
    text = (smoke / "train.log").read_text(encoding="utf-8", errors="replace")
    provenance = json.loads((ROUTES[route]["train"] / "launch_provenance.json").read_text())
    checks = {
        "ATLIF_105": "installed ATLIFTernaryPSN before load: 105 modules" in text,
        "Shiftmax_12": "installed Shiftmax attention before load: 12 modules" in text,
        "overlay_0_missing210_unexpected0": (
            "checkpoint_overlay_keys=0, missing=210, unexpected=0" in text
        ),
        "smoke_exit0": "[mvsec-cicc-train] exit_code=0" in text,
        "same_NB0_initialization": Path(provenance["prev_runid"]).resolve() == nb0_checkpoint.resolve(),
        "not_resume": provenance.get("resume") is False,
        "manifest_sha": (provenance.get("manifest") or {}).get("sha256") == sha256(MANIFEST),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{route} load contract failed: {checks}")
    return {"checks": checks, "smoke_log": binding(smoke / "train.log"), "provenance": binding(ROUTES[route]["train"] / "launch_provenance.json")}


def audit_summary(route: str, protocol: str, checkpoint: Path) -> dict[str, object]:
    path = ROUTES[route][protocol]
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("sequences") or []
    row_by_sequence = {row["sequence"]: row for row in rows}
    expected_samples = {name: 800 for name in SEQUENCES} if protocol == "fixed800" else SEQUENCES
    checks = {
        "schema": raw.get("schema") == "mvsec_evaluation_summary_v1",
        "protocol": raw.get("protocol") == ("fixed800" if protocol == "fixed800" else "full_sequence"),
        "checkpoint_path": Path(raw["checkpoint"]).resolve() == checkpoint.resolve(),
        "checkpoint_sha": raw.get("checkpoint_sha256") == sha256(checkpoint),
        "config_path": Path(raw["config"]).resolve() == ROUTES[route]["config"].resolve(),
        "config_sha": raw.get("source_config_sha256") == sha256(ROUTES[route]["config"]),
        "four_sequences": set(row_by_sequence) == set(SEQUENCES),
        "no_skipped": raw.get("skipped") == [],
        "sample_counts": all(
            int(row_by_sequence[name]["samples"]) == count
            for name, count in expected_samples.items()
        ),
        "finite_metrics": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in ("AEE", "gt_fl_percent", "spikes_g", "energy_uj", "valid_pixels")
        ),
    }
    if protocol == "fixed800":
        checks["manifest_path"] = Path(raw["fixed800_manifest"]).resolve() == MANIFEST.resolve()
        checks["manifest_sha"] = raw.get("fixed800_manifest_sha256") == sha256(MANIFEST)
    else:
        checks["no_fixed800_manifest"] = raw.get("fixed800_manifest") is None

    mean_aee = sum(float(row["AEE"]) for row in rows) / len(rows)
    valid_pixels = sum(int(row["valid_pixels"]) for row in rows)
    weighted_aee = sum(float(row["weighted_aee"]) * int(row["valid_pixels"]) for row in rows) / valid_pixels
    macro_fl = sum(float(row["gt_fl_percent"]) for row in rows) / len(rows)
    total_spikes = sum(float(row["spikes_g"]) for row in rows)
    total_energy = sum(float(row["energy_uj"]) for row in rows)
    checks.update({
        "recomputed_mean_aee": close(mean_aee, raw["mean_aee"]),
        "recomputed_weighted_aee": close(weighted_aee, raw["valid_pixel_weighted_aee"]),
        "recomputed_valid_pixels": valid_pixels == int(raw["total_valid_pixels"]),
    })
    if not all(checks.values()):
        raise RuntimeError(f"{route}/{protocol} summary contract failed: {checks}")
    return {
        "summary": binding(path),
        "checks": checks,
        "mean_aee": mean_aee,
        "valid_pixel_weighted_aee": weighted_aee,
        "macro_gt_fl_percent": macro_fl,
        "total_spikes_g": total_spikes,
        "total_energy_uj": total_energy,
        "total_valid_pixels": valid_pixels,
        "per_sequence": rows,
    }


def run() -> dict[str, object]:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    if comparison.get("schema") != "mvsec_cicc_nb0_h67_local5_comparison_v1":
        raise RuntimeError("unexpected comparison schema")
    if Path(comparison["manifest"]).resolve() != MANIFEST.resolve():
        raise RuntimeError("comparison manifest mismatch")
    comparison_routes = {row["route"]: row for row in comparison["routes"]}
    if set(comparison_routes) != set(ROUTES):
        raise RuntimeError("comparison route set mismatch")

    checkpoints = {}
    rows = {}
    for route in ROUTES:
        best_path = ROUTES[route]["train"] / "best_checkpoint.json"
        best = json.loads(best_path.read_text(encoding="utf-8"))
        train_log = ROUTES[route]["train"] / "train.log"
        losses = validation_losses(train_log)
        recomputed_epoch = min(losses, key=lambda epoch: (losses[epoch], epoch))
        receipt_losses = {
            int(epoch): float(value)
            for epoch, value in (best.get("available_validation_losses") or {}).items()
        }
        if receipt_losses != losses:
            raise RuntimeError(f"{route} best-checkpoint validation-loss receipt drift")
        if (
            int(best["epoch"]) != recomputed_epoch
            or not close(float(best["validation_loss"]), losses[recomputed_epoch])
        ):
            raise RuntimeError(f"{route} best checkpoint is not validation-loss rank1")
        checkpoint = Path(best["checkpoint"])
        if not checkpoint.is_file() or checkpoint.name != f"checkpoint_epoch{best['epoch']}.pth":
            raise RuntimeError(f"{route} best checkpoint missing or mislabeled")
        if Path(comparison_routes[route]["checkpoint"]).resolve() != checkpoint.resolve():
            raise RuntimeError(f"{route} comparison checkpoint mismatch")
        fixed_raw = json.loads(ROUTES[route]["fixed800"].read_text(encoding="utf-8"))
        full_raw = json.loads(ROUTES[route]["full"].read_text(encoding="utf-8"))
        if comparison_routes[route]["fixed800"] != fixed_raw:
            raise RuntimeError(f"{route} embedded fixed800 summary mismatch")
        if comparison_routes[route]["full_sequence"] != full_raw:
            raise RuntimeError(f"{route} embedded full-sequence summary mismatch")
        checkpoints[route] = checkpoint
        rows[route] = {
            "best_checkpoint": binding(checkpoint),
            "best_checkpoint_receipt": binding(best_path),
            "train_log": binding(train_log),
            "best_epoch": int(best["epoch"]),
            "validation_loss": float(best["validation_loss"]),
            "validation_losses_recomputed": losses,
            "fixed800": audit_summary(route, "fixed800", checkpoint),
            "full_sequence": audit_summary(route, "full", checkpoint),
        }

    nb0_checkpoint = checkpoints["nb0"]
    rows["h67"]["load_contract"] = load_contract("h67", nb0_checkpoint)
    rows["local5"]["load_contract"] = load_contract("local5", nb0_checkpoint)

    for route in ("h67", "local5"):
        for protocol in ("fixed800", "full_sequence"):
            candidate = rows[route][protocol]
            reference = rows["nb0"][protocol]
            candidate["vs_NB0_pct_lower_is_better"] = {
                metric: improvement(reference[source], candidate[source])
                for metric, source in (
                    ("mean_aee", "mean_aee"),
                    ("valid_pixel_weighted_aee", "valid_pixel_weighted_aee"),
                    ("macro_gt_fl_percent", "macro_gt_fl_percent"),
                    ("total_spikes_g", "total_spikes_g"),
                    ("total_energy_uj", "total_energy_uj"),
                )
            }

    nb0_by_sequence = {
        row["sequence"]: row for row in rows["nb0"]["full_sequence"]["per_sequence"]
    }
    gates: dict[str, dict[str, bool]] = {}
    for route in ("h67", "local5"):
        full = rows[route]["full_sequence"]
        by_sequence = {row["sequence"]: row for row in full["per_sequence"]}
        route_gates = {
            "AEE_within_NB0_plus5pct": (
                full["mean_aee"]
                <= rows["nb0"]["full_sequence"]["mean_aee"] * 1.05
            ),
            "spikes_reduction_at_least20pct": (
                full["vs_NB0_pct_lower_is_better"]["total_spikes_g"] >= 20.0
            ),
            "all_sequence_AEE_improved": all(
                float(by_sequence[sequence]["AEE"])
                < float(nb0_by_sequence[sequence]["AEE"])
                for sequence in SEQUENCES
            ),
        }
        route_gates["qualifies"] = all(route_gates.values())
        gates[route] = route_gates
    if not gates["h67"]["qualifies"]:
        raise RuntimeError(f"H67 direct-MVSEC DATE gates failed: {gates['h67']}")
    qualified = [route for route in ("h67", "local5") if gates[route]["qualifies"]]
    selected = min(
        qualified,
        key=lambda route: rows[route]["full_sequence"]["mean_aee"],
    )
    return {
        "schema": "mvsec_cicc_nb0_h67_local5_audit_v1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "direct-MVSEC dt1 train, fixed800 and full-sequence evaluation; algorithm evidence only",
        "hardware_boundary": "does_not_replace_DSEC_fullres_H67_ep35_checkpoint_bound_component_RTL",
        "inputs": {"comparison": binding(COMPARISON), "manifest": binding(MANIFEST)},
        "routes": rows,
        "DATE_gates": gates,
        "algorithm_only_selection": {
            "criterion": "minimum full-sequence macro AEE among qualifying candidates",
            "qualified_candidates": qualified,
            "selected": selected,
            "does_not_change_DSEC_hardware_mainline": True,
        },
    }


def write_markdown(payload: dict[str, object]) -> None:
    lines = [
        "# Direct-MVSEC NB0/H67/Local5 Audit",
        "",
        "Status: **PASS**. H67/Local5 share the same NB0 initialization and frozen evaluation populations.",
        "",
        "| route | protocol | epoch | mean AEE | weighted AEE | GT Fl (%) | spikes (G) | energy (uJ) | AEE vs NB0 | spikes vs NB0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route in ("nb0", "h67", "local5"):
        row = payload["routes"][route]
        for protocol in ("fixed800", "full_sequence"):
            metrics = row[protocol]
            delta = metrics.get("vs_NB0_pct_lower_is_better")
            aee_delta = "-" if delta is None else f"{delta['mean_aee']:.2f}%"
            spike_delta = "-" if delta is None else f"{delta['total_spikes_g']:.2f}%"
            lines.append(
                f"| {route} | {protocol} | {row['best_epoch']} | {metrics['mean_aee']:.4f} | "
                f"{metrics['valid_pixel_weighted_aee']:.4f} | {metrics['macro_gt_fl_percent']:.4f} | "
                f"{metrics['total_spikes_g']:.4f} | {metrics['total_energy_uj']:.2f} | {aee_delta} | {spike_delta} |"
            )
    lines.extend([
        "",
        f"Algorithm-only MVSEC selection: `{payload['algorithm_only_selection']['selected']}`. "
        "This does not replace the DSEC H67 hardware anchor.",
        "",
        "The energy value is the existing activity proxy and is not ASIC power. MVSEC checkpoints do not inherit DSEC RTL provenance.",
    ])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_docs(payload: dict[str, object]) -> None:
    marker = "MVSEC_DIRECT_THREE_ROUTE_FINAL_AUDIT_20260812"
    routes = payload["routes"]
    selected = payload["algorithm_only_selection"]["selected"]
    for path in (REDESIGN, HW_BOARD):
        text = path.read_text(encoding="utf-8")
        if marker in text:
            continue
        lines = [
            "",
            f"<!-- {marker} -->",
            "",
            "### direct-MVSEC NB0/H67/Local5 最终审计",
            "",
            "- fail-closed 审计 PASS；三线 best checkpoint 均从原始训练日志重算 "
            "validation-loss rank-1，并绑定 checkpoint/config/manifest SHA。",
            "- full-sequence 同协议结果：",
            "",
            "| route | epoch | macro AEE | weighted AEE | Fl(%) | spikes(G) | energy proxy(uJ) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for route in ("nb0", "h67", "local5"):
            row = routes[route]
            metrics = row["full_sequence"]
            lines.append(
                f"| {route} | {row['best_epoch']} | {metrics['mean_aee']:.6f} | "
                f"{metrics['valid_pixel_weighted_aee']:.6f} | "
                f"{metrics['macro_gt_fl_percent']:.4f} | "
                f"{metrics['total_spikes_g']:.4f} | {metrics['total_energy_uj']:.2f} |"
            )
        lines.extend(
            [
                "",
                f"- MVSEC algorithm-only 合格候选中按 macro AEE 选中 `{selected}`。"
                "该结论不改变 DSEC H67 ep35 硬件主线，MVSEC checkpoint 不继承 DSEC RTL provenance。",
                f"- 机器审计：`{OUTPUT.relative_to(REPO)}`。",
                "",
            ]
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    while not COMPARISON.is_file():
        if not args.wait:
            raise FileNotFoundError(COMPARISON)
        print(f"WAIT {COMPARISON}", flush=True)
        time.sleep(args.poll_seconds)
    payload = run()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload)
    append_docs(payload)
    print(f"PASS direct-MVSEC audit: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
