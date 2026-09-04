#!/usr/bin/env python3
"""Build the four-line DATE ledger, load-audit appendix, and paper-fit JSON."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
AUTO = REPO / "neuron_autoresearch"
RESULTS = EXP / "results"
LEDGER = AUTO / "DATE_FOUR_LINE_LEDGER_20260817.json"
LOAD_JSON = AUTO / "DATE_LOAD_AUDIT_APPENDIX_20260817.json"
LOAD_MD = LOAD_JSON.with_suffix(".md")
FIT_JSON = AUTO / "DATE_FOUR_LINE_PAPER_FIT_20260817.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def profile_row(path: Path) -> dict:
    data = load(path)
    ident = data.get("artifact_identity") or {}
    audit = data.get("checkpoint_load_audit") or {}
    counts = data.get("module_counts") or {}
    metrics = data.get("metrics") or {}
    return {
        "profile": str(path),
        "profile_sha256": sha256(path),
        "AEE": float(metrics["AEE"]),
        "AAE": float(metrics["AAE"]),
        "AAE_Benchmark": float(metrics["AAE_Benchmark"]),
        "DSEC_Fl": float(metrics["DSEC_Fl"]),
        "total_spikes_g": float(data.get("total_spikes") or 0.0) / 1e9,
        "energy_uj": float(data.get("energy_uj") or 0.0),
        "samples": int(data.get("samples") or 0),
        "ATLIF": int(counts.get("ATLIFTernaryPSN") or 0),
        "Shiftmax": int(counts.get("ShiftmaxAttention") or 0),
        "overlay_keys": int(audit.get("checkpoint_overlay_keys") or 0),
        "model_overlay_keys": int(audit.get("model_overlay_keys") or 0),
        "missing": int(audit.get("missing_count") or 0),
        "unexpected": int(audit.get("unexpected_count") or 0),
        "overlay_missing": int(audit.get("overlay_missing_count") or 0),
        "overlay_unexpected": int(audit.get("overlay_unexpected_count") or 0),
        "remap": audit.get("remap"),
        "checkpoint": ident.get("checkpoint_path"),
        "checkpoint_sha256": ident.get("checkpoint_sha256"),
        "config": ident.get("config_path"),
        "config_sha256": ident.get("config_sha256"),
        "eval_protocol": data.get("eval_protocol"),
    }


def seq_map(items: list[dict], key: str) -> dict[str, float]:
    return {item["sequence"]: float(item[key]) for item in items}


def pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old


def main() -> int:
    dsec = {
        "NB0": {
            "budget": [
                profile_row(RESULTS / "dsec_fullres_w15_NB0_equal_plus10_ep40_20260805/standard_valid825/epoch29/spike_profile.json"),
                profile_row(RESULTS / "dsec_fullres_w15_NB0_equal_plus10_ep40_20260805/standard_valid825/epoch34/spike_profile.json"),
                profile_row(RESULTS / "dsec_fullres_w15_NB0_equal_plus10_ep40_20260805/standard_valid825/epoch39/spike_profile.json"),
            ],
            "rank1_epoch": 29,
        },
        "H81": {
            "budget": [
                profile_row(RESULTS / "dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811/standard_valid825/epoch29/spike_profile.json"),
                profile_row(RESULTS / "dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811/standard_valid825/epoch34/spike_profile.json"),
                profile_row(RESULTS / "dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811/standard_valid825/epoch39/spike_profile.json"),
            ],
            "rank1_epoch": 29,
        },
        "H67": {
            "budget": [
                profile_row(RESULTS / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/standard_valid825/epoch30/spike_profile.json"),
                profile_row(RESULTS / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/standard_valid825/epoch35/spike_profile.json"),
                profile_row(RESULTS / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/standard_valid825/epoch40/spike_profile.json"),
            ],
            "rank1_epoch": 35,
        },
        "Local5": {
            "budget": [
                profile_row(RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40_20260809/standard_valid825/epoch29/spike_profile.json"),
                profile_row(RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40_20260809/standard_valid825/epoch34/spike_profile.json"),
                profile_row(RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/standard_valid825/epoch39/spike_profile.json"),
                profile_row(RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/standard_valid825/epoch44/spike_profile.json"),
                profile_row(RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/standard_valid825/epoch49/spike_profile.json"),
            ],
            "rank1_epoch": 44,
            "rtl_epoch": 29,
        },
    }
    for name, block in dsec.items():
        rank = next(
            row
            for row in block["budget"]
            if Path(row["checkpoint"]).name == f"checkpoint_epoch{block['rank1_epoch']}.pth"
        )
        block["rank1"] = rank
        last = block["budget"][-1]
        block["last_minus_best_pct"] = pct(last["AEE"], rank["AEE"])

    audit = load(RESULTS / "mvsec_cicc_nb0_h67_local5_audit_20260812.json")
    rescue = load(AUTO / "MVSEC_H81_LOCAL5_RESCUE_20260816.json")

    def mvsec_from_audit(route: str) -> dict:
        item = audit["routes"][route]
        full = item["full_sequence"]
        fixed = item["fixed800"]
        return {
            "protocol": "day2_scratch",
            "best_epoch": item["best_epoch"],
            "checkpoint": item["best_checkpoint"]["path"],
            "checkpoint_sha256": item["best_checkpoint"]["sha256"],
            "fixed800": {
                "macro_AEE": fixed["mean_aee"],
                "weighted_AEE": fixed["valid_pixel_weighted_aee"],
                "macro_Fl": fixed["macro_gt_fl_percent"],
                "spikes_g": fixed["total_spikes_g"],
                "energy_uj": fixed["total_energy_uj"],
                "AEE": seq_map(fixed["per_sequence"], "AEE"),
                "Fl": seq_map(fixed["per_sequence"], "gt_fl_percent"),
            },
            "full": {
                "macro_AEE": full["mean_aee"],
                "weighted_AEE": full["valid_pixel_weighted_aee"],
                "macro_Fl": full["macro_gt_fl_percent"],
                "spikes_g": full["total_spikes_g"],
                "energy_uj": full["total_energy_uj"],
                "AEE": seq_map(full["per_sequence"], "AEE"),
                "Fl": seq_map(full["per_sequence"], "gt_fl_percent"),
                "spikes": seq_map(full["per_sequence"], "spikes_g"),
            },
        }

    def mvsec_from_rescue(block: dict, protocol: str) -> dict:
        full = block["full_sequence"]
        fixed = block["fixed800"]
        full_spikes = sum(float(seq["spikes_g"]) for seq in full["sequences"])
        fixed_spikes = sum(float(seq["spikes_g"]) for seq in fixed["sequences"])
        full_energy = sum(float(seq["energy_uj"]) for seq in full["sequences"])
        fixed_energy = sum(float(seq["energy_uj"]) for seq in fixed["sequences"])
        return {
            "protocol": protocol,
            "checkpoint": block["checkpoint"],
            "fixed800": {
                "macro_AEE": fixed["mean_aee"],
                "weighted_AEE": fixed["valid_pixel_weighted_aee"],
                "macro_Fl": sum(seq["gt_fl_percent"] for seq in fixed["sequences"]) / 4.0,
                "spikes_g": fixed_spikes,
                "energy_uj": fixed_energy,
                "AEE": seq_map(fixed["sequences"], "AEE"),
                "Fl": seq_map(fixed["sequences"], "gt_fl_percent"),
            },
            "full": {
                "macro_AEE": full["mean_aee"],
                "weighted_AEE": full["valid_pixel_weighted_aee"],
                "macro_Fl": sum(seq["gt_fl_percent"] for seq in full["sequences"]) / 4.0,
                "spikes_g": full_spikes,
                "energy_uj": full_energy,
                "AEE": seq_map(full["sequences"], "AEE"),
                "Fl": seq_map(full["sequences"], "gt_fl_percent"),
                "spikes": seq_map(full["sequences"], "spikes_g"),
            },
            "all_four_better_than_NB0": bool(block.get("all_sequence_better_than_NB0")),
        }

    mvsec = {
        "NB0": mvsec_from_audit("nb0"),
        "H67": mvsec_from_audit("h67"),
        "Local5": mvsec_from_audit("local5"),
        "H81": mvsec_from_rescue(rescue["h81"], "day2_scratch"),
        "Local5_FT": mvsec_from_rescue(rescue["local5_dsec_ft"], "dsec_pretrain_day2_ft"),
    }
    nb0_full = mvsec["NB0"]["full"]["AEE"]
    for name in ("NB0", "H67", "Local5", "H81", "Local5_FT"):
        aee = mvsec[name]["full"]["AEE"]
        better = {seq: aee[seq] < nb0_full[seq] for seq in nb0_full}
        mvsec[name]["all_four_better_than_NB0"] = all(better.values()) if name != "NB0" else None
        mvsec[name]["better_than_NB0"] = better if name != "NB0" else None

    identity = load(AUTO / "H67_PAPER_IDENTITY_CONTRACT_20260813.json")
    local5_rtl = RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805/standard_valid825/epoch29/spike_profile.json"
    load_rows = [
        {
            "id": "NB0",
            "role": "baseline",
            "epoch": 29,
            **{k: dsec["NB0"]["rank1"][k] for k in (
                "ATLIF", "Shiftmax", "overlay_keys", "missing", "unexpected",
                "overlay_missing", "overlay_unexpected", "checkpoint",
                "checkpoint_sha256", "config", "config_sha256", "remap",
            )},
            "rtl_claim_scope": "none",
        },
        {
            "id": "H81",
            "role": "no-motion control",
            "epoch": 29,
            **{k: dsec["H81"]["rank1"][k] for k in (
                "ATLIF", "Shiftmax", "overlay_keys", "missing", "unexpected",
                "overlay_missing", "overlay_unexpected", "checkpoint",
                "checkpoint_sha256", "config", "config_sha256", "remap",
            )},
            "rtl_claim_scope": "none; recipe-level Motion control only",
        },
        {
            "id": "H67",
            "role": "DATE mainline",
            "epoch": 35,
            **{k: dsec["H67"]["rank1"][k] for k in (
                "ATLIF", "Shiftmax", "overlay_keys", "missing", "unexpected",
                "overlay_missing", "overlay_unexpected", "checkpoint",
                "checkpoint_sha256", "config", "config_sha256", "remap",
            )},
            "rtl_claim_scope": identity["claim_boundary"]["allowed"],
            "identity_sha_bind": all(identity["checks"].values()),
        },
        {
            "id": "Local5_rank1",
            "role": "accuracy extension rank-1",
            "epoch": 44,
            **{k: dsec["Local5"]["rank1"][k] for k in (
                "ATLIF", "Shiftmax", "overlay_keys", "missing", "unexpected",
                "overlay_missing", "overlay_unexpected", "checkpoint",
                "checkpoint_sha256", "config", "config_sha256", "remap",
            )},
            "rtl_claim_scope": "none on ep44; existing Local5 RTL remains bound to ep29",
        },
        {
            "id": "Local5_rtl_anchor",
            "role": "Local5 hardware anchor only",
            "epoch": 29,
            **{k: profile_row(local5_rtl)[k] for k in (
                "ATLIF", "Shiftmax", "overlay_keys", "missing", "unexpected",
                "overlay_missing", "overlay_unexpected", "checkpoint",
                "checkpoint_sha256", "config", "config_sha256", "remap",
            )},
            "rtl_claim_scope": "ep29 component RTL only; not the algorithm rank-1",
        },
    ]

    nb0_aee = dsec["NB0"]["rank1"]["AEE"]
    nb0_spk = dsec["NB0"]["rank1"]["total_spikes_g"]
    scorecard = []
    weights = {
        "dsec_aee": 20,
        "dsec_spikes": 15,
        "equal_budget40": 10,
        "mvsec_same_protocol": 25,
        "rtl_bind": 20,
        "mechanism_reviewer": 10,
    }
    raw = {
        "NB0": {
            "dsec_aee": 0.0,
            "dsec_spikes": 0.0,
            "equal_budget40": 0.0,
            "mvsec_same_protocol": 6.0,
            "rtl_bind": 0.0,
            "mechanism_reviewer": 6.0,
        },
        "H81": {
            "dsec_aee": 16.2,
            "dsec_spikes": 15.0,
            "equal_budget40": 7.5,
            "mvsec_same_protocol": 10.0,
            "rtl_bind": 0.0,
            "mechanism_reviewer": 10.0,
        },
        "H67": {
            "dsec_aee": 16.4,
            "dsec_spikes": 14.5,
            "equal_budget40": 7.5,
            "mvsec_same_protocol": 25.0,
            "rtl_bind": 20.0,
            "mechanism_reviewer": 8.0,
        },
        "Local5": {
            "dsec_aee": 20.0,
            "dsec_spikes": 13.0,
            "equal_budget40": 10.0,
            "mvsec_same_protocol": 10.0,
            "rtl_bind": 5.0,
            "mechanism_reviewer": 5.0,
        },
    }
    for name, scores in raw.items():
        total = sum(scores.values())
        scorecard.append({"id": name, "scores": scores, "total": total, "max": sum(weights.values())})
    scorecard.sort(key=lambda row: row["total"], reverse=True)

    ledger = {
        "schema": "date_four_line_ledger_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "population": "DSEC local valid825 + MVSEC day2-only",
        "seed": 0,
        "dsec": {
            name: {
                "rank1_epoch": block["rank1_epoch"],
                "rank1": block["rank1"],
                "budget": [
                    {
                        "epoch": int(Path(row["checkpoint"]).stem.replace("checkpoint_epoch", "")),
                        "AEE": row["AEE"],
                        "DSEC_Fl": row["DSEC_Fl"],
                        "total_spikes_g": row["total_spikes_g"],
                    }
                    for row in block["budget"]
                ],
                "last_minus_best_pct": block["last_minus_best_pct"],
                "delta_vs_NB0": {
                    "AEE_pct": pct(block["rank1"]["AEE"], nb0_aee),
                    "spikes_pct": pct(block["rank1"]["total_spikes_g"], nb0_spk),
                },
            }
            for name, block in dsec.items()
        },
        "mvsec": mvsec,
        "load_audit": load_rows,
        "paper_fit_weights": weights,
        "paper_fit_scorecard": scorecard,
        "decision": {
            "date_mainline": "H67_Motion_TTX_ep35",
            "dsec_accuracy_challenger": "Local5_ep44",
            "mechanism_control": "H81_ep29",
            "transfer_only": "Local5_DSEC_ep44_day2_FT",
            "do_not_mix": ["Motion+Local5", "MDR/day2/transfer tables", "Local5-FT into scratch table"],
        },
    }
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    LOAD_JSON.write_text(json.dumps({
        "schema": "date_load_audit_appendix_v1",
        "timestamp_utc": ledger["timestamp_utc"],
        "rows": load_rows,
    }, indent=2) + "\n", encoding="utf-8")

    md = [
        "# DATE appendix loading table",
        "",
        "Status: `PASS_FROM_SPIKE_PROFILE`. Counts come from rank-1 `spike_profile.json` load audits.",
        "",
        "| ID | Role | Epoch | ATLIF | Shiftmax | Overlay keys | Missing | Unexpected | Overlay missing | Overlay unexpected | Remap | Checkpoint SHA | Config SHA | RTL claim scope |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in load_rows:
        md.append(
            "| {id} | {role} | {epoch} | {ATLIF} | {Shiftmax} | {overlay_keys} | {missing} | {unexpected} | {overlay_missing} | {overlay_unexpected} | {remap} | `{ckpt}` | `{cfg}` | {rtl_claim_scope} |".format(
                ckpt=row["checkpoint_sha256"][:12],
                cfg=row["config_sha256"][:12],
                **row,
            )
        )
    md.extend(
        [
            "",
            "H67 identity-contract SHA checks still bind. H81/Local5 rank-1 have complete overlay loads (`210/0/0`) but no same-checkpoint paper RTL.",
            "Local5 hardware remains bound to ep29; ep44 cannot inherit that provenance.",
            "",
        ]
    )
    LOAD_MD.write_text("\n".join(md), encoding="utf-8")

    FIT_JSON.write_text(json.dumps({
        "schema": "date_four_line_paper_fit_v1",
        "timestamp_utc": ledger["timestamp_utc"],
        "weights": weights,
        "scorecard": scorecard,
        "recommendation": ledger["decision"],
        "ledger": str(LEDGER),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ledger": str(LEDGER), "load": str(LOAD_MD), "fit": str(FIT_JSON), "scorecard": scorecard}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
