"""Generate paper-facing AAE audit configs for valuable DSEC checkpoints."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated/valuable_aae_audit_20260717"
RESULTS = EXP / "results"
MANIFEST = GEN / "manifest.json"


def checkpoint(run: str, epoch: int) -> Path:
    return RESULTS / run / f"checkpoint_epoch{epoch}.pth"


CANDIDATES = (
    {
        "id": "nb0_ep59",
        "label": "NB0 baseline",
        "group": "paper core",
        "epoch": 59,
        "source_config": REPO / "configs/generated/upstream_baseline_stride.yml",
        "checkpoint": REPO / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth",
        "expected_atlif": 0,
        "expected_shiftmax": 0,
    },
    {
        "id": "nts11bd_ep19",
        "label": "NTS11bd mixed ternary/binary",
        "group": "historical hardware control",
        "epoch": 19,
        "source_config": EXP / "configs/nts11bd_u12_ds_w720_fastlr_full30_20260613_223042.yml",
        "checkpoint": checkpoint(
            "nts11bd_u12_ds_w720_fastlr_full30_20260613_223042_bs8_20260613_223042_setsid", 19
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
    {
        "id": "binary_tx_ep19",
        "label": "All-binary + TX",
        "group": "historical attention control",
        "epoch": 19,
        "source_config": EXP / "configs/generated/date11full_all_binary_atlif_tx_w720_fastlr_full30.yml",
        "checkpoint": checkpoint(
            "date11full_all_binary_atlif_tx_w720_fastlr_full30_bs8_20260617_024526_setsid", 19
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
    {
        "id": "ttx_h60_ep2",
        "label": "Frozen TTX/H60",
        "group": "paper core",
        "epoch": 2,
        "source_config": EXP / "configs/generated/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml",
        "checkpoint": checkpoint(
            "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid", 2
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
    {
        "id": "h66a_ep19",
        "label": "H66a a-XNOR matrix",
        "group": "attention family ablation",
        "epoch": 19,
        "source_config": EXP / "configs/generated/h66a_allbinary_all12_axnor_matrix_shiftmax_w720_fastlr_full30.yml",
        "checkpoint": checkpoint(
            "h66a_allbinary_all12_axnor_matrix_shiftmax_w720_fastlr_full30_bs8_full30_20260712_setsid", 19
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
    {
        "id": "h66b_ep29",
        "label": "H66b Hamming linear",
        "group": "attention family ablation",
        "epoch": 29,
        "source_config": EXP / "configs/generated/h66b_allbinary_all12_hamming_linear_w720_fastlr_full30.yml",
        "checkpoint": checkpoint(
            "h66b_allbinary_all12_hamming_linear_w720_fastlr_full30_bs8_full30_20260712_setsid", 29
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
    {
        "id": "h66c_ep19",
        "label": "H66c TP-TTX",
        "group": "attention family ablation",
        "epoch": 19,
        "source_config": EXP / "configs/generated/h66c_allbinary_all12_tp_ttx_w720_fastlr_full30.yml",
        "checkpoint": checkpoint(
            "h66c_allbinary_all12_tp_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid", 19
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
    {
        "id": "h67_float_ep19",
        "label": "H67 Motion-XOR (float)",
        "group": "paper core",
        "epoch": 19,
        "source_config": EXP / "configs/generated/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml",
        "checkpoint": checkpoint(
            "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid", 19
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
    {
        "id": "h67_rtl_ep19",
        "label": "H67 Motion-XOR (RTL-exact)",
        "group": "paper core",
        "epoch": 19,
        "source_config": EXP / "configs/generated/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_dyadic_int8_deploy_rtl_exact.yml",
        "checkpoint": checkpoint(
            "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid", 19
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
    {
        "id": "h68_rtl_ep19",
        "label": "H68 Castling-trained/H60 deploy (RTL-exact)",
        "group": "paper core",
        "epoch": 19,
        "source_config": EXP / "configs/generated/h68_allbinary_all12_castling_ttx_deploy_full30_dyadic_int8_deploy_rtl_exact.yml",
        "checkpoint": checkpoint(
            "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid", 19
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
    {
        "id": "h69_dyadic_ep19",
        "label": "H69 fixed dyadic temperature",
        "group": "mechanism ablation",
        "epoch": 19,
        "source_config": EXP / "configs/generated/h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30_dyadic_int8_deploy.yml",
        "checkpoint": checkpoint(
            "h69_allbinary_all12_dyadic_temperature_ttx_x8_w720_fastlr_full30_bs8_full30_20260711_setsid", 19
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
    {
        "id": "h70_dyadic_ep19",
        "label": "H70 event-selective dyadic TTX",
        "group": "mechanism ablation",
        "epoch": 19,
        "source_config": EXP / "configs/generated/h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30_dyadic_int8_deploy.yml",
        "checkpoint": checkpoint(
            "h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30_bs8_full30_20260711_setsid", 19
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
    {
        "id": "h71_ep19",
        "label": "H71 window-context TTX",
        "group": "mechanism ablation",
        "epoch": 19,
        "source_config": EXP / "configs/generated/h71_allbinary_all12_window_context_ttx_w720_fastlr_full30.yml",
        "checkpoint": checkpoint(
            "h71_allbinary_all12_window_context_ttx_w720_fastlr_full30_bs8_full30_20260711_setsid", 19
        ),
        "expected_atlif": 105,
        "expected_shiftmax": 12,
    },
)


def main() -> int:
    GEN.mkdir(parents=True, exist_ok=True)
    manifest = []
    for candidate in CANDIDATES:
        source = Path(candidate["source_config"])
        ckpt = Path(candidate["checkpoint"])
        if not source.is_file():
            raise FileNotFoundError(source)
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        config = deepcopy(config)
        config["experiment"] = f"valuable_aae_{candidate['id']}"
        config.setdefault("metrics", {})["name"] = ["AEE", "AAE", "AAE_Benchmark"]
        config.setdefault("test", {}).update({"sample": 825, "n_valid": 1})
        generated = GEN / f"{candidate['id']}.yml"
        generated.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        row = dict(candidate)
        row["source_config"] = str(source)
        row["config"] = str(generated)
        row["checkpoint"] = str(ckpt)
        row["output"] = str(RESULTS / "valuable_aae_audit_20260717" / candidate["id"])
        manifest.append(row)
        print(f"{candidate['id']}: {generated}")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
