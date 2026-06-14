"""NTS-11 phase-4: binary/ternary scope combinatorics on the two-neuron line.

All variants keep the deployment story to *at most* two neuron types at inference
(ternary ATLIF-PSN + binary official ATLIF-PSN). No vanilla PSN is left once
sn2_q is explicitly covered (11r fixes the latent 12-path vanilla gap in 11q).

Training knobs follow phase-1/2 best direction: fast LR + threshold freeze816.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "overlay"))

from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
from models.STSwinNet_SNN.atlif_ternary_psn.installer import iter_non_qk_spiking_neuron_paths

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nts11_two_neuron_only_configs import (
    BASE,
    apply_hparam_overrides,
    apply_two_neuron_only_policy,
    blocks_for,
    read_yaml,
    set_runtime,
    write_yaml,
)


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"

BEST_HPARAMS = {
    "neuron_lr": 5.0e-5,
    "backbone_lr": 2.0e-6,
    "threshold_freeze_after_step": 816,
}


def binary_group(name: str, *, paths: list[str] | None = None, path_selection: str = "", **overrides: Any) -> dict[str, Any]:
    group: dict[str, Any] = {
        "name": name,
        "output_mode": "binary",
        "threshold_mode": "official_atlif",
        "center_mode": "zero",
        "threshold_eta": 0.0,
        "activity_eta": 0.0,
        "target_rate": None,
        "target_rate_eta": 0.0,
    }
    if path_selection:
        group["path_selection"] = path_selection
    if paths is not None:
        group["paths"] = list(paths)
    group.update(overrides)
    return group


def ternary_group(name: str, paths: list[str], **overrides: Any) -> dict[str, Any]:
    group: dict[str, Any] = {
        "name": name,
        "paths": list(paths),
        "output_mode": "ternary",
        "threshold_mode": "symmetric_bsa_tsn",
        "center_mode": "bias",
        "threshold_eta": 6.5e-4,
        "threshold_lr_scale": 50000.0,
        "activity_eta": 0.0,
        "target_rate": None,
        "target_rate_eta": 0.0,
    }
    group.update(overrides)
    return group


def build_path_sets() -> dict[str, list[str]]:
    cfg = read_yaml(BASE)
    swin = dict(cfg["swin_transformer"])
    swin["input_size"] = list(cfg["loader"]["crop"])
    model_cfg = cfg["model"].copy()
    model_cfg["spiking_neuron"] = cfg["spiking_neuron"]
    model = MS_SpikingformerFlowNet_en4(model_cfg, swin)
    model.init_weights()

    all_sn = sorted(name for name, module in model.named_modules() if module.__class__.__name__ == "Spiking_neuron")
    sn2q = [name for name in all_sn if name.endswith(".sn2_q")]
    non_qk = iter_non_qk_spiking_neuron_paths(model)

    def pick(predicate) -> list[str]:
        return sorted(path for path in non_qk if predicate(path))

    attn_aux = pick(lambda p: "attn_sn" in p or "proj_sn" in p)
    ffn = pick(lambda p: ".mlp." in p)
    downsample = pick(lambda p: "downsample" in p)
    patch_embed = pick(lambda p: "patch_embed" in p)
    decoder_pred = pick(lambda p: ".decoders." in p or ".preds." in p)
    unet_resblock = pick(lambda p: p.startswith("sttmultires_unet.resblocks."))
    decoder_head = sorted(decoder_pred + unet_resblock)

    ffn_s0 = pick(lambda p: ".mlp." in p and ".layers.0." in p)
    ffn_s2 = pick(lambda p: ".mlp." in p and ".layers.2." in p)
    ffn_sn1 = pick(lambda p: p.endswith(".mlp.sn1"))

    encoder_body = sorted(attn_aux + ffn + downsample + patch_embed)

    return {
        "sn2q": sn2q,
        "attn_aux": attn_aux,
        "ffn": ffn,
        "ffn_s0": ffn_s0,
        "ffn_s2": ffn_s2,
        "ffn_sn1": ffn_sn1,
        "downsample": downsample,
        "patch_embed": patch_embed,
        "decoder_head": decoder_head,
        "encoder_body": encoder_body,
        "non_qk": list(non_qk),
    }


def apply_scope_policy(cfg: dict[str, Any], policy: str, paths: dict[str, list[str]]) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif.pop("target_paths", None)

    if policy == "baseline_11l":
        atlif["target_groups"] = [
            binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"),
        ]
        return

    if policy == "sn2q_binary":
        atlif["target_groups"] = [
            binary_group("sn2q_binary", paths=paths["sn2q"]),
            binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"),
        ]
        return

    if policy == "sn2q_ternary":
        atlif["target_paths"] = list(paths["sn2q"])
        atlif["target_groups"] = [
            binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"),
        ]
        return

    groups: list[dict[str, Any]] = []

    if policy == "attnaux_ternary":
        groups.extend(
            [
                ternary_group("attn_aux_ternary", paths["attn_aux"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    elif policy == "ffn_all_ternary":
        groups.extend(
            [
                ternary_group("ffn_all_ternary", paths["ffn"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    elif policy == "ffn_s0_ternary":
        groups.extend(
            [
                ternary_group("ffn_s0_ternary", paths["ffn_s0"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    elif policy == "ffn_s2_ternary":
        groups.extend(
            [
                ternary_group("ffn_s2_ternary", paths["ffn_s2"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    elif policy == "ffn_sn1_ternary":
        groups.extend(
            [
                ternary_group("ffn_sn1_ternary", paths["ffn_sn1"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    elif policy == "downsample_ternary":
        groups.extend(
            [
                ternary_group("downsample_ternary", paths["downsample"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    elif policy == "decoder_ternary":
        groups.extend(
            [
                ternary_group("decoder_head_ternary", paths["decoder_head"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    elif policy == "encoder_ternary":
        groups.extend(
            [
                ternary_group("encoder_body_ternary", paths["encoder_body"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
                binary_group("decoder_head_binary", paths=paths["decoder_head"]),
            ]
        )
    elif policy == "patch_embed_ternary":
        groups.extend(
            [
                ternary_group("patch_embed_ternary", paths["patch_embed"]),
                binary_group("sn2q_binary", paths=paths["sn2q"]),
            ]
        )
    else:
        raise ValueError(f"unknown scope policy: {policy}")

    groups.append(binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"))
    atlif["target_groups"] = groups


def make_config(base: dict[str, Any], spec: dict[str, Any], paths: dict[str, list[str]]) -> Path:
    cfg = deepcopy(base)
    set_runtime(cfg, spec["name"], spec["note"])
    apply_two_neuron_only_policy(cfg)
    apply_scope_policy(cfg, str(spec["scope_policy"]), paths)
    apply_hparam_overrides(cfg, {**BEST_HPARAMS, **spec})

    attn = cfg.setdefault("bsa_attention", {})
    attn["target_blocks"] = blocks_for("s23", base)

    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    paths = build_path_sets()
    specs: list[dict[str, Any]] = [
        {
            "name": "nts11q_hw_h60_s23_scope_baseline11l_s1224",
            "scope_policy": "baseline_11l",
            "note": (
                "NTS-11q: scope control — same as 11l (fastlr+freeze816, all_non_qk binary). "
                "sn2_q remains uncovered (12 vanilla) for A/B reference."
            ),
        },
        {
            "name": "nts11r_hw_h60_s23_scope_sn2q_binary_s1224",
            "scope_policy": "sn2q_binary",
            "note": (
                "NTS-11r: strict two-neuron fix — explicit binary ATLIF on 12 sn2_q paths, "
                "then all_non_qk for the rest."
            ),
        },
        {
            "name": "nts11s_hw_h60_s23_scope_sn2q_ternary_s1224",
            "scope_policy": "sn2q_ternary",
            "note": "NTS-11s: extend ternary expressiveness to sn2_q (12 paths) via target_paths.",
        },
        {
            "name": "nts11t_hw_h60_s23_scope_attnaux_ternary_s1224",
            "scope_policy": "attnaux_ternary",
            "note": "NTS-11t: Q/K + attn_sn/proj_sn (24) ternary; sn2_q binary; remainder all_non_qk binary.",
        },
        {
            "name": "nts11u_hw_h60_s23_scope_ffn_all_ternary_s1224",
            "scope_policy": "ffn_all_ternary",
            "note": "NTS-11u: Q/K + all FFN mlp (24) ternary; sn2_q binary; remainder binary.",
        },
        {
            "name": "nts11v_hw_h60_s23_scope_ffn_s0_ternary_s1224",
            "scope_policy": "ffn_s0_ternary",
            "note": "NTS-11v: Q/K + stage0 FFN only (4 paths) ternary.",
        },
        {
            "name": "nts11w_hw_h60_s23_scope_ffn_s2_ternary_s1224",
            "scope_policy": "ffn_s2_ternary",
            "note": "NTS-11w: Q/K + stage2 FFN only (12 paths) ternary.",
        },
        {
            "name": "nts11x_hw_h60_s23_scope_decoder_ternary_s1224",
            "scope_policy": "decoder_ternary",
            "note": "NTS-11x: Q/K + decoder/pred/unet-resblock head (12 paths) ternary.",
        },
        {
            "name": "nts11y_hw_h60_s23_scope_encoder_ternary_s1224",
            "scope_policy": "encoder_ternary",
            "note": (
                "NTS-11y: Q/K + full encoder body (ffn+downsample+attn_aux+patch_embed) ternary; "
                "decoder head stays binary."
            ),
        },
        {
            "name": "nts11z_hw_h60_s23_scope_ffn_sn1_ternary_s1224",
            "scope_policy": "ffn_sn1_ternary",
            "note": "NTS-11z: Q/K + FFN sn1 up-proj only (12 paths) ternary; sn2 down-proj binary.",
        },
        {
            "name": "nts11aa_hw_h60_s23_scope_downsample_ternary_s1224",
            "scope_policy": "downsample_ternary",
            "note": "NTS-11aa: Q/K + downsample nodes (3 paths) ternary.",
        },
        {
            "name": "nts11ab_hw_h60_s23_scope_patch_embed_ternary_s1224",
            "scope_policy": "patch_embed_ternary",
            "note": "NTS-11ab: Q/K + patch-embed spike path (6 paths) ternary.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec, paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())