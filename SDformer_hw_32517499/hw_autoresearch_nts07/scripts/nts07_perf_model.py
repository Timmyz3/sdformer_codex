#!/usr/bin/env python3
"""Analytical perf/energy model for NTS-07b hardware accelerator autoresearch."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path


ANCHOR_PRESETS: dict[str, dict[str, float]] = {
    # valid825 ep29 — deprecated mixed-attn baseline
    "nts07b": {
        "firing": 0.07942,
        "legacy_attn_mc": 2.10,
        "h60_attn_mc": 1.40,
        "energy_scale": 1.0,
        "unified_h60": 0,
    },
    # valid825 ep19 — deprecated Legacy+H60 mix (s23)
    "nts11aa": {
        "firing": 0.062232,
        "legacy_attn_mc": 1.93,   # S0+S1 Legacy only
        "h60_attn_mc": 1.64,      # S2+S3 H60 (8 blocks)
        "energy_scale": 0.725,    # 22893/31581 uJ vs 07b ep29
        "downsample_firing_peak": 0.528,  # layers.2.downsample.sn ep19
        "ternary_layers": 30,
        "unified_h60": 0,
    },
    # Projected hw-friendly scope (11aw): no downsample ternary, 24 Q/K ternary only
    "nts11aw": {
        "firing": 0.070,
        "legacy_attn_mc": 1.85,
        "h60_attn_mc": 1.55,
        "energy_scale": 0.78,
        "downsample_firing_peak": 0.29,
        "ternary_layers": 24,
        "unified_h60": 0,
    },
    # Minimal 2-bit footprint (11ax): s23 Q/K ternary only (16 paths)
    "nts11ax": {
        "firing": 0.072,
        "legacy_attn_mc": 1.80,
        "h60_attn_mc": 1.52,
        "energy_scale": 0.80,
        "downsample_firing_peak": 0.29,
        "ternary_layers": 16,
        "unified_h60": 0,
    },
    # NTS-11bc/11bd: unified H60 on all 12 encoder blocks (short-test provisional)
    "nts11bc": {
        "firing": 0.062232,
        "legacy_attn_mc": 0.0,
        "h60_attn_mc": 4.45,
        "energy_scale": 0.82,
        "downsample_firing_peak": 0.53,
        "ternary_layers": 27,
        "unified_h60": 1,
    },
}


@dataclass
class HwConfig:
    hw_anchor: str = "nts11bc"
    pe_mac: int = 128
    tx_sc_parallel: int = 1
    skip_empty_windows: int = 1
    window_sram_kb: int = 512
    weight_buffer_kb: int = 256
    freq_mhz: float = 500.0
    firing: float = 0.062232
    e_ac_pj: float = 0.9
    e_mac_pj: float = 4.6
    # Baseline Mcycles from docs/08 (NTS-07b @ 288x384)
    scatter_mc: float = 0.10
    patch_mc: float = 1.20
    legacy_attn_mc: float = 2.10
    h60_attn_mc: float = 1.40
    sparse_mac_mc: float = 0.90
    decode_mc: float = 0.20
    encode_mc: float = 0.12
    # Segment-1 literature-inspired knobs
    unified_atlif_encode: int = 0
    bishop_ttb_depth: int = 1
    firefly_popcount_par: int = 32
    shared_encode_lanes: int = 8


def model(cfg: HwConfig) -> dict[str, float]:
    preset = ANCHOR_PRESETS.get(cfg.hw_anchor, ANCHOR_PRESETS["nts11bc"])
    legacy_base = preset["legacy_attn_mc"]
    h60_base = preset["h60_attn_mc"]
    energy_scale = preset["energy_scale"]
    # Bishop TTB: depth-2 bundles window×timestep for better skip hit rate
    if cfg.skip_empty_windows:
        skip_gain = 0.84 if cfg.bishop_ttb_depth >= 2 else 0.88
    else:
        skip_gain = 1.0

    pop_par = max(8, cfg.firefly_popcount_par)
    h60_par = 0.92 if cfg.tx_sc_parallel else 1.0
    h60_par *= min(1.0, 32.0 / pop_par) * 0.98 + 0.02  # FireFly-style wider popcount

    pe_gain = 128.0 / max(1, cfg.pe_mac)
    firing_gain = (1.0 - cfg.firing) / (1.0 - 0.085)

    encode_gain = 0.88 if cfg.unified_atlif_encode else 1.0
    lane_gain = min(1.0, cfg.shared_encode_lanes / 8.0) * 0.04 + 0.96

    legacy = legacy_base * skip_gain
    h60 = h60_base * skip_gain * h60_par
    mac = cfg.sparse_mac_mc * firing_gain * pe_gain
    encode = cfg.encode_mc * encode_gain * lane_gain
    total_mc = cfg.scatter_mc + cfg.patch_mc + legacy + h60 + mac + encode + cfg.decode_mc
    total_cycles = total_mc * 1e6

    # NTS-07b valid825 ep29 profile energy ~29.3 mJ (vs NB0 ~37.6 mJ)
    base_energy = 29.3 * energy_scale * firing_gain * skip_gain * pe_gain
    encode_energy = 0.35 * (1.0 - encode_gain) + 0.15 * (skip_gain - 0.88) * 10
    energy_mj = base_energy - encode_energy
    sram_kb = cfg.window_sram_kb + cfg.weight_buffer_kb + 4

    # Unified encoder shares comparator tree (~8% encode-area savings)
    area_encode_mm2 = 0.42 * encode_gain * lane_gain
    legacy_area_save = 0.35 if preset.get("unified_h60", 0) else 0.0
    area_total_mm2 = 2.85 - (0.42 - area_encode_mm2) if cfg.unified_atlif_encode else 2.85
    area_total_mm2 -= legacy_area_save

    return {
        "effective_cycles": total_cycles,
        "effective_energy_mj": energy_mj,
        "sram_kb": sram_kb,
        "area_mm2": area_total_mm2,
        "epe_drift": 0.0,
        "legacy_mc": legacy,
        "h60_mc": h60,
        "mac_mc": mac,
        "encode_mc": encode,
        "fps_at_500mhz": cfg.freq_mhz * 1e6 / total_cycles,
        "hw_anchor": cfg.hw_anchor,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = HwConfig()
    if args.config and args.config.exists():
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    m = model(cfg)
    if args.json:
        print(json.dumps({"config": asdict(cfg), "metrics": m}, indent=2))
    else:
        for k, v in m.items():
            print(f"METRIC {k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())