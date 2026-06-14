#!/usr/bin/env python3
"""Export TTB window_enable mask from spike_profile.json for hardware scheduler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def layer_to_stage_window(layer_name: str) -> tuple[int | None, str]:
    if "swin_blocks" not in layer_name:
        return None, layer_name
    parts = layer_name.split(".")
    try:
        stage = int(parts[parts.index("layers") + 1])
        block = int(parts[parts.index("swin_blocks") + 1])
        leaf = parts[-1]
        return stage, f"s{stage}_b{block}_{leaf}"
    except (ValueError, IndexError):
        return None, layer_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--firing-threshold", type=float, default=0.0, help="Min firing_rate to enable window")
    args = parser.parse_args()

    data = json.loads(args.profile.read_text(encoding="utf-8"))
    layers = data.get("layer_firing_rates", {})
    entries = []
    for name, stats in layers.items():
        fr = float(stats["firing_rate"])
        stage, tag = layer_to_stage_window(name)
        entries.append(
            {
                "layer": name,
                "tag": tag,
                "stage": stage,
                "spikes": int(stats["spikes"]),
                "elements": int(stats["elements"]),
                "firing_rate": fr,
                "window_enable": fr > args.firing_threshold,
            }
        )
    entries.sort(key=lambda x: (-x["firing_rate"], x["layer"]))
    out = {
        "source_profile": str(args.profile),
        "global_firing_rate": data.get("global_firing_rate"),
        "sparsity_ratio": data.get("sparsity_ratio"),
        "effective_flops": data.get("effective_flops"),
        "firing_threshold": args.firing_threshold,
        "enabled_layers": sum(1 for e in entries if e["window_enable"]),
        "total_layers": len(entries),
        "layers": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({out['enabled_layers']}/{out['total_layers']} layers enabled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())