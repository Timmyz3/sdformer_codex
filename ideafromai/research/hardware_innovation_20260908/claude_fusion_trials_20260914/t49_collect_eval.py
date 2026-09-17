#!/usr/bin/env python3
"""T49-5: 把一批 eval 的 spike_profile.json 汇总成一张表（远端或本地都能跑）。

用法：
  python t49_collect_eval.py <root> [<root> ...]
会递归找 <root> 下的 **/spike_profile.json，按路径排序，打印一行一个 run：
  path | AEE | DSEC_Fl | AAE_Benchmark | FR% | sparsity% | spikes(G) | energy(uJ)
profile 里的 metrics 值是**字符串**（踩过一次 `must be real number, not str`）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COLS = [("AEE", "%s"), ("DSEC_Fl", "%s"), ("AAE_Benchmark", "%s")]


def fmt(v, spec: str) -> str:
    try:
        return spec % float(v)
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:]] or [Path(".")]
    found: list[Path] = []
    for r in roots:
        if r.is_file() and r.name.endswith(".json"):
            found.append(r)
        else:
            found.extend(sorted(r.rglob("spike_profile.json")))
    if not found:
        print("no spike_profile.json under", roots)
        return 1

    print("%-58s %-10s %-10s %-10s %8s %9s %10s %12s" %
          ("run/epoch", "AEE", "DSEC_Fl", "AAE_Bench", "FR%", "sparsity%", "spikes_G", "energy_uJ"))
    for p in found:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:                      # noqa: BLE001
            print("%-58s  <unreadable: %s>" % (str(p), exc))
            continue
        m = {k: v for k, v in (d.get("metrics") or {}).items()}
        label = ("%s/%s" % (p.parents[2].name, p.parent.name)) if len(p.parents) > 2 else str(p.parent)
        print("%-58s %-10s %-10s %-10s %8.3f %9.3f %10.3f %12.1f" % (
            label[:58],
            fmt(m.get("AEE", "-"), "%.5f"),
            fmt(m.get("DSEC_Fl", "-"), "%.5f"),
            fmt(m.get("AAE_Benchmark", "-"), "%.5f"),
            100.0 * float(d.get("global_firing_rate", 0.0)),
            100.0 * float(d.get("sparsity_ratio", 0.0)),
            float(d.get("total_spikes", 0.0)) / 1e9,
            float(d.get("energy_uj", 0.0)),
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
