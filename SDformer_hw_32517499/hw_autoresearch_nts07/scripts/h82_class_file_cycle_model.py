#!/usr/bin/env python3
"""Cycle proxy for H82 Class File vs the live C7 directory. Not RTL, not DC.

Old 2s top (see h67_temporal_slot_shiftmax_sync_k_2s_top.sv):
  BUILD descriptors + CLASS occupancy scan + ACTIVE token emit
  exp2 is charged once per occupied class (denom) and once per emitted token.

H82 contract:
  CLASSIFY 450 scores into the Class File + one-vote SHIFTMAX + EXPAND
  exp2 is charged only per occupied class. Emit broadcasts gate_c.

This is an execution-object delta. It is not an Amdahl claim and must not
enter docs/359. C_occ on the frozen pack was ~9.54; use that as a sensitivity
point, not as H82 evidence.
"""

from __future__ import annotations

import json
from pathlib import Path


def old_c7_cycles(n_tokens: int, n_occupied: int, n_active: int) -> dict[str, int]:
    return {
        "build": n_tokens,
        "class_scan_and_exp": n_occupied,
        "emit_recompute_exp": n_active,
        "total": n_tokens + n_occupied + n_active,
    }


def h82_cycles(n_tokens: int, n_occupied: int, n_active: int) -> dict[str, int]:
    return {
        "classify": n_tokens,
        "one_vote_shiftmax": n_occupied,
        "expand_broadcast": n_active,
        "total": n_tokens + n_occupied + n_active,
    }


def compare(n_tokens: int = 450, n_occupied: int = 10, n_active: int = 450) -> dict:
    old = old_c7_cycles(n_tokens, n_occupied, n_active)
    new = h82_cycles(n_tokens, n_occupied, n_active)
    return {
        "n_tokens": n_tokens,
        "n_occupied": n_occupied,
        "n_active": n_active,
        "old_c7": old,
        "h82": new,
        "control_cycles_same": old["total"] == new["total"],
        "exp2_old": n_occupied + n_active,
        "exp2_h82": n_occupied,
        "exp2_ratio": (n_occupied + n_active) / max(n_occupied, 1),
        "verdict": (
            "Schedule length is the same order (classify + expand still walk 450). "
            "The object change is that Shiftmax no longer consumes 450 scores and "
            "emit no longer recomputes exp2. That is not a cycle win large enough "
            "to sell DATE 4.0 by itself."
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "h82_class_file_isa_20260817"
    out.mkdir(parents=True, exist_ok=True)
    points = [
        compare(450, 10, 450),
        compare(450, 32, 450),
        compare(450, 9, 225),
    ]
    payload = {"schema": "h82_class_file_cycle_proxy_v1", "points": points}
    (out / "cycle_proxy.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
