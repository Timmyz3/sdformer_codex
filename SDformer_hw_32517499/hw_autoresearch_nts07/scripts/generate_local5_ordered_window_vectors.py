#!/usr/bin/env python3
"""Generate ordered Local5 window vectors (synthetic hardware-order).

Local5-only artifact generator — counterpart to Motion real-trace vector
scripts, but synthetic until post-G0 software export is frozen.
Writes JSON consumed by TB / cycle models.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "local5_ordered_vectors_20260727"


def bits32(rng: random.Random) -> int:
    return rng.getrandbits(32)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260727)
    n_windows = 8
    dests_per_window = 16
    windows = []
    for w in range(n_windows):
        dests = []
        for d in range(dests_per_window):
            # interior-like full stencil; borders thinner
            if d % 8 == 0:
                mask = 0b10101  # self,S,W
            elif d % 8 == 7:
                mask = 0b11011
            else:
                mask = 0b11111
            mask |= 0b00001  # force self
            dests.append(
                {
                    "dest_id": d,
                    "tag": w * 256 + d,
                    "valid_mask": mask,
                    "q": bits32(rng),
                    "k_self": bits32(rng),
                    "k_n": bits32(rng),
                    "k_s": bits32(rng),
                    "k_e": bits32(rng),
                    "k_w": bits32(rng),
                    "last": d == dests_per_window - 1,
                }
            )
        windows.append({"window_id": w, "destinations": dests})

    payload = {
        "schema": "local5_ordered_window_vectors_v1",
        "head_dim": 32,
        "n_cand": 5,
        "evidence": "[synthetic] not post-G0 software export",
        "windows": windows,
        "stats": {
            "windows": n_windows,
            "dests_per_window": dests_per_window,
            "total_dests": n_windows * dests_per_window,
        },
    }
    path = OUT / "ordered_windows.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
