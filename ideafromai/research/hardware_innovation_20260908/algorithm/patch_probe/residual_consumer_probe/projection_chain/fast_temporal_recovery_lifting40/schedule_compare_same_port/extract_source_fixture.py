#!/usr/bin/python3.12
"""Extract the one declared P2 service input from the ordered real capture."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
POSITIONS = [(0, 0), (0, 1)]

def main():
    result = dict(frame="zurich_city_09_a_0001.npy", positions_yx=POSITIONS,
                  source_format="signed24/f14; vector order p then c; each vector T10",
                  I24=None, gate_words={}, theta={})
    for axis in ["ordinary", "lifting_raw"]:
        cap = np.load(HERE / "capture_inputs" / axis / "000_zurich_city_09_a_0001.npz")
        h, w = map(int, cap["sn1_shape"][-2:])
        bits, inputs, expected = cap["sn1_gate_bits"], [], []
        for y, x in POSITIONS:
            for c in range(96):
                inputs.append(cap["I24_halo"][:, c, y, x].astype(np.int64).tolist())
                index = (c * h + y) * w + x
                expected.append(sum(((int(bits[t, index // 8]) >> (index % 8)) & 1) << t for t in range(10)))
        if result["I24"] is None:
            result["I24"] = inputs
        else:
            assert inputs == result["I24"]
        result["gate_words"][axis] = expected
        result["theta"][axis] = float(cap["sn1_theta"])
    (HERE / "source_tile_fixture.json").write_text(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__":
    main()
