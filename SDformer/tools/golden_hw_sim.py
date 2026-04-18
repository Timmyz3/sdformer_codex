"""Generate fixed-point golden vectors for the simplified RTL spike path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import yaml


def quantize_signed(value: float, scale: float, bits: int) -> int:
    max_q = 2 ** (bits - 1) - 1
    min_q = -2 ** (bits - 1)
    q = int(round(value / scale))
    return max(min_q, min(max_q, q))


def spike_step(samples: List[int], membrane: List[int], threshold: int) -> tuple[List[int], List[int]]:
    out = []
    next_mem = []
    for sample, mem in zip(samples, membrane):
        acc = sample + mem
        if acc >= threshold:
            out.append(1)
            next_mem.append(0)
        else:
            out.append(0)
            next_mem.append(acc)
    return out, next_mem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-spec", default="configs/hw/quant_spec.yaml")
    parser.add_argument("--output-dir", default="hw/tb/tb_vectors")
    args = parser.parse_args()

    with Path(args.quant_spec).open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    bits = spec["quantization"]["activation_bits"]
    scale = 1.0 / max(1, 2 ** (bits - 2))
    threshold = 4
    membrane = [0] * 8
    stimuli = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [4, 4, 4, 4, 0, 0, 0, 0],
    ]

    vectors = []
    for step, samples in enumerate(stimuli):
        spikes, membrane = spike_step(samples, membrane, threshold)
        vectors.append({"step": step, "input": samples, "expected_spike": spikes, "membrane": membrane[:]})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({"vectors": vectors, "threshold": threshold}, indent=2), encoding="utf-8")
    (out_dir / "stimulus.hex").write_text(
        "\n".join("".join(f"{value & 0xFF:02x}" for value in vector["input"]) for vector in vectors) + "\n",
        encoding="utf-8",
    )
    (out_dir / "expected.hex").write_text(
        "\n".join("".join(f"{value & 0x1:02x}" for value in vector["expected_spike"]) for vector in vectors) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

