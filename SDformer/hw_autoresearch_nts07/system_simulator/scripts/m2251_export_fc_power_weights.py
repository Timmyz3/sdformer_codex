#!/usr/bin/env python3
"""Real ep34 FC weights for later switching sensitivity, not accuracy admission.

Use the installed pytorch310_cpu environment. Quantization is an explicit
candidate: output-channel power-of-two scale, nearest-even INT8 [-127,127].
Only output tile 0 (96 channels) of the three preselected windows is exported.
"""
import argparse
import json
from pathlib import Path

import torch

HW = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    checkpoint = HW / "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    layers = json.loads((HW / "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901/layers.json").read_text())["layers"]
    by_id = {r["layer_id"]: r for r in layers}
    args.output.mkdir()
    rows = []
    for window, layer_id in (("low", 15), ("median", 30), ("high", 13)):
        key = by_id[layer_id]["module_name"] + ".weight"
        full = state[key].float()
        if full.ndim != 2 or full.shape[1] != 768:
            raise ValueError("Pilot FC dimension differs from frozen G48 source")
        maxima = full.abs().amax(dim=1)
        scale = torch.pow(2.0, torch.ceil(torch.log2(torch.clamp(maxima/127, min=2.0**-30))))
        codes = torch.round(full / scale[:,None]).clamp(-127,127).to(torch.int32)
        bound = codes.abs().sum(dim=1).max().item()
        if bound >= 2**23:
            raise ValueError("Signed unit-source Acc24 bound exceeded")
        selected = codes[:96]
        # Address = (((group * 2 + half) * 6 + slice) * 8 + bank) * 16 + lane.
        words = []
        for g in range(48):
            for half in range(2):
                for output_slice in range(6):
                    for bank in range(8):
                        for lane in range(16):
                            words.append(int(selected[output_slice*16+lane,g*16+half*8+bank]))
        path = args.output / f"{window}_weights.memh"
        path.write_text("".join(f"{v & 255:02x}\n" for v in words))
        rows.append(dict(window=window, layer_id=layer_id, key=key,
            full_shape=list(full.shape), output_channels=[0,96], code_count=len(words),
            min_code=min(words), max_code=max(words), zero_fraction=sum(v==0 for v in words)/len(words),
            all_output_acc24_absolute_bound=bound, scale_exponent_min=int(torch.log2(scale).min()),
            scale_exponent_max=int(torch.log2(scale).max()), file=str(path)))
    result = dict(checkpoint=str(checkpoint), quantization="Candidate, per-output-channel power-of-two scale; ties-to-even [-127,127]",
        scope="Power sensitivity input preparation only; no VCS/PTPX or full-network AEE result", rows=rows,
        checkpoint_weights_changed=False, training=False, paper_accuracy_admission=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
