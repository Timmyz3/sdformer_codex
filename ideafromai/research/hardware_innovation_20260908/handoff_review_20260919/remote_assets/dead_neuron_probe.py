"""Dead-neuron forensic probe for H9 ternary line.

Reuses H9 train.py's install chain (same patching of the baseline entry),
but exits before the training loop. Collects per-ATLIFTernaryPSN stats on a
few DSEC valid batches, at (A) theta as loaded and (B) theta*0.1 to test the
scale-death hypothesis. Also checks reachability of sn2_q/attn_sn modules
(structural death via Shiftmax overlay).

Outputs: /tmp/deadprobe_A_theta_loaded.json, /tmp/deadprobe_B_theta_x01.json,
/tmp/deadprobe_reachability.json
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

TRAIN_PY = Path(
    "/root/private_data/work/sdformer_codex/SDformer/"
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py"
)

CONFIG = sys.argv[1]
PREV_CKPT = sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else "main"

PROBE_BLOCK = '''
# ===== DEAD NEURON PROBE INSERT (begins) =====
import json as _json
from spikingjelly.activation_based import functional as _sjf
from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN as _ATP

print("[PROBE] collecting ternary modules ...", flush=True)
_probe_ternary = [(n, m) for n, m in model.named_modules() if isinstance(m, _ATP)]
print(f"[PROBE] ATLIFTernaryPSN modules found: {len(_probe_ternary)}", flush=True)

# --- structural reachability: hooks on candidate-dead attention submodules ---
_reach = {}
_reach_handles = []

def _mk_reach(name):
    def _h(mod, inp, out):
        _reach[name]["calls"] += 1
    return _h

for n, m in model.named_modules():
    if n.endswith("sn2_q") or n.endswith("attn_sn") or ".sn2_q." in n:
        if len(list(m.children())) == 0 or isinstance(m, _ATP):
            _reach[n] = {"calls": 0, "type": type(m).__name__}
            _reach_handles.append(m.register_forward_hook(_mk_reach(n)))
print(f"[PROBE] watching {len(_reach)} candidate-structurally-dead modules", flush=True)

def _probe_collect(tag):
    acc = {n: {"r": [], "pos_r": [], "neg_r": [], "ptrig": [], "ntrig": []} for n, _ in _probe_ternary}
    handles = []

    def _mk(n):
        def _h(mod, inp, out):
            a = acc[n]
            a["r"].append(mod.r)
            a["pos_r"].append(mod.pos_r)
            a["neg_r"].append(mod.neg_r)
            a["ptrig"].append(mod.positive_trigger_r)
            a["ntrig"].append(mod.negative_trigger_r)
        return _h

    for n, m in _probe_ternary:
        handles.append(m.register_forward_hook(_mk(n)))
    model.eval()
    nb = 0
    _tv = globals().get("transform_valid", None)
    with torch.set_grad_enabled(False):
        for chunk, mask, label in valid_dataloader:
            if nb >= 3:
                break
            _sjf.reset_net(model)
            _sjf.set_step_mode(model, config['data']['step_mode'])
            chunk = chunk.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            mask = torch.unsqueeze(mask.to(device=device), dim=1)
            if _tv is not None:
                chunk, label, mask = _tv((chunk, label, mask.float()))
            if config['model']['encoding'] == 'cnt':
                if config['loader']['polarity']:
                    chunk = chunk.view([chunk.shape[0], -1] + list(chunk.shape[3:]))
            elif config['model']['encoding'] == 'voxel':
                if config['loader']['polarity']:
                    _neg = torch.nn.functional.relu(-chunk)
                    _pos = torch.nn.functional.relu(chunk)
                    chunk = torch.cat((torch.unsqueeze(_pos, dim=2), torch.unsqueeze(_neg, dim=2)), dim=2)
            if config["model"]["norm_input"] == "minmax":
                _mn, _mx = torch.min(chunk[chunk != 0]), torch.max(chunk[chunk != 0])
                if not _mn == _mx:
                    chunk[chunk != 0] = (chunk[chunk != 0] - _mn) / (_mx - _mn)
            elif config["model"]["norm_input"] == "std":
                _m, _s = chunk[chunk != 0].mean(), chunk[chunk != 0].std()
                if _s > 0:
                    chunk[chunk != 0] = (chunk[chunk != 0] - _m) / _s
            if config['data']['spike_th'] is not None:
                chunk[chunk > config['data']['spike_th']] = 1
                chunk[chunk < config['data']['spike_th']] = 0
            _ = model(chunk)
            nb += 1
    for h in handles:
        h.remove()
    rows = {}
    for n, m in _probe_ternary:
        a = acc[n]
        rows[n] = {
            "T": m.T,
            "thresh": float(m.thresh.detach().cpu()),
            "neg_scale": float(m.negative_threshold_scale),
            "r": sum(a["r"]) / max(1, len(a["r"])),
            "pos_r": sum(a["pos_r"]) / max(1, len(a["pos_r"])),
            "neg_r": sum(a["neg_r"]) / max(1, len(a["neg_r"])),
            "ptrig": sum(a["ptrig"]) / max(1, len(a["ptrig"])),
            "ntrig": sum(a["ntrig"]) / max(1, len(a["ntrig"])),
        }
    with open(f"/tmp/deadprobe_{tag}.json", "w") as f:
        _json.dump(rows, f, indent=1)
    print(f"[PROBE] wrote /tmp/deadprobe_{tag}.json n_modules={len(rows)} batches={nb}", flush=True)

print("[PROBE] pass A: theta as loaded", flush=True)
_probe_collect("A_theta_loaded_" + "''' + TAG + '''")

for _n, _m in _probe_ternary:
    with torch.no_grad():
        _m.thresh.mul_(0.1)
print("[PROBE] pass B: theta *= 0.1 (scale-death test)", flush=True)
_probe_collect("B_theta_x01_" + "''' + TAG + '''")

with open("/tmp/deadprobe_reachability_" + "''' + TAG + '''.json", "w") as f:
    _json.dump(_reach, f, indent=1)
print("[PROBE] wrote reachability json", flush=True)
for _h in _reach_handles:
    _h.remove()
print("[PROBE] done, exiting before training loop", flush=True)
raise SystemExit(0)
# ===== DEAD NEURON PROBE INSERT (ends) =====
'''

EPOCH_LOOP_ANCHOR = 'for epoch in range(epoch_initial, config["loader"]["n_epochs"]):'


def main() -> None:
    spec = importlib.util.spec_from_file_location("h9_train_module", TRAIN_PY)
    h9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h9)

    repo_root = h9._repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = TRAIN_PY.parents[1] / "overlay"
    baseline_entry = baseline_root / "train_flow_parallel_supervised_SNN.py"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))
    sys.argv = [
        str(baseline_entry),
        "--config", str(Path(CONFIG).resolve()),
        "--prev_runid", str(Path(PREV_CKPT).resolve()),
        "--save_path", f"/tmp/deadprobe_ckpt_{TAG}/checkpoint_epoch{{}}.pth",
        "--finetune", "1",
    ]

    h9._install_optional_mlflow_stub()
    os.chdir(baseline_root)
    source = h9._patch_source(baseline_entry.read_text(), baseline_entry)
    indented_anchor = "    " + EPOCH_LOOP_ANCHOR
    assert source.count(indented_anchor) == 1, "epoch-loop anchor not unique"
    probe_indented = "\n".join("    " + line if line.strip() else "" for line in PROBE_BLOCK.split("\n"))
    source = source.replace(
        indented_anchor,
        probe_indented + "\n" + indented_anchor,
        1,
    )
    code = compile(source, str(baseline_entry), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(baseline_entry)})


if __name__ == "__main__":
    main()
