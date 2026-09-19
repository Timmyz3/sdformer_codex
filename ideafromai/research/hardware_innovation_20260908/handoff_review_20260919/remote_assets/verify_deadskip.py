"""Bit-exact verification: statically skipping attention branches of blocks whose
sn_k is dead (K==0 -> attn=K*gate==0) must leave model outputs unchanged.

Reuses the H9 train.py install chain (like /tmp/dead_neuron_probe.py), runs the
model twice on the same 3 DSEC valid batches (same reset/seed), once normally and
once with the dead blocks' attn.forward replaced by a zeros-return, then compares
final outputs bitwise.

Usage: python /tmp/verify_deadskip.py <config> <ckpt> <tag>
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

TRAIN_PY = Path(
    "/root/private_data/work/sdformer_codex/SDformer/"
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py"
)

CONFIG = sys.argv[1]
PREV_CKPT = sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else "h67ep35"

SKIP_BLOCKS = {
    "h67ep35": ["layers.1.swin_blocks.0", "layers.2.swin_blocks.3"],
    "local5ep44": ["layers.1.swin_blocks.0", "layers.0.swin_blocks.1", "layers.2.swin_blocks.3"],
}[TAG]

VERIFY_BLOCK = '''
# ===== DEAD-ATTN SKIP VERIFY (begins) =====
import hashlib as _hl
import json as _json
from spikingjelly.activation_based import functional as _sjf

_SKIP = ''' + repr(SKIP_BLOCKS) + '''

def _collect_tensors(obj, out_list):
    if hasattr(obj, "detach"):
        out_list.append(obj.detach())
    elif isinstance(obj, dict):
        for k in sorted(obj.keys(), key=str):
            _collect_tensors(obj[k], out_list)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_tensors(v, out_list)


def _out_digest(out):
    tensors = []
    _collect_tensors(out, tensors)
    if not tensors:
        return "NO_TENSORS:" + ",".join(sorted(out.keys())) if isinstance(out, dict) else "NO_TENSORS"
    h = _hl.sha256()
    for v in tensors:
        h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]

_skip_handles = []

def _install_skip():
    for _bn, _blk in model.named_modules():
        if _bn.endswith("swin_blocks") or not hasattr(_blk, "attn"):
            continue
        key = _bn.split("sttmultires_unet.")[-1]
        if any(key.startswith(sb) or key == sb for sb in _SKIP):
            def _zforward(x, mask=None, _blk=_blk):
                return torch.zeros_like(x)
            _skip_handles.append(_blk.attn.register_forward_hook(
                lambda mod, inp, out: torch.zeros_like(out)))
            print(f"[SKIPVERIFY] patched attn of {key}", flush=True)

def _run_pass(tag):
    _dig = []
    model.eval()
    _nb = 0
    with torch.set_grad_enabled(False):
        for _chunk, _mask, _label in valid_dataloader:
            if _nb >= 3:
                break
            _sjf.reset_net(model)
            _sjf.set_step_mode(model, config['data']['step_mode'])
            _chunk = _chunk.to(device=device, dtype=torch.float32)
            _mask = torch.unsqueeze(_mask.to(device=device), dim=1)
            _tv = globals().get("transform_valid", None)
            if _tv is not None:
                _chunk, _label, _mask = _tv((_chunk, _label, _mask.float()))
            if config['model']['encoding'] == 'cnt':
                if config['loader']['polarity']:
                    _chunk = _chunk.view([_chunk.shape[0], -1] + list(_chunk.shape[3:]))
            elif config['model']['encoding'] == 'voxel':
                if config['loader']['polarity']:
                    _neg = torch.nn.functional.relu(-_chunk)
                    _pos = torch.nn.functional.relu(_chunk)
                    _chunk = torch.cat((torch.unsqueeze(_pos, dim=2), torch.unsqueeze(_neg, dim=2)), dim=2)
            if config["model"]["norm_input"] == "minmax":
                _mn, _mx = torch.min(_chunk[_chunk != 0]), torch.max(_chunk[_chunk != 0])
                if not _mn == _mx:
                    _chunk[_chunk != 0] = (_chunk[_chunk != 0] - _mn) / (_mx - _mn)
            _out = model(_chunk)
            _dig.append(_out_digest(_out))
            _nb += 1
    print(f"[SKIPVERIFY] {tag}: {_nb} batches digests={_dig}", flush=True)
    return _dig

_dig_base = _run_pass("baseline")
_install_skip()
_dig_skip = _run_pass("skipped")
_bitexact = _dig_base == _dig_skip
print(f"[SKIPVERIFY] RESULT tag={TAG} bit_exact={_bitexact}", flush=True)
with open("/tmp/deadskip_verify_" + os.environ.get("DEADSKIP_TAG", "x") + ".json", "w") as _f:
    _json.dump({"tag": os.environ.get("DEADSKIP_TAG", "x"), "skip_blocks": _SKIP, "baseline": _dig_base, "skipped": _dig_skip,
                "bit_exact": _bitexact}, _f, indent=1)
for _h in _skip_handles:
    _h.remove()
raise SystemExit(0)
# ===== DEAD-ATTN SKIP VERIFY (ends) =====
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
        "--save_path", f"/tmp/deadskip_ckpt_{TAG}/checkpoint_epoch{{}}.pth",
        "--finetune", "1",
    ]

    h9._install_optional_mlflow_stub()
    os.chdir(baseline_root)
    source = h9._patch_source(baseline_entry.read_text(), baseline_entry)
    anchor = "    " + EPOCH_LOOP_ANCHOR
    assert source.count(anchor) == 1, "epoch-loop anchor not unique"
    block = "\n".join("    " + ln if ln.strip() else "" for ln in VERIFY_BLOCK.split("\n"))
    source = source.replace(anchor, block + "\n" + anchor, 1)
    code = compile(source, str(baseline_entry), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(baseline_entry)})


if __name__ == "__main__":
    main()
