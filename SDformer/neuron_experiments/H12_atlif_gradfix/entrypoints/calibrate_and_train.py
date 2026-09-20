#!/usr/bin/env python3
"""H12 v2: ITERATIVE per-layer 2^k theta calibration + ternary warm-start training.

v1 教训（2026-09-18）：一次性用旧网络（θ=1.0 大面积死）的膜电位分位数定 θ，
全网同时激活后下游膜电位暴涨 → θ 全部偏低 → 发放饱和（eval firing 0.83，
AEE 7.78 vs 基线 1.33）。饱和区 5 epoch 爬不出来。

v2 修复：闭环迭代——每轮设 θ → 前向实测 firing r → 幂次校正
    exp_l += round(log2(max(r_meas, eps) / r_target))
（log2 域等分搜索；θ 恒为 2 的幂）。只更新有前向调用的模块；
不可达模块（结构性死 sn2_q）保持初值、不参与校正。

Usage:
  python calibrate_and_train.py --config CFG --prev-runid CKPT --save-path FMT \
      [--calib-batches 30] [--r-target 0.25] [--calib-rounds 5] [--theta-out PATH]
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_PY = HERE / "train.py"

EPOCH_LOOP_ANCHOR = 'for epoch in range(epoch_initial, config["loader"]["n_epochs"]):'

CALIB_BLOCK = '''
# ===== H12 THETA CALIBRATION v2 (iterative, begins) =====
import collections as _collections
import json as _json
import math as _math
from spikingjelly.activation_based import functional as _sjf
from models.STSwinNet_SNN.atlif_ternary_psn import ATLIFTernaryPSN as _ATP

_cal_rounds = int(os.environ.get("H12_CALIB_ROUNDS", "5"))
_cal_n = int(os.environ.get("H12_CALIB_BATCHES", "30"))
_cal_r = float(os.environ.get("H12_CALIB_RTARGET", "0.25"))
_cal_out = os.environ.get("H12_CALIB_OUT", "/tmp/h12_theta_table.json")
print(f"[H12CAL] v2 iterative calibration: rounds={_cal_rounds} batches/round={_cal_n} r_target={_cal_r}", flush=True)

_cal_mods = [(n, m) for n, m in model.named_modules() if isinstance(m, _ATP)]
print(f"[H12CAL] ATLIFTernaryPSN modules: {len(_cal_mods)}", flush=True)

_cal_exp = {}
_cal_reached = {n: False for n, _ in _cal_mods}


def _cal_prep(chunk, label, mask):
    chunk = chunk.to(device=device, dtype=torch.float32)
    label = label.to(device=device, dtype=torch.float32)
    mask = torch.unsqueeze(mask.to(device=device), dim=1)
    _tv = globals().get("transform_valid", None)
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
    return chunk, label, mask


def _cal_forward_passes(n_batches):
    model.eval()
    _nb = 0
    with torch.set_grad_enabled(False):
        for _chunk, _mask, _label in valid_dataloader:
            if _nb >= n_batches:
                break
            _sjf.reset_net(model)
            _sjf.set_step_mode(model, config['data']['step_mode'])
            _chunk, _label, _mask = _cal_prep(_chunk, _label, _mask)
            _ = model(_chunk)
            _nb += 1
    return _nb


def _cal_apply_theta():
    for _n, _m in _cal_mods:
        with torch.no_grad():
            _m.thresh.data.fill_(float(2.0 ** _cal_exp[_n]))


def _cal_r_hooks():
    _acc = {n: [] for n, _ in _cal_mods}
    _hs = []

    def _mk(n):
        def _h(mod, inp, out):
            _acc[n].append(mod.r)
        return _h

    for _n, _m in _cal_mods:
        _hs.append(_m.register_forward_hook(_mk(_n)))
    return _acc, _hs


# ---- round 0: quantile init from current (loaded) network ----
_cal_res = {n: [] for n, _ in _cal_mods}
_cal_orig = {}


def _cal_mk(n, orig):
    def _rec(h, *a):
        flat = h.detach().flatten()
        if flat.numel() > 0:
            k = min(flat.numel(), 2048)
            idx = torch.randint(0, flat.numel(), (k,), device=flat.device)
            _cal_res[n].append(flat[idx].float().cpu())
        return orig(h, *a)
    return _rec


for _n, _m in _cal_mods:
    _cal_orig[_n] = _m.act
    _m.act = _cal_mk(_n, _cal_orig[_n])

_nb0 = _cal_forward_passes(_cal_n)
for _n, _m in _cal_mods:
    _m.act = _cal_orig[_n]
print(f"[H12CAL] round0 quantile init from {_nb0} batches", flush=True)

for _n, _m in _cal_mods:
    _vals = torch.cat(_cal_res[_n]) if _cal_res[_n] else torch.tensor([1.0])
    if _vals.numel() > 100000:
        _vals = _vals[torch.randperm(_vals.numel())[:100000]]
    _q = torch.quantile(_vals, 1.0 - _cal_r).item()
    _q = max(_q, 1e-3)
    _e = max(-8, min(4, int(round(_math.log2(_q)))))
    _cal_exp[_n] = _e

# ---- iterative closed-loop rounds ----
for _rd in range(_cal_rounds):
    _cal_apply_theta()
    _acc, _hs = _cal_r_hooks()
    _nb = _cal_forward_passes(_cal_n)
    for _h in _hs:
        _h.remove()
    _updates = 0
    for _n, _m in _cal_mods:
        if not _cal_reached[_n] and len(_acc[_n]) > 0:
            _cal_reached[_n] = True
        if len(_acc[_n]) == 0:
            continue  # 不可达模块（结构性死参数）不参与校正
        _r = sum(_acc[_n]) / len(_acc[_n])
        if _r < 1e-6:
            _de = 1  # 完全不发放：降一档阈值
        else:
            _de = int(round(_math.log2(max(min(_r / _cal_r, 64.0), 1.0 / 64.0))))
        if _de != 0:
            _cal_exp[_n] = max(-8, min(4, _cal_exp[_n] + _de))
            _updates += 1
    _mean_r = sum(sum(v) / max(1, len(v)) for v in _acc.values()) / max(1, len(_acc))
    print(f"[H12CAL] round {_rd}: batches={_nb} mean_r={_mean_r:.4f} updated={_updates}", flush=True)
    if _updates == 0:
        break

_cal_apply_theta()
_theta_table = {n: {"exp": e, "theta": float(2.0 ** e), "reached": _cal_reached[n]} for n, e in _cal_exp.items()}
with open(_cal_out, "w") as _f:
    _json.dump(_theta_table, _f, indent=1)
_th = [float(2.0 ** e) for e in _cal_exp.values()]
print(f"[H12CAL] wrote {_cal_out} ({len(_theta_table)} modules)", flush=True)
print(f"[H12CAL] theta: min={min(_th)} max={max(_th)} histogram={dict(_collections.Counter(_th))}", flush=True)

# ---- final verification on 2 batches ----
_acc, _hs = _cal_r_hooks()
_vb = _cal_forward_passes(2)
for _h in _hs:
    _h.remove()
_rs = {n: (sum(v) / max(1, len(v))) for n, v in _acc.items() if len(v) > 0}
_all_r = list(_rs.values())
_dead = sum(1 for n in _cal_reached if _cal_reached[n] and (_rs.get(n, 0.0) < 0.001))
print(f"[H12CAL] firing after v2 calibration: mean={sum(_all_r)/max(1,len(_all_r)):.4f} "
      f"min={min(_all_r):.4f} max={max(_all_r):.4f} dead(reached&r<0.001)={_dead} unreachable={sum(1 for x in _cal_reached.values() if not x)}", flush=True)

_sjf.reset_net(model)
model.train()
print("[H12CAL] v2 calibration done, entering training loop", flush=True)
# ===== H12 THETA CALIBRATION v2 (ends) =====
'''


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prev-runid", required=True)
    ap.add_argument("--save-path", required=True)
    ap.add_argument("--calib-batches", type=int, default=30)
    ap.add_argument("--r-target", type=float, default=0.25)
    ap.add_argument("--calib-rounds", type=int, default=5)
    ap.add_argument("--theta-out", default="")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("h12_train_module", TRAIN_PY)
    h12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h12)

    repo_root = h12._repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = TRAIN_PY.parents[1] / "overlay"  # H12's FIXED overlay
    baseline_entry = baseline_root / "train_flow_parallel_supervised_SNN.py"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))
    sys.argv = [
        str(baseline_entry),
        "--config", str(Path(args.config).resolve()),
        "--prev_runid", str(Path(args.prev_runid).resolve()),
        "--save_path", str(Path(args.save_path).resolve()),
        "--finetune", "1",
    ]

    os.environ.setdefault("H12_CALIB_BATCHES", str(args.calib_batches))
    os.environ.setdefault("H12_CALIB_RTARGET", str(args.r_target))
    os.environ.setdefault("H12_CALIB_ROUNDS", str(args.calib_rounds))
    if args.theta_out:
        os.environ.setdefault("H12_CALIB_OUT", str(Path(args.theta_out).resolve()))

    h12._install_optional_mlflow_stub()
    os.chdir(baseline_root)
    source = h12._patch_source(baseline_entry.read_text(), baseline_entry)
    anchor = "    " + EPOCH_LOOP_ANCHOR
    assert source.count(anchor) == 1, "epoch-loop anchor not unique in baseline source"
    block = "\n".join("    " + ln if ln.strip() else "" for ln in CALIB_BLOCK.split("\n"))
    source = source.replace(anchor, block + "\n" + anchor, 1)
    code = compile(source, str(baseline_entry), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(baseline_entry)})


if __name__ == "__main__":
    main()
