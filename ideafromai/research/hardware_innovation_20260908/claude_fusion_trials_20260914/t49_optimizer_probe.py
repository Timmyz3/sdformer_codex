#!/usr/bin/env python3
"""T49-1c: 把 optimizer.state 里所有「标量参数」(shape=()) 的 exp_avg 列出来。

ATLIF 的 thresh 是 shape=() 的标量，全网只有这 105 个标量参数
（backbone 里没有标量）。所以标量 state 条目 ≡ ATLIF 阈值。
判读：exp_avg 若恰好为 0.0 ⇒ AdamW 的更新恒等于 lr*0/(0+eps)=0 ⇒ 阈值永不动。
"""
from collections import Counter

import torch

RUN = ("/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/"
       "H9_bipolar_self_attention/results/dsec_c12_alpha0125_ep29_resume5_20260830/")
opt = torch.load(RUN + "checkpoint_epoch34_state_dict.pth", map_location="cpu",
                 weights_only=False)["optimizer"]
state = opt["state"]

scalars = []
for pid, st in state.items():
    ea = st.get("exp_avg")
    if ea is None or ea.numel() != 1:
        continue
    scalars.append((float(ea.reshape(-1)[0]), float(st["exp_avg_sq"].reshape(-1)[0]),
                    float(st.get("step", -1))))

print("标量 state 条目（≡ATLIF thresh）: %d" % len(scalars))
nz = [s for s in scalars if s[0] != 0.0]
print("exp_avg 恰好为 0 的: %d / %d" % (len(scalars) - len(nz), len(scalars)))
if nz:
    print("非零 exp_avg 的取值范围: [%.3e, %.3e]" % (min(abs(s[0]) for s in nz),
                                                    max(abs(s[0]) for s in nz)))
print("exp_avg_sq: %s" % dict(Counter("%.3e" % s[1] for s in scalars).most_common(5)))
print("step: %s" % dict(Counter(int(s[2]) for s in scalars).most_common(5)))

print("\n前 15 个标量条目：")
for v, v2, step in scalars[:15]:
    print("   exp_avg=%-14.6e exp_avg_sq=%-14.6e step=%.0f" % (v, v2, step))

# 对照：非标量参数（backbone）的 exp_avg 一定非零
others = [st["exp_avg"] for st in state.values()
          if st.get("exp_avg") is not None and st["exp_avg"].numel() > 1]
print("\n对照（非标量参数）%d 个，exp_avg 全零的: %d"
      % (len(others), sum(1 for o in others if float(o.abs().max()) == 0.0)))
print("  非标量 |exp_avg| 中位数 = %.3e"
      % float(torch.stack([o.abs().mean().reshape(1) for o in others]).median()))
