#!/usr/bin/env python3
"""T52: 用锚点的 spike_profile.json + checkpoint 的 ATLIF key 名单，
算出「自适应阈值能省多少 SOPS」的**上界**。

动机：用户认为「自适应阈值实打实降低了很多 sops」。这个说法值多少，取决于
**ATLIF 控制的神经元占全部脉冲的多少** —— 只有那部分的发放率能被 θ 压下去。
本脚本不动 GPU，纯 CPU 读两个已有产物。

口径：
  - ATLIF 模块路径 = checkpoint 里 `.spiking_neuron.` 之前的那个路径
    （installer 把 ATLIFTernaryPSN 装在该路径上，所以它与 profile 的层名同名）。
  - 死模块（`.sn2_q` / `.attn_sn`）排除（它们 spike 本来就恒为 0）。
  - SOPS 就是 total_spikes（锚点实测 synops_total == total_spikes == 72891240701）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ANCHOR = Path("/root/private_data/work/sdformer_codex/SDformer/hw_autoresearch_nts07/"
              "system_handoff/incoming/m2041_ep34_quant_binding_inputs")
CKPT = ANCHOR / "checkpoint_epoch34.pth"
PROFILE = ANCHOR / "spike_profile.json"


def main() -> None:
    prof = json.loads(PROFILE.read_text())
    rates = prof["layer_firing_rates"]
    total = float(prof["total_spikes"])

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))

    atlif_paths = set()
    for k in sd:
        if ".spiking_neuron." not in k:
            continue
        p = k.split(".spiking_neuron.", 1)[0]
        if p.endswith(".sn2_q") or p.endswith(".attn_sn"):
            continue
        atlif_paths.add(p)

    matched, unmatched = {}, []
    for p in atlif_paths:
        if p in rates:
            matched[p] = rates[p]
        else:
            # profile 用短名（去掉 sttmultires_unet. 前缀），按后缀兜底匹配
            cands = [n for n in rates if n.endswith(p) or p.endswith(n)]
            (matched.__setitem__(p, rates[cands[0]]) if len(cands) == 1 else unmatched.append(p))

    sp_atlif = sum(v["spikes"] for v in matched.values())
    el_atlif = sum(v["elements"] for v in matched.values())

    print("total_spikes(锚点)          = %d" % total)
    print("ATLIF 活模块数              = %d" % len(atlif_paths))
    print("在 profile 里匹配上的       = %d   (未匹配 %d)" % (len(matched), len(unmatched)))
    if unmatched:
        print("  未匹配样例:", unmatched[:5])
    print("ATLIF 控制的 spikes         = %d  (%.2f%% of total)" % (sp_atlif, 100.0 * sp_atlif / total))
    print("ATLIF 控制的 elements       = %d  (%.2f%% of total)" % (el_atlif, 100.0 * el_atlif / sum(v["elements"] for v in rates.values())))
    print("ATLIF 平均发放率            = %.4f" % (sp_atlif / el_atlif))
    print()
    print("逐模块（按 spikes 降序，前 25）：")
    print("%-70s %14s %10s" % ("layer", "spikes", "rate"))
    for name, v in sorted(matched.items(), key=lambda kv: -kv[1]["spikes"])[:25]:
        print("%-70s %14d %10.4f" % (name.replace("sttmultires_unet.", ""), v["spikes"], v["firing_rate"]))
    print()
    print("== 上界含义 ==")
    print("若把 ATLIF 控制的发放率整体压到 0，SOPS 最多降到 %.2f%%（即省 %.1f%%）；"
          % (100.0 * (1 - sp_atlif / total), 100.0 * sp_atlif / total))
    print("实际可及收益远小于此：θ 上去之后发放率会趋于一个非零平台。")


if __name__ == "__main__":
    main()
