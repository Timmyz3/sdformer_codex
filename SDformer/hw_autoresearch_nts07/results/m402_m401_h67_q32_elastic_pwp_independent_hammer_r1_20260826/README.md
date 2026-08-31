# M402 — M401 H67 q32 elastic-PWP 独立全量打铁

结论：`PASS`，评分 `93/100`，`P0=0, P1=0, P2=6`。接受 M401 的 frozen-H67 四-Conv trace-cycle 结果；combined 点通过冻结的 1.15 门，可进入新的 q32 elastic-PWP selected RTL/VCS/Synopsys。当前仍不准入 RTL 实测、系统加速、能耗或 headline。

## 独立性与范围

独立脚本不 import M401、M397、M394 或 M43，也没有读取任何候选 CSV。它从 M40 80 payload、M338 q128 nested center 和 M41 四个 INT8 权重重新构造：

- 442,368 个静态 PWP blocks、42,467,328 lanes；
- 17,280 个 runtime phases、51,840,000 source rows；
- q16/q32/q48/q64/q80/q96/q112/q128 eligible first-exact prefix；
- q32 lowest-ID、严格 fallback、exact residual、used-center runs；
- tile0/tile1 narrow descriptor、DMA、overlap、四 variant sweep 和 blocking。

另生成独立逐-phase 账本 `m402_per_phase_independent_ledger.csv`，不是 M401 CSV 的副本或输入。

## 核心结果

| Variant | cmd32/L8 cycles | Speedup |
|---|---:|---:|
| M397 anchor | 669,012,336 | 1.1093194341× |
| Elastic only | 645,542,312 | 1.1496510333× |
| Early only | 665,260,728 | 1.1155752245× |
| Combined | 641,790,704 | 1.1563713550× |

共同 baseline 为 742,148,386 cycles。Combined 低于 645,346,422-cycle ceiling，超过 1.15×，余量 3,555,718 cycles。

runtime PWP rows 为 16,971,357；tile0/tile1 narrow descriptor 分别为 11,943,449 和 12,643,363，总计 24,586,812，占 135,770,856 个 PWP block descriptors 的 18.1090%。q16 eligibility-clean exact hit 节省 3,751,608 个 matcher prefix tasks。

贡献分解完全闭合：elastic 减少 24,586,812 compute cycles，但增加 17,280 config cycles 和 1,099,508 exposed tile0 DMA cycles，净减 23,470,024；early-hit 再减 3,751,608，合计相对 anchor 减少 27,221,632 cycles。

全部 16 个 sweep 点 tile1 exposed cycles 为 0；cmd32/L8 combined 的最小 tile1 隐藏余量仍有 1,166 cycles。

## 风险边界

性能余量很薄。1.15 门最多容忍平均约 0.0651098 blocking cycle/replayed descriptor；0.25 sensitivity 已降到 655,443,488 cycles、1.1322843229×。因此 selected RTL 的关键不是再报 simulator 数字，而是证明 narrow decoder、D8 FIFO、SRAM response 与 backend 几乎无气泡。

640B fixed stride 还使两-tile PWP traffic 从 633,316,608B 增至 703,685,120B；当前模型中 tile1 被重叠隐藏，不代表物理 DMA/SRAM 免费。

## 复跑

```bash
mkdir -p /tmp/m402_replay
/opt/anaconda3/bin/python \
  hw_autoresearch_nts07/results/m402_m401_h67_q32_elastic_pwp_independent_hammer_r1_20260826/independent_full_recompute_m402.py \
  --repo-root /home/zhumd/work/sdformer_codex/SDformer \
  --contract hw_autoresearch_nts07/contracts/m401_h67_q32_elastic_pwp_full_replay_contract_r1_20260826.json \
  --candidate hw_autoresearch_nts07/results/m401_h67_q32_elastic_pwp_full_replay_r1_20260826/m401_h67_q32_elastic_pwp_full_replay_r1.json \
  --output-dir /tmp/m402_replay
```

脚本 fail-closed。`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce…`。
