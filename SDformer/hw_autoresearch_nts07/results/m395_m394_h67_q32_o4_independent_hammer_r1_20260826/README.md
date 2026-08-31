# M395 — M394 H67 q32/O4 独立打铁评审

结论：`P0=0, P1=0, P2=5`，评分 `94/100`。M394 的 H67 ep35/no-running 四个 bottleneck Conv3x3 周期估计可以接受，并可进入 M393 H67 real-trace RTL cycle miter；它仍不是 RTL 实测、系统加速、能耗、paper-PPA 或 DATE headline。

## 独立重算结果

独立脚本不 import M394、M381、M339 或 M43。它直接解包 M40 的 little-endian support plane，显式重建 valid-pad Conv3x3 taps，独立完成 q32 最近模式选择、精确 residual 重建和周期递推。

- 重哈希 80 个 payload，共 42,346,309 B；重放 40 个 changed plane。
- 完整覆盖 10 samples × 4 operators × 432 partitions = 17,280 phases，共 51,840,000 source rows。
- 人口 exact match：24,534,432 zero；27,305,568 active；16,971,357 PWP；10,334,211 fallback；7,516,420 popcount-one fallback。
- 16,971,357 个 PWP row 全部精确重建，逐样本与全人口 mismatch 均为 0。
- 38,891,950 个存在最近中心并列的 row 验证最低索引优先；329,218 个等价成本 row 验证严格 `1+distance < popcount`，全部 fallback。
- 28 个 command/SRAM sweep 点和 6 个 blocking sensitivity 点与 M394 全部精确一致。

cmd32/SRAM-L8 robust 点独立重现：baseline `742,148,386` cycles，candidate `669,012,336` cycles，四-Conv module speedup `1.1093194341337227×`，超过冻结的 `1.05×` M393 准入线。

## 身份与边界

运行时是 checkpoint `4f33e086…`、BN `no_running` 的 H67 ep35 M40 S10。M73/M77 catalog 来源是同一 checkpoint 的 train-only 人口，`test_or_validation_data_used=false`、与 valid825 key overlap 为 0，且 `paft_catalog=false` / `paft_checkpoint=false`。M381 的 `1.0763828768×` 仅作分离人口对照，没有把 PAFT rows/cycles 混入 H67。

主要剩余缺口是 H67 real-trace 尚未驱动 RTL；matcher、DMA、descriptor SRAM、O4 backend 与 tail cycles 仍是行为模型。其次，bit-sparse baseline 没有 matched-frequency RTL，物理 SRAM、SPEF、SAIF/PTPX、系统 Amdahl 和跨 sequence/density 代表性也尚未闭合。

## 复跑

```bash
/opt/anaconda3/bin/python \
  hw_autoresearch_nts07/results/m395_m394_h67_q32_o4_independent_hammer_r1_20260826/independent_recompute_m395.py \
  --repo-root /home/zhumd/work/sdformer_codex/SDformer \
  --contract hw_autoresearch_nts07/contracts/m394_h67_ep35_q32_o4_burst_streaming_contract_r1_20260826.json \
  --candidate hw_autoresearch_nts07/results/m394_h67_ep35_q32_o4_burst_streaming_r1_20260826/m394_h67_ep35_q32_o4_burst_streaming_r1.json \
  --receipt /tmp/m395_independent_recompute_receipt_r1.json
```

复跑脚本 fail-closed，不覆盖既有 receipt。`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce…`。
