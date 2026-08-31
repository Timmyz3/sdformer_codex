# M398 — M397 H67 fixed-product q/O 独立打铁评审

结论：`PASS`，评分 `95/100`，`P0=0, P1=0, P2=5`。接受 M397 DSE 及其负结论：关闭 fixed-product q/O 扩展性能轴，保留已经验证的 q32/O4 实现锚点，不为 q16、q64 或 q128 新开 selected RTL。

## 独立重算

独立脚本不 import M397、M394、M381、M339 或 M43。它从 M40 的 80 个 payload 直接重建 valid-pad Conv3x3 support，完成四档共 69,120 次 phase evaluation，并独立计费 nested prefix、最低 ID tie、严格 `1+d<popcount`、exact fallback、serial16 passes、`ceil(q/32)` seal、32B stride padding、maximal center runs、cmd32/L8、8/O replay 与 D8 事务。

| q/O | candidate cycles | speedup | serial16 extra passes | seal cycles | replay count |
|---|---:|---:|---:|---:|---:|
| 16/8 | 676,931,968 | 1.0963411703× | 0 | 1 | 17,280 |
| 32/4 | 669,012,336 | 1.1093194341× | 1 | 1 | 34,560 |
| 64/2 | 677,482,456 | 1.0954503389× | 3 | 2 | 69,120 |
| 128/1 | 730,375,445 | 1.0161190263× | 7 | 4 | 138,240 |

四点与全部 48 个 sweep、16 个 blocking 点共享并重现同一个 baseline `742,148,386` cycles。q32/O4 的人口、popcount-one fallback 和 robust 点与 M394 完全一致。

q128 虽将 PWP rows 提高到 18,886,324，但需要 7 次额外 serial16 pass、4 个 seal word、8 次 output replay，并把每个 144B useful PWP slot 按 32B 对齐成 160B；总 padding 为 254,821,376B，另暴露 874,769 个 next-tile DMA cycles。因此更高命中率没有变成更高性能。

## D8/L8 账本

四档 request、response、bundle 数完全守恒：27,305,568 / 54,611,136 / 109,222,272 / 218,444,544。固定 L8 startup 按每次 replay 收费。D8 与 L8 相等，实际 descriptor service 下界分别为 8/4/2/1 cycles；最快的一周期流在首次响应后仍可连续服务，较慢服务只增加缓冲，不获得免费吞吐。

## 边界

这是 fixed-product/common-port 的 trace-cycle DSE，不是 equal-area 比较。没有 selected RTL、full real-trace VCS cycle match、物理 SRAM、SPEF、SAIF/PTPX、系统 speedup 或 DATE headline 准入。运行时仍只有 `zurich_city_09_a` 十样本，跨 sequence/density 泛化尚未证明。

## 复跑

```bash
/opt/anaconda3/bin/python \
  hw_autoresearch_nts07/results/m398_m397_h67_fixed_product_qo_independent_hammer_r1_20260826/independent_recompute_m398.py \
  --repo-root /home/zhumd/work/sdformer_codex/SDformer \
  --contract hw_autoresearch_nts07/contracts/m397_h67_fixed_product_qo_finite_dse_contract_r1_20260826.json \
  --candidate hw_autoresearch_nts07/results/m397_h67_fixed_product_qo_finite_dse_r1_20260826/m397_h67_fixed_product_qo_finite_dse_r1.json \
  --receipt /tmp/m398_independent_recompute_receipt_r1.json
```

脚本 fail-closed，不覆盖已有 receipt。`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce…`。
