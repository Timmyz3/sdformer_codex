# M419 — M418 三模式周期重放独立打铁

结论：`PASS`，评分 `94/100`，`P0=0, P1=0, P2=3`。M418 可准入为冻结 H67 ep35/no-running、S10 `zurich_city_09_a`、仅四个 bottleneck Conv3x3 的可执行周期仿真证据；不得升级为系统、全网、headline、RTL 实测、功耗、能耗或 paper-ready PPA。

## 独立复算

M419 没有 import 或执行 M418/M401/M397 聚合器。独立脚本解码冻结 M410R2 的 51,840,000 个 ordered row words，逐 phase 对照 M401 ledger 的 active/eligible/PWP/fallback、pass1/early matcher，并用 ledger 中的 used-center/run 与 tile0/tile1 narrow 统计重建三个时间轴。

- M401 phase 记录交叉核对：17,280 条，0 mismatch。
- M418 时间戳/组件记录逐字段核对：3 × 17,280 = 51,840 条，0 mismatch。
- 每条记录及全局 compute/scan/DMA/overlap/tail/commit 守恒：0 mismatch。
- M397、M401、M418 的 manifest 与外层 seal 全通过；`docs/359` SHA 仍为 `dedde7ce...`。

## 三模式结果

| 模式 | 基线强度 | Cycles | Candidate speedup |
|---|---|---:|---:|
| Dense16 same-resource | weak | 6,636,544,610 | 10.340668× |
| Exact zero-elided bit-sparse | strong | 742,148,386 | 1.156371× |
| M401 combined exact product reuse | candidate | 641,790,704 | 1.000000× |

10.340668× 与 1.156371× 必须分栏：前者相对弱 dense16，后者相对强 exact zero-elided。后者是可辩护的主性能比较；前者只能作为 dense 上下文，二者都只覆盖四个 H67 bottleneck Conv3x3。

## 资源与账本

三个模式遵守同一 `SHARED96/cmd32/L8/D8/tile2` 合同。Dense/zero 没有收取 candidate metadata，但没有漏掉公共工作：两者均保留 51,926,400 scan 请求周期、212,336,640B 权重请求、552,960 command 周期、6,635,520 data 周期、34,560 tail 与 960,000 commit。

Selected 精确保留 703,685,120B PWP、212,336,640B base weight、2,382,080 两-tile DMA command 周期、28,625,680 data 周期、67,912,100 matcher 周期、17,280 seal、276,480 L8 startup。Tile1 的 15,503,880 DMA 请求周期全部被 replay0 隐藏，exposed 为 0，与 M401 一致；没有借用 WIDE、systolic、跨 phase overlap 或免费 metadata。

## 负控与边界

一位 row payload 篡改、phase 缺失、timestamp +1、baseline candidate metadata 注入、公共 weight 省略、selected PWP 省略、used-pattern 变化以及 weak/strong namespace 互换均能触发检查。

剩余三个 P2 是范围而非数值缺陷：单序列十样本而非多序列/全网；cycle simulator 而非 end-to-end RTL measured latency；无物理 SRAM、布线与 SAIF/PTPX 功耗。因此 DATE 表中必须同时写明 `four H67 bottleneck Conv3x3 operators only` 与 `trace-cycle simulation`。

## 复跑

```bash
/opt/anaconda3/bin/python \
  hw_autoresearch_nts07/results/m419_m418_three_mode_independent_hammer_r1_20260826/independent_three_mode_recompute_m419.py \
  --repo-root /home/zhumd/work/sdformer_codex/SDformer \
  --contract hw_autoresearch_nts07/contracts/m419_m418_three_mode_independent_hammer_contract_r1_20260826.json \
  --output-json /tmp/m419_independent_recompute_receipt.json
```
