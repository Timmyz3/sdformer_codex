# M24 temporal-cohort 与 resident-fusion 严格 DSE

## 结论

M24 完成了 T=10 coefficient cohort 的 bit-exact 样本 DSE，但**没有通过全网
headline 门槛**。冻结的 P4 packed tiles 对其中每一条记录都是 exact bitmap，却每次
operator call 只采 4 个 row pair；它覆盖 H67 的 31 个、Local ep44 的 36 个 exact
operator 名称，不是所要求的 57/79 eligible coefficient 全量 census。

按 `dual_line_operator_trace.csv` 的 aggregate exact work 作分母，当前 packed bitmap
的 exact coefficient 覆盖只有：

| 身份 | 路线 | exact coefficient coverage | fallback |
|---|---|---:|---:|
| H67 ep35 | Local | 0.147954% | 99.852046% |
| H67 ep35 | Motion shared | 0.142672% | 99.857328% |
| Local ep44 | Local | 0.226021% | 99.773979% |
| Local ep44 | Motion shared | 0.221003% | 99.778997% |

这远高于 `<5% fallback` 上限，所以产物状态为
`GAP_EXACT_COEFFICIENT_COVERAGE_LT95_PERCENT_NO_HEADLINE`。算子名覆盖与 coefficient
work 覆盖已被分开，禁止用前者替代后者。

## Temporal-cohort 样本内精确结果

对每个 `(sample, operator call, row, chunk, weight group)`，脚本直接解包冻结 NPZ，验证：

- `previous[t] == current[t-1]`，T0 previous 为零；
- valid tail 全零；
- current、positive 和 negative popcount 与 CSV 完全相等；
- Local 用 T10 presence mask；Motion shared 按实际 row selector 生成互斥的 positive/
  negative signed mask；
- destination update 总数不变，cohort 只减少重复 coefficient vector read。

样本内 coefficient read 降幅和 operation envelope 如下：

| 身份 | 路线 | coefficient read 降幅 | read+update 串行 operation envelope | read/update 完全重叠 envelope |
|---|---|---:|---:|---:|
| H67 ep35 | Local | 50.1155% | 1.33436x | 1.00000x |
| H67 ep35 | Motion shared | 46.8872% | 1.30623x | 1.00000x |
| Local ep44 | Local | 49.7877% | 1.33145x | 1.00000x |
| Local ep44 | Motion shared | 47.2646% | 1.30945x | 1.00000x |

两种 envelope 都给了相同资源 baseline：1 个 coefficient-read path、1 个 destination
update path、相同 resident capacity。前者是标量操作相加，后者是两个 stage 完全重叠；
它们都不是硬件 cycle。更强的 composable baseline 可以用相同容量做 generic coefficient-
resident cache，并获得相同 cohort read 数；相对这个 strongest same-resource baseline，
traffic 与 operation envelope 都是 **1.000x**。所以表中 1.30--1.33x 只允许称为
step-major implementation ablation，不能称为新架构优势。

样本中最大的 payload-only cohort resident 容量为 H67 3,140,478 bit、Local ep44
5,658,818 bit；它含 coefficient cache、稀疏 mask/index control 与 Acc32，未加
ECC/tag/banking，不能当作 paper SRAM macro 容量。

## M4→M21→ATLIF 边界

M18 与 M21 的 13 个 producer 名称一一相等，冻结 H67 checkpoint 身份一致；13 条路径
合计 552,960,000 个 FP32 元素。每条路径均跨越 `no_running` dynamic BN 的全局统计
barrier。M21 又明确要求 moment work 在系数生成前 drain，并保留：

1. producer unnormalized output/Acc32 write；
2. barrier 后 fused BN+ATLIF replay read。

canonical M22 已经按这一个最强 two-movement baseline 记账。因此 M24 相对 M22 严格可删
transaction 数为 **0**、可删 byte 为 **0**，相对 strongest composable baseline 为
**1.000x**。M22 的 `serialized_byte_service_ticks` 只原样封存为 logical transport ticks，
从未用于 cycle、latency 或 speedup。

M21 的 3-tile FIFO40 payload 为 122,880 bit，最大 moment state 为 187,392 bit；二者之和
310,272 bit 也只是 payload bookkeeping，不是物理 macro sizing。M7 仍是 premacro
logic slice，不能提升这个边界。

## 2x 门槛审计

旧账中的 eligible system fraction `0.573719` 重新代入 Amdahl，达到 2x 所需 eligible
engine speedup 是 `7.782512x`，不是旧冻结的 `7.687553x`。反推后者隐含 eligible
fraction `0.574765763`，两者相差 0.104676 percentage point。M24 不替这两个数任选其一，
而是将 threshold consistency 标为 NO-GO，等待 cycle-defined coverage ledger 统一。

即便把当前 exact 样本的串行 operation envelope 乐观套用到旧 coverage，假想系统结果也
只有约 1.155x--1.168x；由于 exact coefficient fallback 超标且 operation 不是 cycle，
这些数只保留为 what-if，不是性能主张。

## 缺口合同与复现

缺口合同要求增加 streaming `--dual-line-cohort-census-dir`：不保存全网激活，而是在 GPU
运行中精确累积 T10 Local 1024-bin presence mask、Motion positive/negative mask，以及
operator/source chunk/fanout 的 coefficient/update conservation。现有 P128 v2 命令只能改善
分层样本，仍不能关闭 exact headline。远端诊断命令和 producer patch 后命令均已写入：

- `contracts/m24_exact_temporal_bitmap_gap_contract_r1_20260822.json`

本地复现：

```bash
python3 hw_autoresearch_nts07/system_simulator/scripts/analyze_m24_temporal_cohort_resident_fusion.py \
  --repo-root . \
  --input-contract hw_autoresearch_nts07/contracts/m24_temporal_cohort_input_contract_r1_20260822.json \
  --input-contract-sha256 4dc18567204b07bb8d9bdc3949c91b69e99b61868a4ee7fb63465f83f8bf0429 \
  --gap-contract hw_autoresearch_nts07/contracts/m24_exact_temporal_bitmap_gap_contract_r1_20260822.json \
  --output-dir <new-output-directory>

python3 -m unittest -v \
  hw_autoresearch_nts07/system_simulator/tests/test_m24_temporal_cohort_resident_fusion.py
```

Python 3.6 compile 与测试 8/8 PASS。canonical output 的 `m24_evidence.sha256` 4/4
复核通过。主 JSON SHA256 为
`fd1e1e437489e98c6be6714449fd3f767195485cd5e3dd39b9a284d51dfbaedc`，目录为
`results/m24_temporal_cohort_resident_fusion_r3_strongbaseline_failclosed_20260822`。r1/r2
是加固 manifest、signed-update 与 strongest composable baseline 前的本地草案，不作为
canonical output。

## 声明边界

允许声称 frozen sampled rows 内 bit-exact T10 mask、coefficient read/update/control/capacity
DSE、M22 额外边界删除为零、以及 Amdahl 常数不一致。禁止声称 57/79 全网复用、系统
cycle/FPS/energy/speedup、M22 ticks 是 cycles、P4 样本外推、resident fusion 相对最强
two-movement baseline 有加速，或物理 SRAM/PPA 已完成。
同样禁止把 step-major ablation 的 coefficient read 降幅写成相对 generic resident cache 的
创新收益。
