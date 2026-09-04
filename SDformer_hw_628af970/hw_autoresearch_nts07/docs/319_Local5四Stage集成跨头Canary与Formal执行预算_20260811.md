# Local5 四 Stage 集成跨头 Canary 与 Formal 执行预算

> 日期：2026-08-11  
> 证据范围：`[rtl]`、`[软件整数金参考]`、`[prof]`、`[rtl校准模型]`  
> 当前裁决：四种真实 head 宽度的单窗数值 smoke 全部通过；旧结果未覆盖真实
> stage/block/window tag；`formal G0 = DENY`

## 1. 本轮目标

H=3 的集成 canary 已证明单个 stage0 窗口能够从正式 descriptor 走到最终跨头
Acc32，但不能覆盖 Local5 fullres 四种头数的参数边界。本轮只回答以下问题：

1. 同一套集成 RTL 在 H=6、H=12、H=24 时是否仍能完成；
2. head/output-tile 索引、权重矩阵和跨头 RMW 是否越界或错序；
3. 最终 Acc32 是否逐项等于独立软件整数金参考；
4. 全量 formal 应采用什么分片、存储和双模拟器策略。

这不是新架构 RTL，也不是性能评估。

## 2. 验证数据流

四个 stage 使用同一条数据流：

```text
formal descriptor
  -> Q/K/invalid-mask 反演
  -> Q7 score + masked integer Shiftmax5 Q1.7
  -> relation transpose
  -> source-major gate/lane term
  -> checkpoint INT8 projection weight
  -> H 个 input head 到 H 个 output tile 的 DUT 内 RMW
  -> tile-major final Acc32
```

软件 expected 直接消费 producer destination-major `item_*`，不经过 descriptor
方向反演；RTL actual 从 descriptor 重建输入并重新计算 score/Shiftmax5。两条路径在
最终 Acc32 才相遇。

## 3. 四种头数结果

| 拟对应 stage | H | final Acc32 | mismatch | max abs error | Verilator 周期 | 仿真墙钟 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 43,200 | 0 | 0 | 695,572 | 3.04 s |
| 1 | 6 | 86,400 | 0 | 0 | 2,273,944 | 14.68 s |
| 2 | 12 | 172,800 | 0 | 0 | 8,333,761 | 37.77 s |
| 3 | 24 | 345,600 | 0 | 0 | 31,212,395 | 266.30 s |

结果目录：

- H3：`results/local5_erep_integrated_cross_head_canary_v4_final_20260811`；
- H6：`results/local5_erep_integrated_stage1_h6_smoke_20260811`；
- H12：`results/local5_erep_integrated_stage2_h12_smoke_20260811`；
- H24：`results/local5_erep_integrated_stage3_h24_smoke_v2_20260811`。

H3 同时通过 Icarus 与 Verilator/SVA，两个模拟器的 43,200 个 actual Acc32 SHA
一致，并有 source bundle、工具、命令和 executable 绑定。H6/H12/H24 是
Verilator 单模拟器 smoke，未达到 H3 的 provenance 等级。

独立复审发现旧 TB 将 executor 的 stage/block/window 输入硬连为 0。因此这些结果
使用了各 stage 的真实 descriptor/权重和真实 H，但 RTL 控制 tag 仍是 stage0/block0/
window0。它们只能证明 H={3,6,12,24} 的容量、地址和 HxH 归约边界，不能证明四个
stage 或 12 个 block 的控制闭环。

## 4. 能证明和不能证明的内容

### 4.1 已证明 `[rtl]`

- 正式 manifest 对应的 H={3,6,12,24} 四种头数容量均可执行；
- `HEADS == OUTPUT_TILES` 的全 H×H 跨头投影在最大 H=24 时未越界；
- H24 使用 589,824 个 INT8 权重，输出 345,600 个 pre-bias/pre-requant Acc32，
  逐项零失配；
- 固定随机 service 下 assertion 未触发；
- 当前 TB 对 head/output-tile 参数、输入索引和权重索引进行显式范围检查。

### 4.2 尚未证明 `[待验证]`

- 真实 stage/block/window tag 和 12-block 控制覆盖；
- 1,200 个窗口和 100 个 sample 的全量数值等价；
- 每个窗口的 phase ledger 完整性；
- 随机反压种子和多个真实窗口的覆盖；
- H6/H12/H24 的 Icarus 交叉模拟器一致性及完整 provenance；
- Local5 EREP 性能、full encoder 吞吐、ASIC 面积/功耗/时序。

因此四级 smoke 只关闭“头数/容量参数边界”风险，不能放行 formal G0。

当前数值边界由 checkpoint projection contract 明确冻结为 theta-folded、逐输出通道
dyadic INT8 权重的 `pre-bias/pre-BN/pre-requant/pre-residual Acc32`。bias 确实存在，
但不在本 miter 边界内；不能把该结果称为完整部署输出或网络逐 bit 等价。

## 5. 软件金参考优化

原 expected 对每个 producer item 和每个 H×H tile 逐项循环，H24 单窗超过数分钟。
本轮把每个 destination 的 term 精确聚合为 `[T450,32]` 整数系数，再将所有 input
head 拼成 `[T450,H*32]`，最后用一次 INT64 矩阵乘得到所有 output tile。

stage0 新旧路径比较结果：

- 数组逐项相同；
- mismatch=0；
- 生成的 NPZ SHA 与旧路径相同。

H24 新路径耗时 `59.13 s`、峰值 RSS `242,816 KiB`。这是 formal expected 生成器的
运行预算，不是硬件性能。

## 6. 验证基础设施回归

合并执行 formal preflight、archive replay、ledger replay、来源隔离 canary 和集成
cross-head canary 单测：

```text
37 tests / 37 PASS
```

Verilator lint 返回 0。仍保留已有 warning：两个未使用 memory-implementation 参数，
以及 retirement scheduler 的 stripe 状态未驱动；这些路径不是本轮新增，正式 G0
runner 仍需将 warning 列入 receipt，不能静默隐藏。

## 7. 全量 Archive 规模

正式 profile 的确定规模为：

| 项 | 数量 |
|---|---:|
| sample | 100 |
| joint window | 1,200 |
| input-head group | 13,800 |
| H×H task | 210,600 |
| frozen phase | 462,600 |
| final Acc32 | 198,720,000 |
| 若保存全部 partial Acc32 | 3,032,640,000 |

只保存最终 Acc32 时，一份纯 payload 下限为 `0.740 GiB`；expected 与一份 Verilator actual
合计 `1.481 GiB`。保存全部 partial 会产生 `11.297 GiB`/份，且不是 admission 所需，
因此禁止物化。

## 8. 运行时间口径

旧 H3 探针按 H² task 数保守外推：

- Verilator 全量约 `19.76 h`；
- Icarus 全量约 `354.25 h`。

这是 `[rtl校准模型]`，没有计入不同 stage、I/O、编译和分片启动差异。按本轮四种
H 单窗实测与每 sample 的 block 数加权，Verilator 约为 `22.07 h/100 samples`；它
仍未包含 expected/vector/I/O 开销，也只能用于资源预留，不能写成论文性能。

## 9. Formal Runner 决策

正式执行采用以下 fail-closed 设计：

1. 100 个 sample shard，每个 shard 对应一个真实 sample，可断点续跑；
2. 每个 shard 独立生成 expected、运行 Verilator actual、只读 miter；
3. 每个 shard 只保存 final Acc32、task/phase ledger、source/tool/command/executable SHA；
4. shard 完成标志必须最后原子写入，缺失、重复、SHA 改变或 mismatch 均拒绝；
5. 全局 admission 只读取 100 个闭合 receipt，并复核 1,200 window、462,600 phase、
   198,720,000 Acc32 的精确计数；
6. Icarus 不跑全量，只在 H3/H6/H12/H24 做分层交叉模拟器 canary。

启动 100 sample 前先完成一个 sample 的全流程试运行，验证断点恢复和 archive
合并合同。

## 10. DATE 证据边界

本轮对 DATE 的作用是提高 Local5 系统完整度和 RTL 可信度，不构成新的架构贡献。
允许表述：

> Local5 的 attention-to-projection pre-bias Acc32 数据流已在四种真实 head 宽度上
> 通过单窗 RTL 与独立整数金参考逐项比对；真实 12-block 控制 tag 尚待重跑。

禁止表述：

- “Local5 formal 已通过”；
- “完成了 full encoder RTL”；
- “31,212,395 cycle 是部署延迟”；
- “smoke 证明了 EREP 性能收益”；
- “Verilator 时间是 ASIC 吞吐”；
- “当前结果是 DC/ASIC PPA”。

当前裁决：

```text
formal G0 = DENY
EREP candidate RTL = DENY
下一步 = 单 sample 正式 shard runner 试运行
```

## 11. 独立 DATE 复审与整改

独立审稿人对本包给出 `3/5 Major Revision`，其中 H3 为 `4/5`，H6/H12/H24 为
`2.5/5 [rtl smoke]`。有效意见及处理如下：

1. 旧 TB stage/block/window 硬连 0：成立。已参数化并加入 stage-to-H、block、window
   范围检查，actual 日志和 receipt 必须匹配 task plan；待同版本重跑；
2. expected 依赖 manifest 行顺序：成立。已改为按 head canonical key 建表并拒绝
   重复，新增乱序/重复单测；
3. bias/scale 边界不清：合同本身已声明 scope，但本文表述不够明确。现统一为
   `theta-folded INT8 pre-bias/pre-requant Acc32`；
4. 四级结果源码版本不同、provenance 不同：成立。旧结果只留作 smoke，正式 sample
   shard 将用同一源码和四个冻结 executable 重跑全部 12 block；
5. expected/actual 共享 producer lineage：成立。只能称 adapter 路径隔离，不能称
   完全独立模型 oracle；
6. 三层 phase ledger 未生成：成立，仍是 formal G0 的独立 P0。
