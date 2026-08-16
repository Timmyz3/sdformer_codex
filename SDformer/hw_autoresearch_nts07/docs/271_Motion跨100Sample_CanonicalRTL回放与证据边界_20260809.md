# Motion 跨 100 Sample Canonical RTL 回放与证据边界

## 1. 本轮结论

本轮只补上一轮独立 DATE 评审指出的最高优先级缺口：全量 ordered count
profile 已覆盖 672000 条 head-row，但逐 bit RTL 仍只有 sample0/window0 的
138 条真实 row。由于现有 567 MiB profile 没有保存其余 row 的原始 32-bit
Q/K，本轮从可实现的 count/overlap/motion 合同构造 canonical Q/K，将 RTL
控制状态覆盖扩展到 100 个 sample。

结果为：

- `[rtl-canonical]` 5570 条 row、2506500 个 token 和 850021 个 gated-K
  输出完成 Icarus 与 Verilator+SVA 双仿真器差分；
- `[rtl-canonical]` 无反压与固定重反压两种模式均逐 gate、K bits、token id、
  输出次数及最终账本零失配；
- `[rtl-canonical]` 无反压 RTL 与 ordered profile 周期模型在 5570 条 row 上
  残差为 0；
- `[rtl-canonical]` 无反压 canonical 子集内，RQTB2S 与 TTB8-ZKQI 含共同
  225 拍 preload 的周期为 4653604/4007043，即 1.1614x；
- `[rtl-canonical]` 固定重反压下二者周期为 8022464/6550562，工作量与全部
  输出保持不变；该 1.2247x 只证明协议鲁棒性，不作为部署吞吐；
- `[prof]+[rtl校准模型]` 100 sample 全量总体收益仍引用 docs/270 的 1.3142x，
  不用按密度刻意选取的 canonical 子集替代总体统计；
- `[待验证]` 本轮不恢复原始 lane 身份，不产生真实 toggle、SAIF、功耗、EDP、
  encoder FPS 或目标工艺 PPA 证据。

本轮提升的是实现可信度，不新增 DATE 架构贡献。

## 2. 为什么 canonical 构造是可实现的

每个 temporal pair 的 ordered trace 给出：

```text
|Q0|, |Q1|, |K0|, |K1|,
|Q0 & K0|, |Q1 & K1|, |K0 xor K1|
```

构造器先由：

```text
|K0 & K1| = (|K0| + |K1| - |K0 xor K1|) / 2
```

恢复 K0/K1 的集合交、差，再分别在 K 内和 K 外选择 Q 的 overlap 与 Q-only
lane。它对以下非法情况 fail closed：

1. K count 与 motion 的奇偶不成立；
2. K 交集为负、超过任一集合或并集超过 32 lane；
3. Q/K overlap 超过 Q 或 K count；
4. Q-only 数量超过 K 补集容量。

构造后重新检查全部七个计数，并通过共同 32-bit rotation 扩展 lane 位置覆盖。
1000 组随机真实 bit 集合的 round-trip 与双时间片 H67 Q7 score 均为零失配。

## 3. 保持与不保持的语义

Canonical Q/K 严格保持：

- Q0/Q1 与 K0/K1 popcount；
- Q0&K0、Q1&K1 overlap；
- K0 xor K1 motion count；
- H67 Motion-XOR Q7 score；
- both-K-zero 三类 quotient 注入类别；
- score class、Shiftmax gate、gated-K 输出数和 TTB active mask。

Canonical Q/K 不保持：

- 原始数据集的 lane 身份；
- Q0/Q1 之间未被 score 使用的 temporal overlap；
- 相邻 token、window、head 或 sample 的真实切换相关性；
- 门级 toggle、SAIF、功耗或能量。

因此证据等级单独标为 `[rtl-canonical]`，不能升级成“真实多样本逐 bit trace”。

## 4. 行选择与覆盖

每个 sample/block 确定性选择：

1. 固定 hash 的 window/head；
2. active-pair 最小行；
3. active-pair 中位行；
4. active-pair 最大行；
5. 首个 TTB8 模型慢行，若该 block/sample 存在；
6. 额外保证 12 个 block 各自的全部 head 至少被覆盖。

最终覆盖：

| 项目 | 数值 |
|---|---:|
| sample | 100 |
| block | 12 |
| canonical head-row | 5570 |
| token | 2506500 |
| gated-K output | 850021 |
| active pair 范围 | 0..225 |
| 全活动 TTB8 模型慢行 | 1398 |

这个策略故意覆盖稀疏、中位、稠密和已知最坏状态，因此不能用其均值估计数据集
均值。总体均值、p95 和 p99 必须继续使用 docs/270 的 672000 行全量统计。

## 5. RTL 差分结果

### 5.1 无反压

| 项目 | RQTB2S | TTB8-ZKQI | TTB8 加速 |
|---|---:|---:|---:|
| 执行周期 | 3400354 | 2753793 | 1.2348x |
| 共同 preload | 1253250 | 1253250 | 1.0000x |
| 含 preload 周期 | 4653604 | 4007043 | 1.1614x |
| gated-K output | 850021 | 850021 | 相同 |

无反压 5570 条逐行周期与 ordered profile 模型全部相同，周期残差为 0。

### 5.2 固定重反压

| 项目 | RQTB2S | TTB8-ZKQI | 比值 |
|---|---:|---:|---:|
| 含 preload 周期 | 8022464 | 6550562 | 1.2247x |
| gated-K output | 850021 | 850021 | 相同 |
| Q/K 逻辑读 bit | 187616672 | 95045792 | -49.34% |

固定重反压同时作用于 descriptor issue 和输出 ready。Icarus 与 Verilator+SVA
逐行账本一致，且反压没有改变 active pair、score descriptor、three-class
seed、读 bit 或输出数。该模式用于验证 valid/ready 稳定性与 drain，不模拟真实
SRAM 仲裁，周期不得进入部署主表。

### 5.3 分 stage 无反压结果

| Stage | row | active pair | RQTB2S 含 preload | TTB8 含 preload | 加速 |
|---:|---:|---:|---:|---:|---:|
| S0 | 904 | 40.41% | 720280 | 610716 | 1.1794x |
| S1 | 818 | 14.84% | 474181 | 336258 | 1.4102x |
| S2 | 2806 | 43.42% | 2402746 | 2083002 | 1.1535x |
| S3 | 1042 | 62.46% | 1056397 | 977067 | 1.0812x |

canonical 子集继续呈现 S1 稀疏收益最大、S3 稠密收益最小的方向，但具体比例仍
不能替代全量 stage 分布。

## 6. 对 DATE 证据链的实际提升

上一轮的证据关系是：

```text
138 real-bit RTL rows
  -> 校准 672000 count-model rows
```

本轮变为：

```text
138 real-bit RTL rows
  -> exact count/score calibration
  -> 5570 realizable canonical-bit RTL rows
  -> 672000 count-model rows
```

这补强了“count 模型覆盖的空、稀疏、中位、稠密、全活动状态能否由真实 RTL
执行”的桥梁，并专门覆盖 1398 条 TTB8 慢 1 拍的全活动行。它没有补齐真实
switching activity，也没有扩大系统边界到 full encoder。

## 7. 当前允许与禁止的论文主张

允许写：

> 在 100 sample 全量 ordered profile 上校准的 row-cycle 模型显示，
> TTB8-ZKQI 相对 RQTB2S 的 preload-inclusive 加速为 1.3142x；该模型由
> 138 条真实 bit row 和覆盖 100 sample 状态分布的 5570 条可实现 canonical
> Q/K row 进行零残差 RTL 校验。

禁止写：

- “5570 条是原始数据集 bit trace”；
- “全部 672000 条 row 已经 RTL replay”；
- “canonical 回放证明功耗或 EDP”；
- “逻辑读 bit 等同 SRAM 物理流量或能量”；
- “row-level 周期等于 full encoder FPS”；
- “本轮提出了新的架构机制”。

## 8. 复现入口

一键入口：

```bash
sim_h67/run_h67_zkqi_canonical_replay.sh
```

核心产物：

```text
tb_h67/vectors/h67_zkqi_canonical_multisample_20260809/
results/h67_zkqi_canonical_replay_20260809/report.{md,json}
```

回归包含 7 个 Python 单元测试、向量重新生成、Icarus 两种模式、
Verilator+SVA 两种模式及最终跨日志 fail-closed 汇总。

## 9. 下一轮门槛

本轮完成后才允许评估 B4/B8/B16/B32。候选方向是复用同一 225-bit active
metadata 的多分辨率层次扫描，由 12-block descriptor 静态选择粒度，而不是
复制 bitmap 或加入逐 row predictor。晋级必须同时满足：

1. 100 sample 全量模型给出独立于 zero-K gating 的周期收益；
2. 在本轮同一 canonical 集合上逐 gate/Acc 语义不变；
3. 相对 B8 的 mask 选择器、元数据和控制面积被显式计入；
4. 固定 5 ns、相同六 SRAM 宏的开放物理代理闭合；
5. 若收益只是参数微调或面积归一吞吐不增，则记为负结果，不列入 DATE 贡献。

Local5 watcher 继续等待同窗 all-head profile。真实 profile 到达后，Local5 的
checkpoint-bound T450、12-block 调度和多窗口 SRAM/反压闭环仍是双线中的当前
系统完整度优先项；Motion 不停止，也不冻结新的架构候选。

## 10. 独立 DATE 包级评审

独立子代理只读审查 docs/268--271、结构化结果、生成器、周期模型和 miter，未
修改文件。裁决为：

| 维度 | 分数 |
|---|---:|
| 总推荐 | 3.3/5，Weak Reject |
| 新颖性 | 3.1/5 |
| 架构完整度 | 2.9/5 |
| 实现可信度 | 3.9/5 |
| 实验完整度 | 3.1/5 |

评审确认本轮把逐 bit 状态覆盖扩大约 40.4 倍，覆盖全部 `0..225` active-pair
计数和 1398 条最坏慢行，实质补强了 count 模型到 RTL 的桥梁；但 canonical
row 只占全量模型行约 0.83%，且与周期期望共同来源于同一 profile，不能独立
验证 profile 本身，也不能替代真实 switching、系统调度或 ASIC PPA。

评审保留的拒稿理由为：

1. 真实数据集 bit-level row 仍只有 138 条；
2. 当前范围仍是 attention row 子系统；
3. ZKQI 精确语义有价值，但主要周期收益仍来自 Bishop 启发的 TTB 层次跳扫；
4. 没有真实 SAIF、能量、EDP、目标 SRAM 或 full-encoder 性能。

下一轮裁决为“允许进入模型 DSE，不允许直接实现四套 RTL”。必须预先划分
sample calibration/held-out 集，在 calibration 上冻结全局、per-stage 和
12-block 粒度描述符，再只在 held-out 上比较：

- B8；
- 最佳全局固定 B；
- per-stage 静态 B；
- 12-block 静态 B；
- 冻结 `{B4,B8,B16,B32}` 候选集合内的无代价逐 row oracle 上界。

若 12-block 静态策略相对最佳全局固定 B 的 held-out preload-inclusive 周期
增益低于 5%，或后续固定 5 ns 面积归一吞吐增益低于 5%，它只能作为参数 DSE
负结果。只有“同一 225-bit metadata 的多分辨率视图 + 统一扫描器”、held-out
泛化、完整 selector/mux/control 代价和 p95/p99 不回退同时成立，才可能晋级为
有限的 DATE 架构贡献。
