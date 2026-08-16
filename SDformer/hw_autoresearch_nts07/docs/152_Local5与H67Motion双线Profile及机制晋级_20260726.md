# Local5 与 H67 Motion 双线 Profile 及机制晋级

**日期**：2026-07-26  
**范围**：Local5 profile100、H67 Motion profile100 同口径复核、可借用机制
筛选和下一阶段硬件决策。  
**详细机器可读结果**：
`results/local5_h67_dual_profile_decision_20260726/`。

## 1. 回答当前问题

### 1.1 Local5 是否需要补 profile

需要，且本轮已经补完 `100` 个样本。原因是 Local5 虽然每次归一化只有
3/4/5 个邻居，但每个窗口共有 `738` 条有效有向边，不能仅凭“局部注意力”
判断它比 H67 的 162-token score 更便宜。

profile100 给出的 **pre-G0 探索结果** 为：

| 指标 | Local5 profile100 |
|---|---:|
| 四方向 K-XOR lane density | 1.8916% |
| exact-K edge | 86.0097% |
| delta count p50/p95/p99 | 0/5/10 |
| 变化边中 <=4 lane | 61.4091% |
| 变化边中 <=8 lane | 87.2761% |
| source-resident K-bit 读取理论减少 | 78.0488% |
| detector/metadata 免费时的 selected-lane 减少 | 76.5724% |

边界 mask 修复会改变前层 attention 输出，进而改变后续 block 的 Q/K 和
K-XOR。因此除固定 738-edge 拓扑及由拓扑推导的 `78.0488%` 全 lane 读取
比例外，上述数据都必须在 G0/G1 后复跑。它们当前只支持继续做 RCSD DSE，
不能作为论文最终统计。

当前 JSON 已补 config/checkpoint SHA256，并从 `dataset.files[:100]` 重建
ordered sample-key manifest。它与 H67 JSON 的 100 个 sample key 逐项一致，
共同 SHA256 为
`c49ebac2188beb3a35beb8e8ccbe1185d847ff59316b5e3dc20010eb4fd85b40`。

### 1.2 这些机制是否只适用于 Local5

不是。H67 Motion 已有 ordered profile100，本轮按相同问题重新计算：

| 指标 | H67 Motion profile100 |
|---|---:|
| temporal pair 完全无更新 | 74.0896% |
| Q/K 时间 update lane density | 2.5029% |
| 变化 pair 平均更新 lane | 3.0912 |
| 变化 pair 中 <=4 lane | 76.9060% |
| 变化 pair 中 <=8 lane | 95.3225% |
| detector/metadata 免费时的 selected-lane 减少 | 48.7485% |
| TTB4 empty | 61.0828% |
| 两时间片 score 相同 | 98.6949% |
| 双 K-zero 且同 class | 83.0333% |
| final-gate term-count 减少 | 82.4926% |
| motion-zero bypass 候选 | 83.2087% |
| K temporal source-read reuse | 9.9922% |
| changed-run 平均长度 | 3.6997 |

因此 H67 Motion 同样适用：

1. Prosperity 式 exact/partial residual reuse；
2. FireFly-T 式多 lane sparse decoder 候选；
3. 复旦 ISSCC 蝶形 zero-skip/compact 网络；
4. Bishop TTB4 打包和 empty bundle gating；
5. SpAtten 式级联 issue，但改成无损的 pair-class coalescing；
6. H67 已有的 SCS 和 gate-class projection folding。

两条线可共享 bitmap detector 和 set-bit extractor 的设计思想；物理上能否
共享 compactor、reduction tree 和 RNE 状态仍需同约束 PPA。锚点和地址生成
不同：

```text
Local5：self K 是空间锚点，恢复 N/S/E/W score
H67：   {Q0,K0} 是时间锚点，恢复 {Q1,K1} score
```

## 2. 对“照搬并小改”的判定

直接借用已有机制没有问题，条件是：

- 论文明确引用，不把 TTB、蝶形压缩或 exact reuse 改名后宣称首次提出；
- 修改必须由本 workload 的语义驱动，并有单独消融；
- 只有加入 detector、compactor、FIFO、SRAM 端口和 fallback 后仍降低
  cycle/energy，才能写成性能收益。

当前推荐的借用和修改如下：

| 来源机制 | 直接借用部分 | 本工作的修改 |
|---|---|---|
| Bishop TTB | token-time bundle、empty gating | H67 固定 T=2；Local5 改为带 halo/方向 mask 的 STT |
| Prosperity | exact/partial residual reuse | 不做在线相似度搜索；使用静态空间/时间锚点 |
| FireFly-T | multi-lane sparse decoder | 解码 Q/K update mask 或 N/S/E/W K-XOR mask；4/8-lane 只是 DSE 点 |
| 复旦 ISSCC 蝶形网络 | zero-skip compaction | 压缩 32-bit delta mask；必须比较多拍、fallback 和 burst |
| Phi | pattern + residual | 只保留为 codebook 对照；首版不增加 pattern SRAM |
| SCS/GateStack | gate-class folding | H67 保留；Local5 必须保持 1 至 5 的多重集语义 |
| SpAtten cascade | 分级 issue 的控制组织 | 不删除 token；H67 同 score 时把两次 class commit 合成 multiplicity=2 |

## 3. 架构候选：可切换的静态锚定差分核

```text
operand SRAM / Local5 line buffer
                 |
                 v
     static-anchor delta detector
       |                         |
       | zero                    | nonzero mask
       v                         v
 exact bypass          32 -> 4/8 lane compactor
                                 |
                         sparse delta issue
                                 |
                 +---------------+---------------+
                 |                               |
          delta/remainder path             direct fallback
                 |                               |
                 +---------- reduction tree ----+
                                 |
                  RNE score + Shiftmax/SCS
                                 |
                      term projection backend
```

### 3.1 Local5 模式

- 先算 self anchor 的无界整数累加值和舍入余数；
- 邻居 K 与 self K 做有向 XOR；
- exact-K 直接复用 anchor；
- 稀疏 delta 经过待选宽度 compactor 更新 anchor；当前 4/8-lane 只作为
  多拍/fallback DSE 点；
- delta 过密时走 direct reduction；
- 最后只做一次 RNE，避免从已舍入 Q7 score 做差。

### 3.2 H67 Motion 模式

- `{Q0,Q1,K0,K1}` 共驻留；
- `K0 XOR K1` 同时服务 Motion-XOR bias 和 temporal delta detector；
- `u=0` 时精确复用 T0 score；
- `u!=0` 时只发射变化 lane；4-lane 覆盖约 `76.91%` 的变化 pair，
  8-lane 覆盖约 `95.32%`。这只是变化 pair 的单拍覆盖；仍有
  `23.09%/4.68%` 超宽，需要多拍或 direct fallback；
- 密集残差回到原 H67 direct score，不改变任何 score/Shiftmax 结果。
- 两时间片 score 相同时，SCS 前端只提交一次 `{class, multiplicity=2}`；
  该提交只更新 SCS class-count，不合并 K 不同的 projection contribution。
  `98.6949%` 是可合并上界，真实增量收益受现有 SCS folding、SRAM 写口和
  流水相序限制。
- Motion-zero `83.2087%` 可关闭 Motion-popcount 分支；Stage update density
  从 S1 的 `0.3570%` 到 S0 的 `4.1636%`，需评估 per-stage width/power gating。
- changed-run 平均长度为 `3.6997`，应与 TTB4、FIFO 和 ordered-run issue
  联合建模，不能只看单 pair coverage。

## 4. 为什么当前不先做异构双核

数据支持 direct/delta 两种执行模式，但还没有证明复制两个物理 core，或把
两条算法的 reduction tree 强行共享，会更省 EDP。对于 32-lane 小问题，
stratifier、双 FIFO、跨核路由和 idle leakage 可能抵消收益。

所以 Bishop 当前优先借用的是 TTB 和分流思想，不是直接复制 dense/sparse
双核。第一版先验证参数化前端和部分累加资源，后续用同约束 DC/SAIF 决定
是否共享或物理分裂。

## 5. 不能混淆的数字

| 数字 | 正确含义 | 不能声称 |
|---|---|---|
| Local5 76.5724% | detector/metadata 免费时，相对 dense edge enumeration 的 selected-lane 减少 | RTL cycle、功耗，或优于 H67 |
| H67 48.7485% | detector/metadata 免费时，相对 dense T=2 compare 的 selected-lane 减少 | attention 端到端加速，或劣于 Local5 |
| 4/8-lane coverage | 变化项落在稀疏 issue 单拍宽度内的比例 | compactor 零开销或宽度已冻结 |
| Local5 MFEP 92.7098% | 当前 Python RTL-like gate 下的 term-count 压缩候选 | G0/G1 后仍成立、cycle 或 PPA 收益 |
| H67 final-gate 82.4926% | active K lane 到 gate-class term 的 term-count 压缩 | full encoder 能耗减少 |
| H67 score-equal 98.6949% | 两时间片可进行同 class 原子合并的比例 | SCS cycle 直接减少 98.6949% |

Local5 的结构上限为 `78.0488%`，H67 的结构上限为 `50%`；二者分别达到
各自上限的约 `98.11%` 和 `97.50%`。因此 `76.57% > 48.75%` 主要来自
基线拓扑差异，不能用于排序两种架构。

## 6. 独立评审结果

本轮独立 DATE 审阅结论是：**profile 足以继续机制 DSE，但不接受架构晋级
或论文定量结论。**

已按评审修复：

1. 将 Local5 Q/K 和差分数据改为 pre-G0 探索证据；
2. 将 lane-work 改为 detector/metadata 免费的 selected-lane 模型；
3. 将 4/8-lane 降级为多拍/fallback/burst DSE 候选；
4. 将 MFEP/H67 后端收益改为 term-count 压缩；
5. 将 pair coalescing 限定为 SCS class-count commit；
6. 将统一 substrate 改为“具有共享潜力”，等待同约束 PPA。

仍未关闭：G0/G1 后同 cohort 复跑、ordered burst/FIFO、SRAM latency 和
同约束 DC/SAIF。当前 cohort 对齐已通过 manifest hash 关闭。

## 7. 下一阶段

1. 修复 Local5 的边界 mask、Shiftmax x2、score RNE 三个数值 P0；
2. 建 H67 Motion-Delta + pair-class coalescing 无界整数参考和 cycle model；
3. 对 priority encoder、4/8-lane 蝶形 compactor 做多拍/fallback/burst
   cycle model 和同约束综合；
4. 建 Local5 RCSD 无界整数参考，证明 direct/delta score_q7 零失配；
5. 补 Local5 ordered STT 的 burst、FIFO、SRAM bank conflict；
6. 只在上述结果证明净 cycle/EDP 收益后，把对应机制并入主 RTL。

近期验证优先级是 **H67 Motion-Delta > Local5 RCSD > Local5 MFEP**。
这个顺序不是认定 H67 最终一定是算法主线，而是控制硬件迁移风险：前两项
至少能复用 bitmap detector/set-bit extractor 的设计与验证方法；更深层的
物理共享必须等 PPA 后再确认。
