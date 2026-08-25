# M91 dependency-safe fusion-aware parent 独立打铁

## 结论

评分 **84/100**，`P0=0 / P1=5 / P2=5`。

M91 的 contract/probe/result/log/receipt 与 M89 K6 baseline exact SHA 均通过；raw log
正好覆盖 40 个互异 record，raw result 的 40 个 per-record ledger 已独立重聚合为 10 个
sample。source、integrated、p95、逐样本 delta、1% gate 和 20,270-cycle miss 全部复算一致。

- **GO**：冻结 transaction-model 负筛选、exact-source dependency safety，以及 fusion-aware
  parent selection 是值得继续诊断的硬件算法假设。
- **NO-GO**：M91 promotion、RTL 倍速、selector PPA、全网/系统和 DATE headline。
- 不得把 0.9736% 四舍五入成 1%，也不得事后放宽 M91 gate。

## 独立复算

| 项 | M89 K6 | M91 | M91−baseline |
|---|---:|---:|---:|
| source cycles | 69,964,176 | 69,211,896 | -752,280 |
| integrated cycles | 76,677,320 | 75,930,816 | -746,504 |
| p95 integrated | 7,843,680 | 7,769,480 | -74,200 |

source 减少 1.075235989%，integrated 减少 **0.973565586%**，integrated speedup
1.009831371x。十个 sample 的 source/integrated 都改善；integrated delta 从 -61,448 到
-99,016 cycles。

冻结的 1% 上限是 `floor(76,677,320 × 0.99)=75,910,546`。M91 为 75,930,816，
明确多 **20,270 cycles**，所以 `PASS_EXECUTION_NO_GO_PROMOTION` 判定正确。

receipt 的 admitted prose 写成 0.973565946%，与精确值有极小数字转置；不影响 gate，后续
摘要应使用 0.973565586%，但不能修改冻结 receipt。

## Dependency contract

exact-SHA 源码确实先用 canonical names 构造 DAG，之后才在 residency admission 选择 parent：

- local-zero 不需要依赖；
- previous-timestep 只在 `timestep>0` 出现，外层调度已完成上一 timestep；
- up 只在 `y>0` 出现，而 canonical M45 DAG 对所有 `y>0` 都保留 up-row edge；
- left 只在 canonical parent 本来就是 left 时出现，因此 exact left edge 已在 DAG 中。

choice 写入 `chosen[task]` 后没有重赋值。因此在当前 frozen source 下，新增 dependency 确为 0，
parent ready/acyclic 合同成立。

但结果中的 `new_dependency_edges_equal_zero` 是直接写死的 `True`，没有运行时输出 canonical-edge
与 selected-parent-required-edge 的 set difference。结论依赖静态源码审计，而不是 raw result
自身动态证明。

## Fusion-aware score 与硬件风险

3,240,000 次 admission 中：

- fusion-aware 2,846,529 次，empty-resident fallback 393,471 次；
- reselection 589,095 次，占全部 18.1819%，占 fusion-aware admission 20.6952%；
- 最终选择 local-zero/left/up/previous 分别为 1,776,331 / 691,725 / 529,692 /
  242,252。

实现的 lexicographic tuple 与合同一致，但第一项只是“任一 resident anchor 的最小 pairwise OR
cost + 非零 parent 固定 1 cycle”。真正的 K6 group 稍后才从 prepared tasks 形成，可能根本不含
winning anchor；command、parent-port contention、dependency readiness、response/FIFO、final
accumulator 和 output wait 也没有进入 score。因此它是 proxy，不是 integrated stall objective。

硬件复杂度更危险。最坏每次 admission 为 `16 ready × 4 options × 15 anchors = 960` 次
256-bit OR、8×32-bit population/max；2,846,529 次 fusion admission 的静态上界是
2,732,667,840 次 pairwise evaluation。raw result 没记录实际 evaluation 数；selector latency、
流水线、比较归约、功耗，以及八个 output block 之间决策是存储还是重算全部免费。20,270-cycle
gate miss 很小，但这不是放宽 gate 的理由，反而说明任何 selector 成本都会关键。

## 下一最小冻结实验

建议 M92 只做 `M91 decision causal + selector cost audit`，不改变原 M91 gate：

1. 保持同一 40 records、K6/CTX16/W16、canonical DAG、合法 parents、M89 baseline 和
   `integrated <= 75,910,546`。
2. 记录每次 admission 的真实 option/anchor evaluation 数、winning anchor 是否进入实际 K6
   group、predicted score 与 realized union/wait、parent-read 暴露 stall、canonical→selected
   transition，以及八 block 的 store/recompute 成本。
3. 冻结 1/4/16 score-lane selector 的 initiation interval/latency 并计入 cycles。零延迟 selector
   不得 promotion。
4. 只有“candidate + selector 全部收费”仍通过原 gate 才能进入后续 RTL；否则杀线。

机器结果见 `m91_dependency_safe_fusion_aware_parent_independent_hammer_review.json`。复算脚本未
import/运行 producer，也未修改 contract、实现、result、log 或 receipt。
