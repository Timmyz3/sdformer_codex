# M95 marginal bank-growth member 独立打铁

## 结论

评分 **88/100**，`P0=0 / P1=3 / P2=5`。

contract、probe、remote raw/log、receipt、M89 baseline exact SHA 全部通过；交错日志正好有
120 个 completion marker，三策略各 40 record / 10 sample。三组 record→sample→aggregate、
saved-first 逐样本 exact reproduction、阈值、p95、候选审计恒等式、逐样本 delta 和全部 gate
均独立复算一致。

- saved-first 精确复现 M89 K6。
- marginal-growth 在 10/10 样本的 source 和 integrated 全部回退。
- standalone-heavy negative control 更差。
- `PASS_EXECUTION_NO_GO_PROMOTION` 正确。
- M95 marginal/standalone **不得进入 VCS 或 DC**。

## 三策略

| policy | source | integrated | p95 | groups | unique issues | candidate evals |
|---|---:|---:|---:|---:|---:|---:|
| saved-first | 69,964,176 | 76,677,320 | 7,843,680 | 10,436,792 | 416,232,640 | 42,377,904 |
| marginal-growth | 70,086,824 | 76,783,744 | 7,852,008 | 10,426,200 | 417,211,800 | 42,135,400 |
| standalone-heavy | 70,408,432 | 77,122,968 | 7,881,520 | 10,440,160 | 417,063,032 | 42,521,216 |

marginal 相对 saved-first：

- groups `-10,592`，候选评价 `-242,504`；
- non-source overhead `-16,224`；
- 但 unique issues `+979,160`，source `+122,648`；
- integrated `+106,424`，p95 `+8,328`。

它距冻结的 source/integrated promotion limits 分别还差 472,469 / 489,811 cycles，逐样本、
p95 和 aggregate gate 全失败。

## 为什么 group 更少仍然更慢

总 group 减少 0.1015%，但平均每 group 的 source cost 从 6.70361 增到 6.72218，增幅
0.2771%。七个样本 group 减少，三个样本反而增加；无论 group 增减，十个样本的 unique
issues、source 和 integrated 全部回退。

在同一步 group completion 中，current union 对所有 candidate 是常数：

- marginal-growth 最小化 `fused-current`，只优化眼前新增成本；
- saved-first 最大化 `current+standalone-fused`，等价于奖励“被当前 union 吸收掉的 standalone
  工作”。

因此 marginal 会优先消耗低-growth、往往也较轻的 candidate，把 standalone 成本高的工作留给
后续 group；局部 union 增长更小，却得到更差的全局贪心分箱。unique issues 全样本增加和 source
全样本增加直接支持这一解释。

不过 audit 的 standalone/current/fused/saved/growth sums 覆盖全部候选，不是只覆盖 winner；
没有 chosen-member 和 stranded-tail ledger，因此机制是强支持解释，而非逐决策因果隔离。

## 源码与硬件成本

静态审计确认 oldest seed、parent、canonical DAG 和 K6 fill loop 都没变。冻结 M53 transform 后，
只把唯一原 member tuple 替换为 `rank_member(...)`：

- saved：`(-saved, fused, candidate)`；
- marginal：`(fused-current, -saved, candidate)`；
- standalone：`(-standalone, fused, candidate)`。

三者与合同一致；每个 candidate 都检查 `saved=current+standalone-fused`。audit 的全部 sum 字段
均能被 8 整除，saved/growth 恒等式闭合，min/max 没被错误放大。

但“new lanes=0、metadata=0”只是相对 Python/M45 数据路径的结构声明。K6 selector 本身有约
42M 个逻辑候选评价，模拟器没有收费 compare-tree II/latency、growth subtraction、tie-break、
mask ports、开关功耗或 achieved clock，也没有综合过的 K6 selector 证明 key 更换对时序免费。
由于性能已经全样本为负，没必要用 VCS/DC 继续投入。

## 下一方向

M94 已关闭 seed 极值，M95 又关闭 marginal/standalone member key。保留唯一基线：

`oldest seed + saved-first member completion`

下一性能优先级应是等待 PAFT，用改变后的 masks 重放完全不变的 saved-first scheduler。可以只加
winner overlap、final group cardinality 和 stranded-tail 诊断反哺算法，但不能继续事后调 comparator，
也不能称新倍速。

只有 PAFT+saved-first 出现正向冻结结果，或出现一个完全不同、明确收费的 bounded module，才应
恢复 VCS/DC。

机器结论见 `m95_marginal_bank_growth_member_independent_hammer_review.json`；独立脚本没有
import/执行 producer，也未修改生产代码、合同、结果、receipt 或 docs/359。
