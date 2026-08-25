# M94 critical-first fusion seed 独立打铁

## 结论

评分 **87/100**，`P0=0 / P1=3 / P2=6`。

contract、probe、raw result/log、receipt 和 M89 baseline exact SHA 全部通过。交错日志包含
正好 **120** 个 completion marker：三策略各 40 个 record、10 个 sample、每 sample 四个
operator。三组 raw record→sample→aggregate、p95、fusion groups、oldest 逐样本 exact reproduction、
critical 每样本 delta 及全部 gate 均独立复算一致。

- oldest 精确复现 M89 K6。
- critical-first 在 10/10 sample 的 source、integrated 和 fusion-group count 全部回退。
- sparse-first 比 critical-first 更差，不能重写成新的“sparse seed”故事。
- `PASS_EXECUTION_NO_GO_PROMOTION` 完全正确；critical/sparse seed 轴应关闭。

## 三策略数字

| policy | source | integrated | p95 | groups | unique weight issues |
|---|---:|---:|---:|---:|---:|
| oldest | 69,964,176 | 76,677,320 | 7,843,680 | 10,436,792 | 416,232,640 |
| critical-first | 70,120,296 | 76,834,608 | 7,859,920 | 10,457,696 | 417,023,664 |
| sparse-first | 70,181,504 | 76,881,496 | 7,860,912 | 10,448,216 | 417,521,832 |

critical-first 相对 oldest：

- fusion groups `+20,904`（+0.20029%）；
- source `+156,120`（+0.22314%）；
- integrated `+157,288`（+0.20513%）；
- p95 `+16,240`；
- non-source overhead 仅 `+1,168`。

因此 99.2574% 的 integrated 回退来自 source-cycle 回退，不是外围 stall。每样本 source delta
为 +7,832 至 +24,136，integrated delta 为 +8,352 至 +20,704，group delta 为 +1,120
至 +3,080；没有例外窗口。

critical 距冻结的 source/integrated 晋级上限分别差 505,941 / 540,675 cycles。p95、逐样本
source、逐样本 integrated gate 同样失败。

## 为什么 oldest 更好

三策略 parent choice 与 logical source updates 完全相同，唯一变化是 seed。critical-first 确实把
selected standalone score sum 从 35,963,640 提到 41,641,360，并有 2,643,736 次 non-oldest
选择，但结果产生更多 group 和更多 union work。

standalone bank cost 不等于 group union value，也不等于 DAG frontier value。critical-first 抽走
一个高成本但非 oldest 的 task 后，留下的 prepared set 可能更难组成满组；它还可能延迟低 index
task，而这些 task 往往负责推进 canonical up/left DAG。task index 对应 flattened spatial order，
oldest 自然保留时空邻近、mask 相关性及依赖 frontier 的推进。

这是有数据支持的机制解释：groups +20,904、unique issues +791,024、source +156,120。但 raw
result 没输出 group cardinality、seed/member spatial distance、mask intersection 或 descendant
unlock，因此 locality/DAG 仍是强推断，不是因果隔离证明。

sparse-first 更不能救线：它相对 critical 又多 61,208 source 和 46,888 integrated cycles，且
合同已预声明为永不 promotion 的 negative control。两个 standalone 极端都输给 oldest。

## 源码合同

静态审计确认 M94 在冻结 M53 temporal transform 之后，只把唯一一行
`seed = min(prepared)` 替换成 `select_seed(...)`；parent、canonical DAG 与后续 greedy member
completion 均未改变。

- critical key：`(-standalone cycles, task index)`；
- sparse key：`(standalone cycles, task index)`；
- score 范围 0..32，因此六位数值上足够；16 entries 原始新增 96 bit；
- vector payload 新增 0。

不过六位 metadata 只是注释账本：模拟器仍直接对 delta mask 调 `cycle_cache`，未实现 metadata
entry、对齐、比较树 II/latency、时序或功耗。96 bit 也不能直接当成物理 12B macro cost。

seed audit 在一个 output-block schedule 上统计后乘以 8，并严格等于 aggregate fusion groups；
这符合冻结 transaction model。但真实硬件是每 block 重算，还是保存一次并回放八次，尚未冻结，
所以这些是逻辑选择事件，不是 selector 功耗活动 trace。

## 下一方向

seed-order 轴已经被 oldest/critical/sparse 三点共同关闭，不应继续事后调 seed score。

当前只保留：**oldest 固定 seed + 既有 overlap-aware greedy member completion**。现有 member
selector 本来就在最大化 saved union cycles，因此新的 member-selection 只有在小型、预冻结的诊断
显示二候选/小窗口规则能改善 group fill 且硬件很轻时才值得开线。

更高优先级是等待 PAFT：让算法改变 delta masks，再用完全不变的 M89-K6/oldest scheduler replay。
在等待期间可以只加 oldest 的 group cardinality、spatial span、union overlap、DAG unlock 诊断，
但不得称新倍速，也不得重开 M94 gate。

机器结论见 `m94_critical_first_fusion_seed_independent_hammer_review.json`；独立脚本未 import/执行
producer，也未修改 contract、script、result、log 或 receipt。
