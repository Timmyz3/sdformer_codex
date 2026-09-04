# M96 fixed-group reversible bank-skew 独立打铁

## 结论

评分 **91/100**，`P0=0 / P1=3 / P2=5`。

contract、probe、raw、remote log、receipt 和 M89 receipt 的 exact SHA 全部通过。独立脚本没有
import 或执行 producer；它从 40 条 raw record 重建 operator、sample、aggregate、oracle、门槛和
stage2 状态，并逐条对齐 completion log 与 M89 H0 record。

M96 的 stage1 负结论成立：四个 operator 全选 H0，selected source 仍为 69,964,176 cycles，
gain=0、speedup=1.0×。H1/H2/H3 不仅 aggregate 回退，而且在 **40/40** 个 sample×operator
record 中都严格慢于 H0。因此：

- `PASS_M96_STAGE1_EXECUTION_NO_GO` 正确；
- stage2 integrated replay 不应运行；
- M96 bank-skew RTL 不应进入 VCS、Formality、DC/PT；
- 不存在 M96 integrated、RTL、PPA、system 或 headline speedup。

## 独立复算

| mode | source cycles | 相对 H0 delta | 相对 H0 |
|---|---:|---:|---:|
| H0 identity | 69,964,176 | 0 | 1.0000× cycles |
| H1 xor-row | 73,329,584 | +3,365,408 | +4.8102% |
| H2 add-row | 72,742,080 | +2,777,904 | +3.9705% |
| H3 add-3row | 73,371,104 | +3,406,928 | +4.8695% |

H0 还精确复现：10,436,792 groups、416,232,640 unique issues、562,451,704 logical
updates、每样本 212,336,640 weight-DMA bytes。每个样本 selected delta 都是 0；更宽松但禁止
用于实现的“每 sample×operator record 选一个 mode”oracle 仍是 69,964,176，gain=0。

promotion limit 是 69,614,355，当前还高 349,821 cycles。唯一失败 gate 正是 0.5% source
改善门槛，其余 identity/conservation/fixed-mode/non-regression gate 一致通过。

## 可逆性与冻结边界

独立枚举证明每个 mode 对 32 个 weight row 都把 base bank 0..7 双射到 bank 0..7；每个 bank
仍恰好承载 32/256 个位置。因此在抽象模型中，H0-H3 都能作为离线 weight layout permutation，
不增加 bank 深度或 weight bytes。

源码审计确认 M96 audit hook 插在 frozen group 已选定、`group_cycles` 已计算之后，而且结果不反馈
给 scheduler。raw 中每条 H0 的 group、union popcount、source cycle 又与 baseline replay record
逐条相等；这足以支持本次 transaction-model preservation。

但这不是地址生成 RTL 或 SRAM macro 证明。`zero_extra_ports_capacity_and_vector_storage` 是构造性/
硬编码 gate，没有 netlist、macro pin、decoder timing、weight image 或 numeric address miter。由于性能
为负，继续做这些物理证明没有收益，停止是正确的。

## 下界不能写成结果

`sum_group ceil(union_popcount/8)` 是 56,660,864 cycles，距 H0 尚有 13,303,312 cycles；若能
完全达到，下界对应 1.23479× source-only ceiling。这个数字只由八 bank work conservation 得到：

- 它不是 H0-H3 中任何 mapping 达到的周期；
- 它不保证存在一个固定 per-operator row permutation 能同时平衡所有 group；
- 它不包含 integrated calendar、decoder、SRAM macro 或频率成本；
- 不能写成预计 speedup、系统 speedup 或 headline。

它仍说明“bank imbalance”在数学上存在，但这四个简单 row-phase hash 把本来较好的 H0 布局打乱了。

## claim 边界

receipt 中“closing the fixed reversible weight-row bank-skew axis”应收窄为：

> 在冻结 M89-K6 的 10 样本、4 operator、40 record 上，关闭 H0/H1/H2/H3 四种预注册、
> 每 operator 固定的 row-phase mapping。

当前证据没有穷尽每 row 的任意 8! permutation，也没有计算真正的 per-group mode oracle。现有
“per-record oracle”是一条 sample×operator record 选一个 aggregate mode，虽比 per-operator 固定
选择更宽松，仍可能掩盖同一 record 内不同 group 的 mode 得失抵消。因此不能据此宣布所有可逆
bank packing 都不可能。

## 下一方向

保留 `H0 + M89 K6 oldest/saved-first`，不要为 M96 启动 stage2/VCS/DC。性能主线优先等待 PAFT
改变真实 masks，然后用完全不变的 H0 scheduler 重放。

如果仍要挖 bank packing，应先加只读诊断：逐 group 的 dominant-bank、row-phase histogram，以及
真正的 per-group H0-H3 oracle。只有 oracle 先显示足够机会，才值得预注册一个训练集离线优化、
验证集冻结的 per-operator row permutation，并明确收费 decoder/weight-layout 成本；不要再盲猜
更多 hash，也不要使用 per-sample/runtime oracle。

机器结论见 `m96_fixed_group_reversible_bank_skew_independent_hammer_review.json`；独立复算见
`m96_independent_recompute.json`。
