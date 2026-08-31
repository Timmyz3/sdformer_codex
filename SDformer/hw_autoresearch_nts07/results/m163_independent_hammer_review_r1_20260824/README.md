# M163 r1 独立打铁评审

结论：`43/100，REJECT_M163_R1_WRONG_BN_GRAIN_AND_FAILED_DC`。

M163 r1 的 96 个 signed-INT8 rank product slots、32 个 square lanes、16 个共享 RNE/saturating requant lanes 是真实结构，sealed VCS 也确实执行成功；但 RTL 把 16 个 hidden-channel lane 的 moment 聚合成一个 scalar `sum/sumsq/count`。冻结 `no_running` BN1 需要每个 hidden channel 独立统计，因此 VCS scoreboard 与 DUT 是在错误粒度上自洽。这个 P0 使 r1 不能作为动态 BN frontend 接纳，也使其 DC 面积成为错误设计的成本。

## 评分

| 项目 | 得分 |
|---|---:|
| identity / provenance | 7/15 |
| arithmetic / protocol | 15/20 |
| production semantics | 3/25 |
| performance / fair baseline | 5/20 |
| commercial flow / physical evidence | 8/15 |
| claim hygiene | 5/5 |
| 总分 | **43/100** |

## P0：16 个 hidden channel 被归约成一个 moment

Contract 明确写明 `hidden_lanes_per_tile=16`；每拍输入为 2 个 time samples × 16 hidden channels。正确 BN1 统计应为：

- 每个 lane 独立维护 `sum[j]`、`sumsq[j]`、`count[j]`；
- 一 tile 后每 lane count 为 10，而不是跨 lane scalar count 160；
- 后续 spatial tiles 继续累积到相同 hidden lane；
- module barrier 后为每个 hidden channel 生成自己的 mean/variance。

r1 mapped netlist 只有：

```text
moment_sum   [47:0]
moment_sumsq [55:0]
moment_count [31:0]
```

独立反例：lane0 为十个 `-100`，lane1 为十个 `+100`，其余 14 个 lane 为零。正确每-lane variance 全为 0；r1 scalar packet 却为 `sum=0, sumsq=200000, count=160, variance=1250`。下游无法从这个 scalar packet 恢复 16 组 moments。

原 TB 也跨 16 lane 累加同一个 reference scalar，因此 sealed VCS 的 PASS 只证明“错误 scalar 规范”的实现一致性。评审期间新增的 cross-hidden-lane correction overlay 正确撤销了 r1，但不能追溯接纳原 run。

## 资源、位宽与 II

独立解析 sealed DC precompile resources：

- `128` 个 signed 8×8 multiplier operations；对应 96 rank products + 32 squares；
- `16` 个 25-bit variable right-shifter/requant datapaths；
- 48 个 rank output values 分 3 拍经过 16 个 requant lanes；
- 每 tile 输入 5 个 accepted beats，每拍 2×16 个 Q8 values。

数值范围安全：

- signed 8×8 product：`[-16256, 16384]`；
- 十项 projection：`[-162560, 163840]`，19-bit signed 已足够，r1 使用 24 bit；
- 十个 factor row sum：`[-1280,1270]`，12-bit signed 足够；
- square 最大 `16384`，16-bit unsigned 足够。

对 r1 已读取的 RNE 函数做了独立静态复算：覆盖可达 projection 全范围和 shift 0–23，共 `7,833,624` 组，无 ties-to-even、饱和或符号反例。但 r1 source bytes 已被 successor 替换，不能把这次静态复算描述为新的 exact-SHA VCS。

`II=5` 只能接纳为 accepted-beat 几何：一个 tile 必须接收五拍。sealed SVA 的连续五拍 cover 仅命中 1 次；主测试包含 306 个随机 gap cycles，没有证明跨多个 tile、跨 FIFO 深度的持续 wall-clock II=5。因此可写“5 accepted cycles/tile under declared no-backpressure geometry”，不可写成 measured speedup。

## VCS 证据边界

Sealed VCS 本身有效：compile/sim rc 均为 0，21 channels、61 tiles、305 input beats、61 rank results、21 moment results和一次 fail-closed attack 均通过；stall、五拍 tile、±INT8 endpoint、pending-output fault covers 非零。

但有两个边界：

1. moment scoreboard 使用错误的跨 lane scalar grain；
2. contract 声称 ties-to-even 和 saturation reference miter，但 VCS 没有对应 cover/counter。随机数据可能命中边界，现有 receipt 无法证明。

后续应加入正负 half tie、偶/奇 quotient、正负饱和、shift 0/23 的显式 vectors 和计数。

## DC parser 与拒绝结论

DC tool 自身 rc 为 0，但 exact runner 正确以 `exit 41` fail-close：

| 指标 | 独立解析 |
|---|---:|
| cell area | 41,749.973937 µm² |
| cells / sequential | 47,432 / 6,055 |
| logic levels | 88 |
| critical path length | 2.53 ns |
| worst setup slack | 0.0002 ns |
| report 最后一条 setup slack | 0.0765 ns |
| worst hold slack | 0.0000 ns |
| macro | 0 |

100 条 setup/hold path 均按 slack 非递减排列。旧 scratch anchor 把末条 `0.0765 ns` 当成 worst；真实 worst 是首条 `0.0002 ns`，低于预先声明的 `0.05 ns` guard。hold 打印为 `0.0000 ns`，也没有可用裕量。

因此必须拒绝：

- run 明确带 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`；
- failed run 没生成 output evidence manifest；
- ideal clock + ZeroWireload 不是 post-layout；
- 更根本地，41.75k µm² 是错误 scalar-moment 设计，不是 corrected per-lane RTL 的面积。

## Q8、PAFT 和 valid825

Q8 early-requant 位于 current-batch moments 之前，会同时改变 BN mean/variance 与 rank projection。当前只用了 synthetic factor 和单一 global shift，没有 checkpoint-bound rank3 factor quantization、scale contract、PAFT、valid825 或网络 miter。

下一版算法门槛：

1. 按 RTL 顺序训练/校准 `fc1 raw -> Q8 -> per-lane moments + rank3 right projection`；
2. 导出 checkpoint-bound factor、shift/scale 和逐 lane vectors；
3. 在 valid825 上对比冻结 `no_running` baseline，报告精度变化；
4. 在此之前，Q8 candidate 仅是模块设计，不得接纳网络精度。

## 公平 baseline 与速度口径

目前没有可引用的 M163 speedup：

- `36.48M / 21.888M = 1.6667x` 是 dense sn2 对完整 rank3 两段投影的理想 arithmetic count；M163 r1 没有 coefficient generation、dynamic correction、left projection、ATLIF 或 fc2。
- `2.9441x` 是 train-required Q8 candidate 相对公平 full-FFN local intermediate bit movement 的计数比，不是 transaction、cycle、energy 或 speedup。
- 41.75k µm² 没有同资源、同频率、同端口的 dense dynamic-BN baseline，而且设计语义错误。

任何 DATE 图表只能暂时引用：“M163 r1 是一个被 P0 撤销的 frontend 原型；其结构资源和失败原因已测量。”禁止引用 cycle、physical、energy、network、system speedup 或 PPA headline。

## r2 必做

1. 16 组独立 `sum/sumsq/count`，明确 spatial-group tag 与 hidden-lane identity。
2. VCS 用不相等 lane distributions 做逐 lane miter；加入持续无 gap 多 tile II、FIFO 边界、RNE 边界、channel length guard。
3. corrected RTL 重新 sealed DC；parser 对全部 path 取 min，保留 setup ≥0.05 ns guard，不引用旧面积。
4. PAFT/valid825 后才允许 Q8/rank3 accuracy admission。
5. cycle simulator 同时建模 moment barrier、coefficient/rsqrt、rank correction、left projection，并提供相同 product pool/ports/frequency 的 dense dynamic-BN baseline。

复算脚本：

```bash
cd /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07
/opt/anaconda3/envs/python310/bin/python \
  results/m163_independent_hammer_review_r1_20260824/audit_m163_independent.py
```

机器可读结果见 `m163_independent_hammer_review.json`。
