# M367：自然 source-stream cumulative-budget gate CPU fast-kill

结论：**NO-GO A800 valid、RTL 与 Synopsys。** 这是 M352 明确指出 M341
未覆盖的一条新执行路由，但完整 frozen scope 上没有一个可实现模型达到
1.15x。

## 新路由是什么

每个 destination group 的 4-bit conservative beta code 不再排序、也不进入
16 个桶。active source 按自然 source-ID 顺序每拍最多 8 个；对每个 source
依次执行：若 `S+U<=B` 则 drop 并更新 `S`，否则 keep，但仍继续考虑后面的
source。B0 硬旁路。

因此它不是 M341 stable-bucket route 的重复实验；但若想获得周期收益，仍
必须把 gate 后的任意 8-active word 压入八个 modulo-bank issue FIFO。没有
lookahead/repack 时，drop 只形成原 K8 timeline 的 bubble，周期是 1.0x
（加 15,696-cycle metadata prefetch 后为 0.999998x）。

## 冻结 CPU 结果

范围完全复用 M328/M341 group4：10 个 FC1、一个 selected patch Conv、110
个 record、1,013,760,000 个 task。B0 精确复现 6,681,676,272 K8 cycles；
所有非零预算均满足
`|raw INT8 error| <= sum exact beta <= sum U <= B`，violation=0。

| B | Drop | 免费 compaction | 同拍组合 8R | 注册 8R+D8 | 8-bank 1R+D8 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.0952% | 1.0007x | 0.9734x | 0.8591x | 0.5428x |
| 32 | 1.5993% | 1.0133x | 0.9820x | 0.8659x | 0.5439x |
| 64 | 3.1030% | 1.0299x | 0.9910x | 0.8747x | 0.5461x |
| 128 | 5.4779% | 1.0573x | 1.0041x | 0.8884x | 0.5493x |
| 256 | 10.1017% | **1.1145x** | 1.0265x | **0.9112x** | 0.5544x |
| 512 | 18.6690% | **1.2369x** | 1.0486x | **0.9388x** | 0.5610x |
| 1024 | 33.4481% | 1.5259x | **1.1064x** | 1.0061x | 0.5762x |

B256 连免费 compaction 都达不到 1.15x。B512 的免费机会能过线，但一旦
按自然 8-active word 的到达顺序收费，即使给八个任意 code 读口、同拍
八 lane greedy、同拍 route、无限 queue，也只有 1.0486x；注册 D8 后为
0.9388x。B1024 的最乐观同拍模型也只有 1.1064x。

分 scope 的 B512 结果同样不过线：FC1 组合 8R=1.0634x、注册 D8=0.9376x；
selected Conv 组合 8R=1.0126x、注册 D8=0.9419x。

## 严格误差界

4-bit codebook 始终满足 `U>=beta=max_j|Wq_j|`。对自然 greedy 实际 drop
集合 D 和任意 destination j：

`|sum_(i in D) Wq[j,i]| <= sum_(i in D)|Wq[j,i]| <= sum beta_i <= sum U_i <= B`。

B256/B512/B1024 的最大观测 raw INT8 error 分别为 256、508、983；最大
conservative bound 分别为 256、512、1024。该证明只覆盖 raw quantized
accumulator，不等于网络精度。

## 硬件成本边界

- persistent beta metadata 为 498,816 B，是 one-bit reference 的 4.0x；
- 最大 group scratchpad 为 432 B。要支持八个任意 code 读，或需真正 8R
  RF/mux，或复制成八份 1R、共 3,456 B；
- D8 为八个 bank 各 8 个 10-bit source ID，共 640 payload bits；D16 为
  1,280 bits，尚未含 valid、head/tail、task tag 与 skid word；
- selector 包含八个 code decoder、八级依赖的 11-bit add/compare/select
  greedy 链、每 bank lane count、8x8 route/compactor 和 B0 bypass；
- 廉价的八 bank 单读 code SRAM 出现 5,667,875,712 个额外 read-conflict
  cycles，所以更慢。8R 端口和八级组合链尚无 DC timing/area admission。

## 决策与口径

M367 证明的是：**M341 未覆盖的自然流、task-local 8-active repack 路由也不
值得进入 A800 valid。** 它没有证明所有 cumulative-budget 架构均失败；
跨 task 多 accumulator、每 bank 多 lookahead 或离线静态重排是新资源模型，
必须给 baseline 同等资源后另立合同。

GO 封存 CPU opportunity、自然 greedy 整数界和这条 route 的 NO-GO；
NO-GO A800、RTL/VCS、新思、系统倍速或论文贡献。`docs/359` 未修改。

B0 行中的 scan/conflict 是 producer 计算的 population diagnostics，硬旁路
总周期没有加它们；`b0_exact` 只在 B0 行有语义，非零行不得引用该字段。
