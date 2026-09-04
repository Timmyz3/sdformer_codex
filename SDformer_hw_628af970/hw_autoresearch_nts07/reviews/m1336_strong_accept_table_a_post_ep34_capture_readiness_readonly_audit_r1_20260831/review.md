# M1336｜Strong-Accept Table-A post-capture readiness audit

## 结论

当前 Table-A 仍是 **0 条系统行**。最新 ep34 capture 一旦成功，也不会自动把系统表补齐：它直接保留的可执行 payload 只有四层 C1 Conv3x3、四层 decoder ConvTranspose2d，以及 attention Q/K/gate NPZ 子集。FC1、FC2、动态 BN、ATLIF、prediction head、其余 Conv/投影与完整 attention 仍缺精确 payload，只能先做统计/解析估算，不能冒充同分母系统重放。

因此最短路径不是再开硬件机制，而是：**capture 准入 → 身份/权重绑定 → 缺失算子 fallback 或补 payload → 六行统一重放 → 17 SRAM+DRAM 能量 → 独立 hammer → Table-A**。

## 当前第一道门

- 本机没有 canonical M1327 结果，`capture_complete=false`。
- M1333 被独立审阅判定 `FAIL_DO_NOT_CITE`；M1335 已修五类 false-negative，但目前只是 source author PASS，仍需 fresh different-author blind hammer。
- M1118 component annex 可用，但 canonical full-system rows 仍为 0。

## 逐算子 readiness

| 算子 | capture 后状态 | 证据边界 | Table-A 前必须补的动作 |
|---|---|---|---|
| Patch embed | 只能 analytic | 8 个 ordered module 仅统计，无 retained tensor | exact payload/adapter，或每行同收费的冻结 fallback |
| C1 四层 Conv3x3 | 身份绑定后可重放 | 有 compressed FP32 + support/sign payload；C1 simulator 已存在 | 绑定 ep34 权重/bias；当前只有 Zurich 09 的 10 sample，不满足三序列系统人口 |
| 其余 Conv/投影 | 只能 analytic | 只有 ordered statistics | exact payload/adapter 或同收费 fallback |
| FC1 | 缺 payload | 12 个 FC1 记录/样本仅统计 | materialize exact source/value stream |
| 动态 BN | 缺 payload | 78 个 BN 记录/样本无 value/state stream | exact value/state 或统一 fallback |
| ATLIF | 缺 payload | 93 个 live module 与累计 activity，不是 per-timestep signed stream | materialize 时序值；M928 只能证明 setup/area |
| Attention Q/K/gate kernel | 严格 schema hammer 后可重放 | 480 个 NPZ | M1335 blind PASS 后重放 RQTB/kernel |
| Attention QKV/projection/completion | 缺 payload | NPZ 子集不是完整 attention | exact payload/adapter 或统一 fallback |
| FC2/C2 | 缺 payload | 当前 C2 是 5 个 directed workload，不是 final 40-sample stream | materialize final FC2 stream |
| Decoder D0–D3 | 身份绑定后可重放 | 三序列×10 sample；M1328 可产 120 calls/240 bitplanes | 绑定权重/bias、D1 dynamic-theta miter；用 M1111DR2 successor 对 B0/B1/B3/Ours 重放 |
| Prediction head | 缺 payload | 只在 aggregate runtime stats 中出现，不在 retained ordered payload | exact payload/adapter 或统一 fallback |
| Preprocess/completion | 只能 analytic | 尚无全覆盖 final-bound executable authority | 冻结并在每行完整收费 |

## 已准入组件数字与禁止外推边界

| 组件 | 可引用数字 | 不能写成什么 |
|---|---|---|
| C1 | CPU 同账本 434,242,823 / 763,908,050 = **1.75917×**；容量 214,912 B < 240 KiB | 不是 RTL/system speedup；M1006 仅 component setup/area，hold、完整存储与 power 未闭合 |
| C2 | K8 vs K1×8：**1.01673× cycle、4.54108× throughput/mm²、logic area −77.6104%** | 五个 directed workload、logic-only pre-macro；不能当 final FC2/system energy |
| C3 | 62,433.503388 µm²，3 ns setup min +0.0003 ns | 无加速比、hold、power、final activity |

三个组件倍率不得相乘。它们只能作为 component annex，直到统一 simulator 产生同一 fixed numerator 的系统行。

## B0/B1/B3/Ours 同分母门

所有行必须固定：28 nm、3 ns、96 source lanes、240 KiB、Acc24、64 GB/s decimal DRAM（192 B/cycle），并显式冻结 queue depth、bank 数、port mode、external read/write ports。必须同时有 B0、B1、B2、B3、C2、Ours 六行；B0/B1 是 headline denominator，B3 是紧邻 equal-service baseline。

同一 final checkpoint、decoder-complete operator scope、同一多序列人口、同一吞吐 numerator；任何 fallback 仍要在每行支付 cycle/traffic/energy。禁止把局部 1.759×、1.017× 或外部 Prosperity 机会相乘成系统倍速。

## 三序列与能量权威

- Decoder payload 已有三条序列；C1 retained payload 只有一条序列。现状不能据此构造同一三序列系统行。
- 尚缺预先冻结的 low/medium/high density strata 与 aggregation weights。
- Table-A 需要 exact 17 SRAM（8 weight + 8 state + 1 parent）与 DRAM 的统一能量权威；要求 native mapped SAIF coverage ≥95%，并绑定 DC/Formality/PT/SAIF/PTPX、每宏 internal/switching/leakage、SRAM access 与 DRAM transaction energy。
- C1 的 9 个 local parent macro、105-macro static capacity equivalent 都不能替代系统 17-macro 权威。

## 唯一最短 post-capture DAG

1. canonical M1327 一次 capture 完成，禁止 retry。
2. 用 pinned interpreter 跑 M1335，并取得 fresh different-author blind PASS。
3. 冻结唯一 measurement identity：ep34 checkpoint/config/profile、decoder-complete scope、fixed numerator、多序列 density strata 与 aggregation weights。
4. 绑定现有 payload：C1、M1328 decoder+D1 miter、attention Q/K/gate；绑定 exact weights/bias/file identity。
5. 对全部缺失算子补 exact payload/adapter，或冻结一个对六行同收费的 executable fallback；不开新加速机制。
6. 在同一资源 tuple 下对 B0/B1/B2/B3/C2/Ours 做唯一 unified address-timed replay，产出 cycles、stalls、SRAM/DRAM transactions、fixed numerator、逐序列/密度分层。
7. 绑定 17-macro native physical/activity、SRAM/DRAM energy 与 final accuracy。
8. different-author hammer 完整 bundle；发布 additive M698/M653 registry authority，才填 Table-A。

## P0 缺件

1. canonical M1327 + M1335 blind PASS。
2. C1/decoder final ep34 weights/bias identity 与 D1 miter。
3. FC1/FC2/BN/ATLIF/patch/head/attention projection 等缺失算子的 payload 或统一 fallback。
4. C1 在冻结三序列系统人口上的证据。
5. 六行同分母地址计时结果。
6. density strata/weights。
7. 17 SRAM + DRAM energy 和 mapped activity。
8. final accuracy + Table-A independent hammer。

结论保持 fail-closed：**component annex ready；system Table-A not ready；Strong-Accept system claim not ready；不开新 mechanism。**
