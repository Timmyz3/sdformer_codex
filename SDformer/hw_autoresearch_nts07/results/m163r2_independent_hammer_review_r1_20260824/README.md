# M163r2 独立打铁评审 r1

结论：**66/100，`PASS_STANDALONE_TILE_LOCAL_RTL_REVISE_BEFORE_NETWORK_ADMISSION`，P0/P1/P2 = 2/7/3。**

M163r2 已经修正 r1 把 16 个 hidden channel 混成一个 moment 的 P0 错误。sealed exact-SHA VCS 和 3 ns logic-only DC 均真实可复核；当前可接纳的是一个“输入已经为 Q8”的 tile-local moment + rank3-right frontend，不是 H67 完整动态 BN/ATLIF/FFN，也没有网络速度或 paper PPA。

## 1. BN1 语义

局部 RTL 语义正确：每拍 16 个 hidden lanes，每 lane 两个时间样本；每个 lane 独立维护 sum/sumsq，五拍后每 lane 每 tile 的 count 是 10。共享一个 count 只有在以下合同同时成立时才正确：

- 每个接受拍固定包含全部 16 lane，没有 lane mask 或尾组；
- 物理 lane 在所有 spatial tile 中始终绑定同一 hidden channel；
- 同一 16-channel group 的全部 spatial tile 在 `channel_start` 与 `channel_last` 之间连续出现；
- 每个 lane 的有效样本数完全相同。

RTL 没有 channel base、spatial index、batch index 或 expected extent，只信任 start/last/tag，因此不能发现漏 tile、重复 tile、lane 换绑或错误提前 last。它证明的是局部归约器，不是完整 `no_running` BN1 population。冻结 B=1、T=10 的四级 population/lane 应分别是 `192000/48000/12000/3000`，而 TB 仅跑每组 1–5 个 tile。

## 2. 运算宽度和 II=5

源码和 DC resource report 交叉闭合：

| 项 | 独立复算 |
|---|---:|
| rank3 right products/accepted beat | `3 × 16 × 2 = 96` |
| products/tile | `96 × 5 = 480` |
| square issues/accepted beat | `2 × 16 = 32` |
| Q8 samples/tile | `32 × 5 = 160` |
| rank outputs/tile | `3 × 16 = 48` |
| requant | 16 lanes，3 个 data cycles/tile |

postcompile resource report 有 32 个显式 8×8 square `DW_mult_tc`；另有 48 个 projection datapath operators，每个包含两次 signed 8×8 product，正好是 96。

在 `tile_valid` 连续、`rank_ready=moment_ready=1` 时，raw arrival 每五拍一次，requant 的 load + 三个 data cycles 能在下一 tile 到来前完成，因此 standalone 结构可接纳 no-backpressure `II=5 cycles/tile`。这不是实际 trace 吞吐：sealed TB 有 306 个主动 input gap，SVA 只 cover 一次五拍连续 tile，没有长串 back-to-back tile/channel 或 full-network wall-cycle 计数。

## 3. RNE 缺口

函数本身的符号 magnitude、tie-to-even 和 INT8 saturation 写法合理，`-2^23` 也能装入 25-bit magnitude。但 sealed TB 只配置 `shift=9`，没有 tie/saturation 事件计数，SVA 只 cover 输入 `-128/+127`；这不能支持合同中的“ties-to-even and saturation reference miter=true”边界覆盖声明。

必须补：shift `0/1/23`，正负的 even/odd exact-half tie，half±1，舍入后 `+127/+128/-128/-129`，raw 24-bit min/max，以及非法 shift=24 fail-close。当前只允许说随机 projection 输出与同一参考函数比较无 mismatch。

## 4. DC 数字的真实意义

| 指标 | sealed r2 |
|---|---:|
| cell area | `53,662.139958 um²` |
| cells / sequential | `60,910 / 9,183` |
| combinational / sequential area | `65.501% / 34.499%` |
| logic levels / critical length | `88 / 2.42 ns` |
| setup / hold | `+0.1053 / +0.0001 ns` |
| worst path | `tile_channel_start → channel_sum_q[4][47]` |
| ports / macros | `2,689 / 0` |

这是 TSMC28 NLDM、3 ns、ZeroWireload、ideal clock、0 macro 的扁平 pre-macro cell estimate。2,689 个端口以及 1,664-bit moment payload 的布线、CTS、拥塞、SRAM/寄存器文件和功耗都没有进入数字。hold `+0.0001 ns` 虽然报告 MET，但没有可迁移的物理裕量。因此可引用 logic-only 数字，不可称 paper PPA、Fmax 或物理加速。

## 5. 最大降本点

映射网表的 9,183 个 sequential cells 中，8,704 个（94.784%）可直接归入八个数据 bank：

| bank | FF |
|---|---:|
| channel sum+sumsq state | 1,664 |
| moment sum+sumsq snapshot | 1,664 |
| projection accumulator | 1,152 |
| raw FIFO ×2 | 2,304 |
| quant raw copy | 1,152 |
| output FIFO ×2 | 768 |

第一优先级不是细调门，而是收窄位宽。对最大 population `N=192000` 的任意 Q8 值，精确 universal bound 是：

- signed sum：26 bits（范围需求 `[-24,576,000, +24,384,000]`）；
- unsigned sumsq：32 bits（最大 `3,145,728,000`）；
- shared count：18 bits；
- 十个任意 signed INT8 product 的 projection：`[-162,560,+163,840]`，19 signed bits 足够。

将两套 moment state/snapshot 从 `48/56/32` 收到 `26/32/18` 可省约 1,500 FF；将 projection accumulator + raw×2 + quant copy 从 24 收到 19 bits 可再省 960 FF。合计约 **2,460 FF，即当前 sequential 的 26.79%**，还会直接缩短当前 48-bit critical carry path。必须先绑定 population/factor 上界，再重跑 overflow boundary VCS/DC。

第二优先级是消重：moment state 与 snapshot 完整重复 1,664 data FF；raw FIFO 两份后又复制 1,152 FF 到 `quant_raw`。去掉 moment snapshot 会牺牲“旧 moment stall 时新 channel 可开始”的 overlap，必须增加 channel-boundary consume 合同或 ping-pong；raw bank 可尝试在 requant 三拍期间直接持有/读取，在 ready=1 的 II=5 合同下有充足服务时间。不能在没有 backpressure 复测前直接删 bank。

r1→r2 sequential 增量是 3,128；把 scalar state 与 scalar snapshot 都扩成 16 lane 理论增加 `2×15×(48+56)=3,120` FF，解释了 99.7% 的增量。这只说明成本来源，r1 语义无效，不能拿它做面积优胜 baseline。

## 6. `tile_channel_start` 路径

最坏 setup 从输入 `tile_channel_start` 穿过 start/reset 选择和宽 carry，到 `channel_sum_q[4][47]`。先做 48→26 bit 收窄；若还不够，再比较：

- 显式 `channel_begin`/preclear，在数据拍前初始化 accumulator；
- 将 beat reduction 与 state accumulation 分级流水；
- 双 bank/epoch ownership，使 start 只切 bank pointer，而不驱动每个宽 D mux。

任何方案都要保持 back-to-back tile II 与 channel barrier 行为，不能只看 STA。

## 7. 公平 baseline、PAFT 与声明边界

M163r2 只实现 rank3 right half；dense `sn2` 的完整 T×T 输出与 rank3 的 correction + left projection 不对称。把 `30` 个 right products 与 dense `100` 个 products 比成 `3.33×` 是不公平的；完整 rank3 理论是 right30+left30，相对 dense100 的 `1.667×`，但也必须在 PAFT 接纳 rank3、实现 correction/left、并加入相同 moment/barrier/memory 后才能成为周期候选。

M161 correction 保留的 Q8 movement `6.1842×`（BN1）和 `2.9441×`（加共同 BN2）只是 local bit-movement sensitivity，不是 transaction、cycle、energy 或 speedup。

PAFT 的硬门不是“训练跑完”，而是：

1. exact-SHA 冻结 12 个 FFN 的 rank3 L/R factors、factor row sums、fc1→Q8 per-layer scale/shift、RNE/saturation 和 threshold bridge；
2. 训练数据与 valid825 严格隔离，不能复用已撤销的 valid825 catalog；
3. 用冻结 `no_running` baseline 预声明 AEE/AAE、per-sequence tail 和 spike/overflow 门槛，再跑 valid825；
4. hardware-order golden 对 fc1→Q8、moments、correction、left/threshold 全链逐层对齐。

在这些门通过前，`paft_valid825/network_cycle_speedup/physical_speedup/paper_ppa/system_speedup/headline` 全部保持 false。

## 8. 给 M164 的顺序

1. 先收窄为 universal Q8 `sum26/sumsq32/count18/projection19`，补最大 population 和完整 RNE 边界 VCS，再做 DC。
2. 同时补 address-bearing population/barrier/replay 软件 recurrence；没有它不要报 network cycles。
3. 做 moment ownership 与 raw→requant bank fusion 两个消重 DSE，明确 ready/backpressure 和 channel overlap。
4. 把 `tile_channel_start` 从宽 carry path 移开。
5. PAFT 通过后，才在同一个周期模拟器比较完整 dense no-running BN+sn2 与完整 rank3 right+correction+left。

机器可读裁决见 `m163r2_independent_hammer_review.json`；所有数字由 `independent_recompute_m163r2.py` 从 sealed 原始报告和 mapped netlist独立复算。
