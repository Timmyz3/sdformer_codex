# M166r2 prefold rank3-left + ATLIF backend 独立打铁评审 r1

结论：**84/100，`PASS_CENTER_CORRECT_STANDALONE_BACKEND_GO_M167_SHARED_POOL_WITH_P0_GATES`，P0/P1/P2 = 4/6/3。**

M166r2 是可信的、center-correct 的 standalone backend 里程碑。canonical r2 的 RTL SHA 同时绑定 r2 VCS 和 DC；241 tile、1205 beat 的 VCS scoreboard 逐拍核对 tag、channel-last、beat 和 32-bit event word，七项 cover 全部命中。独立有限状态枚举覆盖 172 个可达控制状态、688 条任意 input-demand/output-ready 转移，没有 FIFO 越界或 reservation 反例，持续流的 tile start 间隔恒为 5 cycles。

但是当前只能接纳“5-cycle local backend recurrence”和同一 3 ns 流程下的 logic-only area/timing，不能接纳完整 FFN 加速。**M167 应做 FRONT/BACK/PREFOLD 三模式单 96-slot 共享池**；它是下一步最高价值硬件项。更高的科学 P0 是动态 `L'=alpha*L` 的位宽/重平衡、rsqrt、rank epoch SRAM/barrier、ATLIF threshold 幅值到 fc2 的数值顺序，以及 PAFT/valid825 与完整公平周期。

## 1. r1 center 漏项与 r2 身份

r1 的说明公式少了 `-center[t]`：

```text
r1: bias'[t,l] = offset[l]*(L*R*1)[t] + temporal_bias[t]
r2: bias'[t,l] = offset[l]*(L*R*1)[t] + temporal_bias[t] - center[t]
```

这不是可忽略项。`ATLIFTernaryPSN.forward` 在 `center_mode != zero` 时先从 temporal affine 结果减去 `center`，再做 threshold。独立使用整数随机张量复算 64 组、51200 个输出点，r2 prefold 与直接式最大误差为 0；故意删除 center 后，51200 个点全部产生非零见证。

身份链也已收口：

- r1 sealed input RTL SHA：`6ad1ce4551a1550cf651c78321189d8793a4f84c5570084bb407ac035195c6c8`；
- canonical r2 RTL SHA：`9afaf28c92f344c8a1cc0126226579b842420bda4d48f8ddcc26458c86f2d646`；
- r2 VCS receipt SHA：`bde519f9ec0b9110d4bb7d66dab89ebc2a8f1c59c12045dc1d0c8d01e794c5db`；
- DC receipt SHA：`2c43b532d1862a0d780c9ad76a00a75f5ae9dab813650b6b2a2cd7f0a0b377ad`。

当前 RTL、r2 VCS 和 DC 都绑定 r2 SHA。RTL 本来就把外部 `folded_bias` 当通用 Q24 数值原样相加，所以改动修复的是 coefficient identity/comment/contract，不是 backend 加法器。r1 的 directed functional observation 可保留作旁证，但不得再作为 canonical algebra identity。

## 2. VCS scoreboard、七项 cover 与反压安全

sealed r2 compile/sim rc 均为 0，13 个 assertion property 没有 failure signature：

| 项目 | 数值 |
|---|---:|
| tiles / output beats | 241 / 1205 |
| signed products | 115680 |
| slots / service cycles per tile | 96 / 5 |
| steady II=5 hits | 63 |
| input push + release overlap cycles | 237 |
| output stall cycles | 275 |
| mixed event words | 1205 |
| protocol attacks | 1 |

七项 SVA cover 的 match 数分别是：

| cover | matches |
|---|---:|
| five-cycle tile | 241 |
| back-to-back five-cycle tiles | 147 |
| input push + release same cycle | 237 |
| full owned input FIFO | 1195 |
| event stall then accept | 208 |
| mixed event word | 1480 |
| fault after configuration | 2 |

controller 在开始一项服务前为 output FIFO 一次性预留 5 个 beat；服务中每拍只 push 一项，phase 4 才 release 输入。独立枚举允许每个状态下任意 input request 和 event ready，验证 output count 始终不越过 16、input count 不越过 2、phase 0..4 连续推进，并到达 full FIFO、反压、same-cycle release/push 和 back-to-back service。

这个证明范围是 standalone synthetic Q8 backend。它不包含 checkpoint coefficient、rsqrt、epoch address、fc2 或完整 FFN。

## 3. source/netlist 资源账本

source 的 96 个 signed 8x8 products 在 DC `resources_postcompile.rpt` 中对应 96 个唯一 `mult_163[_G*]` operation，并非只靠注释申报。

寄存存储逐项与 mapped DFF 完全一致：

| 类别 | payload | metadata | total bits |
|---|---:|---:|---:|
| folded-left + folded-bias + threshold config | 7680 | 24 | 7704 |
| 2-entry rank input FIFO | 768 | 34 | 802 |
| 16-entry event output FIFO | 512 | 320 | 832 |
| 合计 | 8960 | 378 | **9338** |

另外有 23 个 control DFF，所以 `9338+23=9361`，与 DC sequential cell count 完全相等。原 DC precontract 的 `input_rank_fifo_bits=768` 和 `output_event_fifo_bits=512` 只指 payload；引用总存储时必须写 802 和 832，避免漏掉 tag/last/beat。

## 4. DC 结果与 claim 边界

| 指标 | M166r2 standalone backend |
|---|---:|
| cell area | **44280.053123 µm²** |
| cells | **35675** |
| sequential cells | **9361** |
| logic levels | **47** |
| critical path | **2.30 ns** |
| critical path | `service_phase_q_reg[2] -> output_bits_mem_reg[11][23]` |
| setup / hold slack | **+0.4672 / +0.0000 ns** |
| macros | **0** |

这是 Synopsys DC V-2023.12-SP3、TSMC28 HPC+、3 ns request、ideal clock、ZeroWireload、0 macro 的 flattened logic-only 结果。五类 constraint report 都无 violation，但 hold 只在四位小数上等于 0，不能迁移成物理裕量；也没有 CTS、extracted parasitics、SAIF/PTPX 或 Formality。

### 为什么不能把 M165 和 M166 面积直接相加

M165 standalone frontend 和 M166 standalone backend 各自拥有一套 96-product pool，也各自带 FIFO、寄存式配置、接口和控制。把两份 cell area 数字相加，最多表示一个未实现的 duplicate-pool 上界，**不能表示计划中的复用式 accelerator PPA**，更不能证明 composition timing。

只有把 FRONT/BACK/PREFOLD 做进一份 RTL、重新综合，才可声称共享池面积。M167 还必须把 M165 的 32-square moment sidecar 单列；它不属于这 96 个 temporal products，不能偷偷消失。

## 5. prefold 代数与条件性周期边界

正确实数代数是：

```text
v[r,p,l]       = sum_tau R[r,tau] * x[tau,p,l]
L'[t,r,l]      = alpha[l] * L[t,r]
bias'[t,l]     = offset[l] * (L*R*1)[t] + temporal_bias[t] - center[t]
h[t,p,l]       = sum_r L'[t,r,l] * v[r,p,l] + bias'[t,l]
```

对一个 16-hidden-lane group：

- `L'`：`10×3×16=480` products；
- `bias'` 的 `offset*(LR1)`：`10×16=160` products；
- 合计 640 products，96-slot 容量是 `ceil(640/96)=7` cycles。

每 tile 的纯 temporal products：

| 路径 | products | 96-slot 容量下界 |
|---|---:|---:|
| dense T10 | 1600 | 17 cycles |
| rank3 right | 480 | 5 cycles |
| rank3 left | 480 | 5 cycles |
| rank3 合计 | 960 | 10 cycles |

因此 product count ratio 是 `1600/960=1.666667×`，ideal capacity boundary 是 `17/10=1.7×`。二者都只是条件边界：dense 的 17 是容量下界，rank3 的 10 没含 rsqrt、coefficient generation、moment barrier、rank SRAM write/replay、fill/drain、fc2 和 memory。**不得写成 measured FFN、network 或 system speedup。**

## 6. 新发现：M166 event bit 丢了 ATLIF 非零幅值

这是本轮新 P0。软件 `OfficialATLIFSurrogate` 的输出是 `{0,+threshold}`，实际代码返回 `active * thre`；M166 的 `event_bits` 只保留 comparator bit，没有携带 threshold amplitude，且模块明确不包含 fc2。

因此进入 fc2 前必须二选一：

1. 从 bit 恢复 `{0,threshold}` 的定点幅值；或
2. 把 `threshold × activation scale` 折进 fc2 weight/scale，并对 checkpoint 证明 bias、accumulator width、RNE、saturation 和 output commit 顺序完全等价。

在这个证明之前，不能把 event bit 当成 fc2 的数值输入，也不能声称完整 FFN equivalence。若每个 FFN 的 threshold 是冻结 scalar，weight fold 很可能是更省硬件的路径，但仍须与量化 scale 和舍入点一起证明，不能只做浮点代数替换。

## 7. P0/P1/P2

P0：

1. `L'=alpha*L` 没有 checkpoint-bound INT8 范围、factor rebalance、精确 requant 和 valid825；
2. event bit 未保留 ATLIF `{0,threshold}` 幅值，fc2 weight/scale fold 尚无等价证明；
3. population variance、epsilon、rsqrt、alpha/offset 和 640-product prefold generator 缺失；
4. 没有 rank/coefficient epoch address、global barrier、fc2/BN2/residual commit 和公平完整 FFN 周期。

P1：

1. M165/M166 不能直接面积相加，必须 M167 共享池重综合；
2. `1.7×` 只是 capacity boundary；
3. 7704/9361 个 FF 是寄存式 coefficient envelope，尚无 SRAM macro；
4. 缺 Formality；
5. ideal-clock/ZeroWireload/0 macro 且 hold rounded zero；
6. PAFT/valid825 未绑定。

P2：

1. precontract FIFO bits 标签只统计 payload；
2. 单 seed synthetic coefficient，没有 numeric rail campaign；
3. opaque tag 不能认证 stage/group/spatial/coefficient epoch。

## 8. M167 明确裁决

**GO，做 FRONT/BACK/PREFOLD 三模式单 96-slot 共享池。** 推荐合同至少包含：

1. 每拍 96 slots 的唯一 mode owner 和 accepted-product 账本；
2. FRONT 保留 M165 五拍 right projection，32-square moment sidecar 独立计面积/功耗；
3. PREFOLD 不预设一定 8x8，先绑定 alpha/offset/L/R 位宽；若变宽，重新综合或明确 lane-serial schedule；
4. BACK 保留 M166 五拍 left + comparator；
5. rank epoch、coefficient epoch、空间地址和 barrier fail-close；
6. threshold amplitude/fc2 fold 二选一的硬件顺序；
7. matched shared-pool versus duplicate-pool DC，以及完整 dense versus rank3 address-timed cycle simulation。

这样 M167 的创新点不是“又一个计算模块”，而是利用动态 BN 的天然 barrier，把 right、coefficient prefold、left 三个互斥阶段映射到同一物理乘法池，同时消除 dense activation materialization。它是把目前两个可信 standalone 模块收敛成可发表 operator architecture 的正确方向。

机器可读裁决见 `m166r2_independent_hammer_review.json`；fresh 复算见 `fresh_recompute_m166r2.json`，可由 `independent_recompute_m166r2.py` 只读复跑。
