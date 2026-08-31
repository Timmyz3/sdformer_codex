# M165 owned-raw-bank 独立打铁评审 r1

结论：**86/100，`PASS_MODULE_LEVEL_OWNED_RAW_BANK_AB_PRIORITIZE_CONDITIONAL_BACKEND`，P0/P1/P2 = 2/6/3。**

M165r2 是可信的 M164 模块级后继。修正版 VCS 身份把总样本账本收口到每 hidden lane **192700**；direct raw-bank ownership、满 FIFO、同拍 rank2 release + producer push、输出反压、最大 H67 population 和 sticky fail-close 都有可核证据。matched logic-only DC 也确认 `quant_raw` 的 912-bit 复制在 mapped netlist 中完全消失。允许引用的是同流程 standalone module A/B，不是 BN/FFN/network speedup。

下一步裁决很明确：**转向动态-BN coefficient prefold + rank3-left + binary ATLIF 的 5-cycle backend，不再单独开 moment-snapshot 或 single-raw-bank 面积里程碑。** 后端能补上当前最大的科学 P0，snapshot 约 1k FF 的潜在节省应在 barrier/coefficient wrapper 内顺手解决。

## 1. M165r2 修正身份与 VCS

当前 production RTL、SVA、修正版 TB SHA 分别为：

- RTL `3aee6d899ed79b2f5abd51c1438795e02fb4b2663a765067aa9b14142e46bb0f`；
- SVA `d2f1570c0384ee9dc7c102778d20a576b60a1288f1599f8c1db5a9796b867d99`；
- M165r2 TB `da8e7722af848ac079361c45eb9f759f382bdbd3f899c8395d66199c4c20065c`。

sealed compile/sim rc 都是 0，无 assertion failure。修正后的账本三路一致：

```text
96350 accepted beats / 5 = 19270 tiles
19270 tiles * 10 samples/lane/tile = 192700 samples/lane
3083200 total Q8 samples / 16 lanes = 192700 samples/lane
```

因此旧 M165 r1 的 `192710` 和旧 M164 的 `192650` 都禁止引用；correction overlay 中的 M164 `192640`、M165r2 `192700` 才是有效总数。单个最大 channel 仍严格为每 lane `192000` samples，得到 `sum=-24576000`、`sumsq=3145728000`、`count=192000`。

关键覆盖与检查：

| 项目 | M165r2 |
|---|---:|
| five-beat tile cover | 19209 |
| full raw FIFO while owned | 9 |
| same-cycle raw push + rank2 release | 1 |
| rank / moment stall cycles | 2717 / 9 |
| half-even / saturation / shift23 directed checks | 12 / 6 / 3 |
| maximum-population cover | 1 |
| protocol attack | 1 |

## 2. direct raw-bank ownership 是否真的正确

`raw_count` 的语义是“FIFO 中全部未退休项，包含 requant 正在占有的 bank”。服务开始只锁定 `raw_rd_ptr`，不减 count、不前移指针；rank2 写入 output FIFO 时才 `raw_release` 并前移 read pointer。

满 FIFO 时 `wr_ptr==rd_ptr`。若 rank2 commit 与新 tile 的 final beat 同拍：

1. requant 的 RHS 从 `raw_mem[rd_ptr]` 采旧 owned data；
2. producer 的 nonblocking write 把新 projection 写入同一 bank；
3. read/write pointer 同时前移，count 保持 2；
4. 下一项顺序仍为旧 waiting bank，再到新 bank。

独立抽象控制器允许任意合法 producer push 与 `rank_ready`，遍历 40 个可达状态、160 条非确定转移；raw/output count、bank order、owner=head、reserved output capacity 均无反例，并独立到达 full release+push 状态。VCS scoreboard 同时逐 tile 比较实际 rank packet，若同拍读到了新写值而非旧值会立刻次序/数值失败。

当前 SVA 只直接观察 pointer/count/cover，没有把 owned bank 内容稳定性写成 property；这是 P2，下一身份应增加 bank write-address witness 或内部 bind assertion。

## 3. II=5 与 single raw bank

M165 每 accepted beat 做 `3 rank × 16 lanes × 2 time = 96` 个 signed products 和 32 个 squares；五拍完成一 tile。requant 的三 rank 数据周期加一个 start 周期，低于五拍输入间隔。

独立 cycle model 的结果：

| raw depth | rank 无反压，19200 tiles input stalls | 前 40 cycles rank stall，128 tiles input stalls | drain cycles |
|---:|---:|---:|---:|
| 2 | 0 | 19 | 664 |
| 1 | 0 | 24 | 669 |

所以一个 raw bank 的确不破坏 no-backpressure II=5，但会丢掉一个完整 tile、即五个周期的 burst elasticity。当前没有真实集成 backpressure 分布，不应为最多约 929 个 data/tag/last FF 盲目删除已验证的 depth-2 弹性。

## 4. mapped netlist 与 matched DC A/B

M164/M165 使用同一 TSMC28 library、3.0 ns constraint、flattened `compile_ultra`、ZeroWireload、ideal clock、0 macro：

| 指标 | M164 | M165 | 变化 |
|---|---:|---:|---:|
| cell area | 42376.823933 µm² | 39568.535846 µm² | **-2808.288087，-6.626943%** |
| cells | 45824 | 42174 | **-3650，-7.965258%** |
| sequential | 6723 | 5811 | **-912，-13.565373%** |
| combinational area | 28823.255715 µm² | 27853.559658 µm² | -969.696057 |
| noncomb area | 13553.568218 µm² | 11714.976188 µm² | -1838.592030 |
| logic levels | 34 | 34 | 0 |
| critical path | 2.02 ns | 2.04 ns | +0.02 ns |
| setup slack | +0.7492 ns | +0.7305 ns | **-0.0187 ns，仍 MET** |
| hold slack | +0.0000 ns | +0.0000 ns | MET，无可迁移裕量 |
| ports | 1939 | 1939 | 0 |

mapped netlist 的 `quant_raw_q_reg` 从 912 降到 0；所有其他审计的 named FF prefixes 数量都不变，总 sequential 也恰好 `6723-5811=912=3×16×19`。因此 FF 减少不是综合器偶然优化，而是完全由目标 copy 删除解释。面积还额外少 969.696057 µm² combinational mapping。

关键路径从 M164 的 `beat_expected -> projection_acc` 移到 M165 的 `beat_expected -> raw_mem`，direct-bank write 付出 0.0187 ns setup margin，但 3 ns 下仍有 0.7305 ns。相对语义有效 M163r2 的累计同流程变化为：

- area `53662.139958 -> 39568.535846 µm²`，**-26.263589%**；
- sequential `9183 -> 5811`，**-36.720026%**；
- cells `60910 -> 42174`，**-30.760138%**；
- setup slack `0.1053 -> 0.7305 ns`，+0.6252 ns。

这只能写成 pre-macro logic-only module A/B。没有 CTS、parasitics、SAIF/PTPX、Formality 或 SRAM macro，不能写 physical PPA、Fmax、energy 或 speedup。

## 5. 为什么下一步应做 prefold + left + ATLIF

对 16-hidden-channel group，在 moment barrier 后每 lane 得到动态 `alpha[j]` 和 `offset[j]`。实数代数可重写为：

```text
v[r,p,j]       = sum_t R[r,t] * x[t,p,j]
L'[j,t,r]      = alpha[j] * L[t,r]
bias'[j,t]     = offset[j] * (L*R*1)[t] + bias[t] - center[t]
h[t,p,j]       = sum_r L'[j,t,r] * v[r,p,j] + bias'[j,t]
```

也就是说，没必要在每个 rank/spatial state 上先做 `alpha*v + offset*R1` 再 left projection。`alpha` 可以一次性折进 lane-specific `L'`，`offset` 可以一次性折进 lane/time `bias'`。若 L/R 为冻结 checkpoint factors，`(L*R*1)[t]` 可离线或每次 config 仅计算一次。

每 16-lane group 只需生成：

- `16×10×3=480` 个 `L'`；
- `16×10=160` 个 `bias'`；
- 共 640 次 runtime prefold multiplication；用 96-slot pool 需 7 个 product cycles，另加 alpha/offset 的 rsqrt latency。

冻结四 stage 中最小 group 也有 300 spatial positions，left replay 为 `300×5=1500` cycles。故 7-cycle prefold 最坏只是 left replay 的 0.4667%，且在 barrier 后、replay 前支付，不破坏 replay 内部 II。

### 条件性 5-cycle backend

每 tile 有 48 个 rank values（3 rank × 16 lanes）。把它们装入本地寄存器后，五个周期分别产生两个 time rows：

```text
2 time outputs × 16 lanes × 3 rank products = 96 products/cycle
2 time outputs × 16 lanes = 32 bias-add + threshold decisions/cycle
```

因此 left+binary-ATLIF **有条件地**可达 5 cycles/tile。right 也是 5 cycles/tile，完整 rank3 为 10 cycles/tile。相同 96-slot pool 下，dense T10 temporal matrix 每 tile 要 `10×10×16=1600` products，纯容量下界是 17 cycles；`17/10=1.7×` 只是实现前的条件性 cycle boundary，product count 比为 `1600/960=1.6667×`，均不是当前 measured speedup。

5-cycle backend 必须同时满足：

1. 48 个 Q8 rank values 只读一次并在五周期内复用；
2. 每拍供给 96 个 lane-specific `L'`、32 个 `bias'`；
3. 有 32 条 add/ATLIF threshold lane；
4. alpha/offset、rsqrt 和全部 prefold 在 replay 前完成；
5. rank-state SRAM/address schedule 能每五拍供一个 tile；
6. downstream backpressure 不破坏该 standalone recurrence。

H67 FFN 的 binary ATLIF 推理不是膜电位递推；源码先做 temporal affine，再逐元素比较冻结 threshold，输出 `{0, threshold}`。硬件必须保留 `center_mode` 的 `bias-center` 语义、threshold 的 binary point 和输出幅值，不能只吐一个未经标定的 bit。

## 6. 数值与 accuracy 门禁

上述 prefold 在实数域成立，不能自动继承 96 个 INT8 product slots。动态 `alpha` 可能使 `L'=alpha*L` 超出 signed INT8；改变乘法结合与 requant 点也会改变 threshold 边界。backend admission 前至少需要：

1. exact-SHA PAFT rank3 L/R factors 和 valid825 通过；
2. 冻结 `fc1 raw -> Q8` 的 per-layer scale、RNE、saturation；
3. 冻结 population variance、epsilon、rsqrt、alpha、offset 的格式与舍入；
4. 对 12 个 FFN 做 factor rebalance，证明全部动态 `L'` 可用 signed INT8 表示；否则 widened multiplier 必须重新做资源/周期；
5. 证明 `bias'` accumulator、`bias-center`、saturation、threshold compare 和输出 amplitude 的顺序；
6. 用 hardware-order golden miter 串起 Q8 input、moments、prefold、left projection 和 binary ATLIF；
7. 加 address-bearing barrier/replay，再与同资源 dense dynamic-BN temporal backend 比 cycle。

PAFT/valid825 未通过时，可以先做 synthetic-factor standalone RTL/VCS/DC，声明条件性模块行为；不能接纳 H67 accuracy、完整 FFN 或 speedup。

## 7. P0/P1 与最终优先级

P0：

1. 还没有 PAFT/valid825、checkpoint-bound rank3/Q8 数值身份；
2. 当前 measured block 仍缺 coefficient/rsqrt、rank-state replay、left projection、ATLIF 和公平 dense baseline。

P1：

1. M165 DC precontract 直接绑定旧 r1 VCS receipt；虽有相同 RTL SHA + metadata-only overlay + r2 seal 的完整桥，仍应加 post-run admission overlay；
2. 无 Formality；
3. logic-only、hold 0.0000、无 CTS/parasitics/power；
4. 最大 population 已命中，但没有 post-max overflow attack 和 expected extent/address authentication；
5. II=5 尚未纳入 rank SRAM/barrier/backend；
6. 动态 `L'` 尚未证明能保留 8×8 slot 合同。

moment snapshot 的 data+count+tag 上界约 961 FF，single raw bank 的 data+tag+last 上界约 929 FF，都是可见面积机会，但它们不会补齐性能主张。下一里程碑顺序应为：

1. **coefficient prefold + rank3-left + binary ATLIF backend**；
2. 在这个 barrier wrapper 内做 owned/lane-serial moment consumption，顺带消掉 snapshot；
3. 保留 two raw banks，等真实 consumer backpressure 后再决定是否改单 bank；
4. PAFT 门通过后做完整 dense-vs-rank3 cycle simulator 与 hardware-order miter；
5. 最后 Formality、物理/功耗闭环。

机器可读裁决在 `m165_independent_hammer_review.json`，所有计数和 A/B 由 `independent_recompute_m165.py` 从 sealed logs、reports 和 mapped netlists独立复算。本评审只新增本目录文件，未修改 production/contracts 或 `docs/359`。
