# M399-pre：H67 nested codebook exact early-hit 独立预审

结论：**GO 全 phase prefix 重放；NO-GO 把 distance-zero early termination 单独当成
1.15× 性能轴。** 评分 86/100，P0/P1/P2 = 0/3/5。本预审不产生新 speedup。

机制本身是无损的：q64/q128 继续使用一个 serial16 task server，每拍完成一个
16-center prefix。只有完整 prefix 结束后、running `best_distance==0` 时才停止
后续 prefix。Hamming distance 不可能低于零，而且 nested catalog 的后续 center
ID 更大；M338 的 1,728 个 partition catalog 中也没有任何 q128 内重复 center，
因此 selected lowest ID、distance、PWP/fallback 和 exact reconstruction 都不变。
不能在 prefix 内省半拍，也不能在 distance=1 时提前停。

M397 的 aggregate 能给保守下界，但不能给精确 savings。冻结 exact hit 为：

| prefix | cumulative nonzero exact hits |
|---:|---:|
| q16 | 3,781,137 |
| q32 | 5,380,120 |
| q64 | 7,140,671 |
| q128 | 9,129,057 |

M338 只有 20 个 popcount-one center，且全部已在 q16；q16 之后新增 center 全部
popcount>=2。因此 `E32-E16=1,598,983`、`E64-E32=1,760,551`、
`E128-E64=1,988,386` 都确定是 matcher-eligible 新 exact hit，但 E16 自己可能
包含 pop1 行，不能整项计入 savings。

由此得到：

- q64 aggregate-only 保守下界为 `2*(E32-E16)=3,197,966` matcher cycles；
  理论上界为 `3*E16+2*(E32-E16)+(E64-E32)=16,301,928`。
- q128 保守下界为 `6*(E32-E16)+4*(E64-E32)=16,636,102`；理论上界为
  50,829,770 cycles。

要得到精确结果，必须在全部 17,280 phases 补 q48、q80、q96、q112 的
all-nonzero 与 eligible cumulative exact hit，并输出 eligible
`F16/F32/F48/F64/F80/F96/F112/F128` first-zero-prefix histogram。q64 的精确
saving 是 `S64=3*F16+2*F32+F48`；q128 是
`S128=7*F16+6*F32+5*F48+4*F64+3*F80+2*F96+F112`。

q64 当前需要至少省 8,470,121 matcher cycles 才严格快于冻结 q32，需要省
32,136,034 cycles 才到 1.15 门槛。已有保守下界后，两个缺口分别仍为
5,272,155 和 28,938,068。由于 q64 理论上界仅 16,301,928，单独达到 1.15
已被严格排除；但它仍可能改变 q32/q64 排名。

公平排名还必须把同样机制用于 q32。设 F16 是 eligible q16 首命中，F48 是
q33..q48 首命中，则：

- 对冻结 q32，q64 需满足 `3*F16+F48 >= 5,272,155`；
- 对同样 early-hit 的 q32，q64 需满足 `2*F16+F48 >= 5,272,155`。

后者才是最终架构选择口径。q128 即使取理论最大 saving，也小于它相对冻结
q32 的 61,363,110-cycle 严格反超需求，因此 q128 early-hit 单独同样关闭。

执行合同保持 M397 其余部分完全不变：SHARED96、完整 q config 预载、单
32 B/cycle cmd32 DMA、32-bit bitmap word、II1/L8/D8 48-bit descriptor SRAM、
160 B q128 physical stride、两 32 KiB slot、`8/O` replay、maximal-run command
以及 742,148,386-cycle common baseline。matcher 必须是唯一变化组件；若 RTL
zero-decision 产生 bubble，必须实名加回。

下一条可组合假设是 exact elastic-width PWP，而不是 M399 的一部分。M397 q32
最大项仍是 581,068,416 active-compute cycles；M41 的 H67 INT8 权重允许后续
独立测试 96 B low8 加选择性 64 B-aligned high4 sidecar。q32/O4 worst-case
容量代数为 `64+6144+32*(384+4*64)=26,688 B`，能放入 32 KiB，但真实
non-sign-extension incidence、DMA、周期和 exact miter 均未测，禁止混入本次
early-hit 数字。

`docs/359` 与所有既有证据均未修改。
