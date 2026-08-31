# M161 r1 位宽与公平基线纠正 overlay

## P0：raw fc1 不能直接继承 M31 吞吐

冻结 checkpoint 的 fc1 INT8 dot-product bound 需要 signed 14–16 bit accumulator；现有 M31 的输入 bank、multiplier A/B 和 product pool 均为 signed INT8。M161 r1 把 raw accumulator 直接按 M31 每拍 32 values 处理是不成立的。

因此：

- `10.944M right issues / 21.888M full rank3 / 32 moment lanes / 1.667x dense-sn2 arithmetic` 只属于“先把 raw fc1 requant 到 Q8”的训练路径；
- Q8 在 current-batch moments 之前改变数值与 BN 统计，必须由 PAFT + valid825 接纳；
- 保持 raw 14–16b 时，要么单独综合 16x8 widened pool，要么证明 signed-limb 分解；不能继承 M31 面积/周期；
- 若仅用“两次 8x8 product”作算术下界，right stage 至少 21.888M issues，full rank3 至少 32.832M，对 dense sn2 的上限只剩 1.111x，且 correction/route/barrier 尚未计入。此时 16 moment lanes 才与 issue count 匹配。

## P1：公平 movement 基线

M161 r1 的 `4.203x/12.610x` 对比的是五次移动的朴素动态 BN。既有硬件路线已经允许 fc1 输出时 online moments，并把 normalize 与 ATLIF read 融合，因此公平 BN1 基线应是两次 raw movement：write + fused read。

| 比较 | BN1 only | 连同共同剩余 BN2 |
|---|---:|---:|
| Q24 rank state | 2.061x | 1.682x |
| Q8 rank state（需训练） | 6.184x | 2.944x |

共同 BN2 按 online moment 后 raw write/read计 2,801,664,000 bits；它在两边都不能删除。所有数字仍只是本地 intermediate bit movement，不是 SRAM/DRAM transactions、cycles、energy 或 system speedup。

## 保留的创新

动态 BN 与 rank-space correction 的实数代数仍正确，geometry、moment width 和 barrier storage census 仍可用。真正值得继续的两条线是：

1. 算法训练接受 Q8 early-requant，从而复用 M31 并争取 2.944x 的完整 FFN BN bit-movement 候选；
2. 算法转成 frozen-running BN，直接删除两个动态 moment barrier，通常更有利。

当前所有 rank3 accuracy、fixed-point、RTL/VCS、cycle/system speedup 与 PPA 仍为 false。
