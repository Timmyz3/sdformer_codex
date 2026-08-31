# M161 H67 动态 BN + rank-3 ATLIF 融合 DSE r1

## 动机

冻结推理使用 `no_running/current-batch` BN，不能做 M160 r1 的静态 checkpoint fold。M161 改成正确的动态方案：fc1 输出流过时，同时累计每个 hidden channel 的 `sum/sumsq`，并计算 ATLIF rank-3 右投影；moment barrier 后在 rank state 上完成动态归一化修正，再做左投影和 threshold。

这让 RQTB/rank-3 与完整 FFN 可以共存，不需要复制 Linear/ATLIF MAC pool。

## 精确代数

对每个 hidden channel `j` 和空间位置 `p`：

`v[r,p,j] = Σt R[r,t]·x[t,p,j]`

动态 BN 得到 `alpha[j]` 与 `offset[j]` 后：

`v_bn[r,p,j] = alpha[j]·v[r,p,j] + offset[j]·Σt R[r,t]`

再计算 `h[t,p,j]=Σr L[t,r]·v_bn[r,p,j]+bias[t]`。

100 组 float64 随机 miter 最大误差 `1.42e-14`。这只证明实数代数，不证明 rank-3 训练精度和定点舍入顺序。

## 吞吐匹配发现

现有 M30/M31 rank-3 资源为 96 个 signed-INT8 product slots、16 lanes、T=10、R=3。右投影每拍处理 `2 time × 16 lanes = 32` 个 fc1 值。

因此配置 32 条 square+sum lanes，恰好同拍消费这 32 个值：

| moment lanes | ideal issues/frame | 相对 rank-3 右投影 |
|---:|---:|---:|
| 8 | 43,776,000 | 4.0x |
| 16 | 21,888,000 | 2.0x |
| 32 | 10,944,000 | 1.0x |
| 48 | 7,296,000 | 0.667x |
| 96 | 3,648,000 | 0.333x |

32 lanes 是首个不增加理想 issue count 的平衡点，但 square-lane 面积、STA、输入 ready、moment barrier 都未证明。rank-3 完整两段投影为 21,888,000 ideal issues，对比 dense sn2 的 36,480,000 是 `1.667x` 局部算术候选，不是系统倍速。

## 本地中间态 bit movement

显式基线按动态 BN 的五次移动计数：raw accumulator 写、统计读、归一化读、Q8 normalized 写、ATLIF 读。候选只在 barrier 前写一次 rank state，barrier 后读一次。

| 口径 | bits/frame | reduction |
|---|---:|---:|
| 动态 BN 基线 | 21,196,800,000 | 1.00x |
| Q24 rank-3 state | 5,042,995,200 | 4.203x |
| Q8 rank-3 state（需训练） | 1,680,998,400 | 12.610x |

最大单块 barrier storage 从 1,032,192,000 bits 降至 Q24 的 530,841,600 bits（`1.944x`），或需训练 Q8 的 176,947,200 bits（`5.833x`）。这些是本地中间 buffer 的位数/位移动，不是 SRAM/DRAM 事务、周期、能耗或系统倍率。

## 仍然昂贵的部分

- BN1 的全模块 moment barrier 仍存在；
- 需要 32 条平方/累加 lane、moment state 和 reciprocal-sqrt；
- Q24 rank state 需要实际 SRAM 地址/端口/回放合同；
- BN2 后没有 rank contraction，仍需缓存并回放 fc2 output，再 normalize + residual commit；
- Q8 提前 requant 改变运算顺序，必须由 PAFT 和 valid825 接纳。

最优算法反哺仍是训练/校准一个 frozen-running-BN 部署 checkpoint；成功后两个动态 barrier 都可删除，BN 才能静态折叠。若精度不接受，再实现本动态融合路径。

当前等级：`PASS_DYNAMIC_BN_RANK3_ALGEBRA_AND_BIT_MOVEMENT_DSE`。未接纳 rank-3 accuracy、fixed-point、RTL/VCS、cycle/system speedup 或 PPA。
