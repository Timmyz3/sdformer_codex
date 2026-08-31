# M233 H67 FFN dynamic-BN range capture 独立打铁评审

结论：**89/100，P0/P1/P2 = 0/4/3**。M233输入身份、导入manifest、264个NPZ数组、24个FFN BN的shape/population和浮点关系全部通过独立复算。它足以让M234开始**coefficient-side** rsqrt/alpha/offset定点DSE与RTL，但不足以封最终位宽、溢出、valid825或完整dynamic-BN数值等价。

## 独立复核结果

| 项 | 结果 |
|---|---:|
| incoming `manifest.sha256` | `4f4914aa...` verified, 4/4 |
| capture / contract SHA | `10c1c7de...` / `fb47bfad...` |
| profile / config / checkpoint SHA | `04f692c5...` / `8be3f7bb...` / `4f33e086...` |
| checkpoint load audit | receipt reports missing 0 / unexpected 0 |
| policy-changed BN / captured FFN BN | 78 / 24 |
| BN1 / BN2 modules | 12 / 12 |
| samples / records | 10 / 240 |
| NPZ arrays | 264/264, all finite float32 |
| BN1 / BN2 channel pairs per sample | 17,664 / 4,416 |
| `invstd` relation max abs error | `4.7683716e-7` |
| `alpha` / `offset` relation max abs error | `0` / `0` |
| output endpoint relation max abs error | `2.8610229e-6` |
| all JSON summary quantiles | 0 count mismatch; max delta `3.55e-15` |
| sequence coverage | **1** (`zurich_city_09_a`) |

CSV的四档population/channel为192,000 / 48,000 / 12,000 / 3,000，与`[T=10,N=1,C,H,W]`的八种shape完全一致。每个sample都恰好有24条、覆盖同一组24个模块。

## hook语义

`capture_m233...py`对5D `T,N,C,H,W`输入在`T,N,H,W`维度计算mean与`unbiased=False`的variance。SpikingJelly multi-step `BatchNorm2d`先将`T,N`展平再调用PyTorch BatchNorm2d；profile的`no_running`又将`track_running_stats=False`并清空running buffers，因此实际前向确实使用当前batch biased variance。存储数组进一步满足：

- `invstd = rsqrt(variance + eps)`；
- `alpha = gamma * invstd`；
- `offset = beta - alpha * mean`；
- 由输入min/max和正/neg alpha推导的输出min/max与捕获值最大差`2.861e-6`。

因此“current-batch BN affine代数链正确”可接纳；“原始活动量的moment已被独立复算”不可接纳，因为归档没有原始tensor或sum/sumsq。

## 对M234有用的经验域

| 量 | s10 min | p0.1% | p99.9% | s10 max |
|---|---:|---:|---:|---:|
| mean | -3.63764 | -2.40146 | 2.58147 | 3.45256 |
| variance | 0.0190029 | 0.0292935 | 18.6238 | 29.8395 |
| invstd | 0.183064 | 0.231721 | 5.84172 | 7.25231 |
| alpha | 0.178446 | 0.218241 | 5.81768 | 6.84369 |
| offset | -1.84097 | -1.49339 | 1.68126 | 1.93860 |
| BN input | -29.5988 | -21.8345 | 22.2287 | 30.7064 |
| BN output | -12.8829 | -9.82949 | 10.1844 | 13.7295 |

BN1/BN2的variance域差很大：BN1为`0.0190..2.10475`，BN2为`0.17559..29.8395`。M234应比较指数归一化共享rsqrt与分域LUT，而不是盲目使用一个等距大LUT。但上述都是s10 empirical range，需要guard domain，不是饱和安全证明。

## 未闭环的四个P1

1. 10个样本全在同一sequence，不能外推到valid825或跨sequence极值。
2. 没有原始BN输入/sum/sumsq，无法验证moment accumulator的定点次序与溢出。
3. 没有ATLIF膜电位/阈值margin或valid825，系数误差还没有网络可接受边界。
4. 生产manifest没有封sample-content hash、转移model源码与Torch/SpikingJelly/CUDA版本，因此raw capture重跑还不是hermetic的。

## 接纳与禁止口径

允许：将M233定义为冻结checkpoint上的s10 current-batch FFN BN浮点range/affine receipt；引用上述经验范围和浮点关系；启动使用全220,800组通道向量的M234 coefficient engine DSE/RTL。

禁止：宣称已覆盖valid825/最坏range，已封fixed-point位宽或rsqrt算法，已证ATLIF spike等价，已capture all 78 BN，以及任何VCS/DC/PPA/cycle/speedup/headline结论。

M234应先局限为“量化后mean/variance输入 → rsqrt → alpha/offset系数对”，明确guard domain、RNE/饱和、首延迟、II和valid/ready；对220,800组矢量做exact-SHA VCS miter后再跑新思3 ns logic-only DC。
