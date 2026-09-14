# 固定 Q11：moment 与 native-tap 两个剪枝导出

已完成两种模型的固定系数选择、静态界以及 135 个真实输入的独立整数 gold；未训练、未用活动/valid 拟合、未运行 GPU/RTL，未扫描剪枝比例。目录 `moment/` 与 `native_tap/` 均可独立用于网络及硬件验证；每个目录的 `factors.npz` / `frozen_parameters.json` 保存模型，`gold_tiles.npz` / `manifest.json` 保存输入与 oracle。a_q40、b_q20、output_scale、q1、θ、BN 全部逐字段保持 Q11 母模型。

| 静态模型 | moment | native_tap |
|---|---:|---:|
| 192 组选择 | m2 全部 192，m1 为 0，无平局 | tap0/1/2 = 45/86/61 |
| 原生全 N8 零 tap 组 / 576 | 0 | 192 |
| Winograd 全 N8 零分量组 / 768 | 192（25%） | 106（13.8021%） |
| BN×scale 加权系数 relative L2 | 0.382092 | 0.489623 |
| 135 输入后 BN 局部输出 RMSE 对 Q11 母函数 | 0.211799 | 0.326990 |
| I24 局部 relative L2 对 Q11 母函数 | 0.147225 | 0.227297 |
| I24 不同值 / 518400 | 457198 | 457444 |
| I24 最大差（整数码） | 29798 | 46988 |
| p 任意 rank/tap 前缀绝对界 | 115211848 | 97615614 |
| M 任意 rank 前缀绝对界 | 212172156 | 186328648 |
| 恢复除 2 前绝对界 | 339506348 | 448428860 |

两臂是不同近似函数，25% 的 U 组与 1/3 原生 tap 不是等工作量；各自的 ordinary-factor、Winograd、expanded-W 执行须对各自 gold，不能跨模型要求 raw 相同或借用母模型 AEE。系数加权误差更小、局部输出更近，都不证明网络 AEE 一定更好。

moment 对每个 `(rank,N8输出组)` 分别求 m1/m2 两个候选，按 `Σlane (BN_gain[o]×output_scale[o])² ||g'−g||²` 选较小者。精确投影域是**母 Q11 对称量化代码域 [−1023,1023]**，不是允许额外 −1024 码的完整 two's-complement 域。取 `h=v⊙g`，v=(1,±1,1)，则约束变为 Σh'=0。枚举 h0 全2047整数；固定 h0 后，h1 只需取受 `[max(−B,−h0−B), min(B,−h0+B)]` 约束的二次式中心 floor/ceil，令 h2=−h0−h1。可行区间内的夹取不破等式或位宽，没有投影后独立截断、没有回退。250 个小域穷举对照与 3072 个真实投影的离散交换最优性证书全部通过，详见 `projection_certificate.json`。

选择结果全为 m2，因此所有水平三tap都满足 **g1=g0+g2**，施加了共同的水平 Nyquist 零响应；这是一项真实统一结构约束，不应描述成无方向偏好的删权。native_tap 在每组清除同一个tap，选择同一加权误差最小者，不动剩余tap。`group_choices.csv` 保留每组的两种 moment 代价及所选原生tap，未用样本活动选组。

每臂 135×3840=518400 个 p 输出均验证 `factor == expanded-W == Winograd`；原 Z 与输入FP32→J保持母gold相同。每个R8条带和全rank两种恢复和均为偶数，除2精确；随后只做原消费者的一次RNE26/I24饱和。两臂 Q2 都在±1023，U在±2046，D保守绝对界17124；p/M/恢复均在signed32，wide在signed64。除标准gold外，额外保存 `winograd_M_int`，便于实际RTL逐M核查。`tile_stats.jsonl` 的局部误差是选择完成后的观测，未反向用于调整。

网络质量由根分别测 diverse10，再决定 valid825。当前 Winograd 固定变换/恢复税与VLOAD循环仍需实际支付；这里没有把25%静态U零直接称为25%周期下降。本文导出没有硬件端点结果，不宣称新 WINS。

后续已实做成熟的自由 U-zero 强控制，见 [unconstrained/README.md](unconstrained/README.md)：全部192组母U2置零，保留未除2的phase3 raw p2，以scale/2重定a_q40；signed32界闭合，135+36输入656640个raw均同独立两相位展开。I24局部relative L2在135/36输入分别为0.115428/0.095584，低于moment的0.147225/0.134929与native_tap的0.227297/0.207814；网络AEE仍由根独立评估。`unconstrained/comparison.csv` 并列全部局部量，两个原有子目录另增36记录gold；原冻结系数和135gold未改。不能以“非整数3tap/奇数”排除这个有效控制，候选moment只能主张保留普通三tap表示和平移不变性，不预设质量或周期优势。

复现：在本目录执行 `/opt/anaconda3/bin/python3.12 export.py`。CPU NumPy导出，读取 `../spatial_winograd_inputs` 和原整数导出的纯函数。结果汇总 `SUMMARY.json`，过程 `export.log`，协议 `PLAN.md`。未修改其它实现或提交 Git。
