# 第二轮深读：软件候选与硬件实现线（2026-07-12）

## 1. 结论先行

本轮新增一个可立即 full30 的结构候选：**H75 AX17 Match-Code**。它来自 Flow1D 的正交搜索
分解，但删除原论文动态 softmax/value carrier，改为固定17路跨时位移描述子和静态 codebook。
它满足 all12、one-sided binary ATLIF105、无 SC/TX 混合、无 native carrier。

EMatch 支持“事件光流应显式作为像素对应匹配”的总方向，但其 TRN/SCA、多任务训练和 dense
matching 不直接移植。其最有价值的后续迁移是训练期 GT-displacement descriptor supervision，
部署图不变；先等待 H73-H75 plain full30，再决定是否只做一次受监督版本。

E-STMFlow 的 Mamba/S4 路线是完整 encoder 重构，状态、浮点投影和 selective scan 均会推翻当前
硬件数据流，因此不进入本轮 TTX attention 替换队列。

硬件侧新增 RAWAtten 的 stage-aware window reuse/w-core 映射作为布局依据；不采用其近似
LR-Softmax 替换 Shiftmax，避免未经验证地改变数值。

## 2. 软件线深读

### 2.1 Flow1D（ICCV 2021）到 H75 AX17

来源：

- 论文：<https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_High-Resolution_Optical_Flow_From_1D_Attention_and_Correlation_ICCV_2021_paper.pdf>
- 官方代码：`repos_round2/flow1d`
- attention：`flow1d/flow1d/attention.py:61-65`
- correlation：`flow1d/flow1d/correlation.py:64-92`
- 双轴拼接：`flow1d/flow1d/flow1d.py:111-142`

原论文不是简单的十字邻域。对目标特征先在一条轴上执行

```text
A = softmax(Q K^T / sqrt(D))
F2_hat = A V
```

再沿正交轴做1D correlation；交换横纵方向后得到两组3D cost volume并拼接。官方代码确认
`Attention1D`包含动态`attention @ value`，`Correlation1D`分别生成`[B,H,W,W]`和
`[B,W,H,H]`。所以照搬原式会引入softmax、动态V carrier和大矩阵，不符合当前边界。

H75 的迁移式为：

```text
O_ax = {(0,dx)|dx=-4..4} union {(dy,0)|dy=-4..4}  # 17 offsets
s_o = (popcount(Q_t,p & K_1-t,p+o)
       + alpha*popcount(~Q_t,p & ~K_1-t,p+o)) / D
z = Shiftmax_17(s)
Y[h,t,p,:] = z @ W_ax[h,17,D]
```

它保留“横纵之和代替二维搜索乘积”和位移通道，删除动态 propagation。相对候选：

- H66d 是同时间 local5 且动态聚合K；H75是跨时间半径4、17路静态codebook。
- H73 DE9 是3x3双证据18维；H75是轴向大位移单score17维。
- H74 MC49 保留二维稀疏offset；H75没有对角联合位移，但成本约为其三分之一。

执行协议已冻结：TTX epoch2 warm-start、full30、batch8/workers8/AMP/cupy、warmup720、milestone
20/25、valid825 epochs 0/4/9/14/19/24/28/29。部署codebook量化到signed 2^-7网格。

### 2.2 EMatch（ICCV 2025）

来源：<https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_EMatch_A_Unified_Framework_for_Event-based_Optical_Flow_and_Stereo_ICCV_2025_paper.pdf>

论文把flow和stereo统一为事件特征对应：flow使用2D warp后最小化特征L2距离，stereo使用1D
warp后最小化L1距离。模型用TRN聚合事件时间组，再用SCA填充稀疏事件缺失的空间上下文，最后
做pixel-wise similarity matching。它支撑Match-Code“输出应保留位移索引”的动机，但不能直接
声称H73-H75复现EMatch，因为当前没有多任务stereo、ConvGRU TRN或SCA。

可控的下一步不是换主干，而是训练期 descriptor supervision：把GT flow按每个stage/window
下采样并量化到候选offset，令Match-Code gate对目标offset做交叉熵或有界邻域分布损失。部署
不新增算子。只有H73-H75 full30显示“训练下降但位移descriptor不聚焦”时才启动一次；否则不做
无依据组合。

### 2.3 E-STMFlow（CVPR EventVision 2025）

来源：

- 论文：<https://openaccess.thecvf.com/content/CVPR2025W/EventVision/papers/Humais_Spatio-Temporal_State_Space_Model_For_Efficient_Event-Based_Optical_Flow_CVPRW_2025_paper.pdf>
- 官方代码：`repos_round2/E-STMFlow/model/emamba.py`

论文把4D事件体分patch后用S4/S4D/S5/Mamba序列变换，再reproject为时空特征，并采用convex
upsampling。官方代码确认依赖`mamba_ssm.Mamba/Mamba2`，多层MambaLayer与S4 kernel；这不是
attention局部修改。虽然其DSEC ablation报告Mamba块数量/速度权衡，但迁移会新增连续状态、
浮点A/B/C/Delta投影和scan控制，无法只重做attention硬件。因此裁决为相关工作/未来完整架构线，
不进入当前主线竞争。

## 3. 硬件线深读

### 3.1 RAWAtten（DATE 2023）

来源：<https://past.date-conference.com/proceedings-archive/2023/DATA/333.pdf>

RAWAtten针对Swin类window attention。论文的关键硬件观察是：浅stage窗口多、共享权重带来weight
reuse；深stage窗口少且大，input reuse更重要。它用可组合w-core适配不同stage参数，并在每个
core内配置线性层NMC、矩阵MAC和softmax单元。

对本项目可直接采用的仅是bit-exact映射策略：

- 浅stage按window并行，codebook按head常驻；深stage组合多个core处理更多heads/descriptor。
- 固定offset K halo按row/column/预注册MC49顺序流式读取，减少重复地址和buffer翻转。
- H73/H75/MC49使用同一offset engine与可配置descriptor计数9/17/49。

RAWAtten的LR-Softmax通过估计改写指数/除法，是近似数值模块；当前Shiftmax已有独立部署量化，
不能把LR-Softmax面积数字直接套用，也不能未经valid825替换。论文报告的GPU speedup只用于说明
window-aware专用架构价值，不作为本芯片PPA结果。

### 3.2 E-STMFlow 对状态硬件的反面约束

SSM线性复杂度不等于本项目更低能耗。其state update需要持续读取状态和连续参数，而TTX/Match-
Code只在固定T=2窗口内做binary matching。除非未来重新定义完整encoder与训练，不应为了引用
Mamba而增加另一套状态阵列。当前硬件线继续保持fixed-offset popcount、Shiftmax、静态codebook
和bit-exact temporal packing。

## 4. 当前实验队列与停止条件

H75 已加入 H73/H74 同一 watcher，顺序为 DE9 -> MC49 -> AX17。三项都必须完成full30后才能
比较，不用short失败淘汰。主线门槛：NB0 AEE+5%内、spikes至少-20%；满足后比较H60部署
AEE1.5016/AAE9.8431，并加入完整attention operation/SRAM/NoC成本。

不启动的组合：H73+H75混合、不同stage不同offset、Match-Code与SC/TX并行、SSM+TTX混合。
这些都会破坏统一硬件叙事或无法形成单变量消融。
