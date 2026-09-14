# Spatial R16 → I24：冻结整数合同

本合同替换实际 `r0.conv2.0` 的 **stride1、pad1、C96→N96** 空间卷积：纵向 `3×1, R16` 后横向 `1×3`，接原 BN2、原 identity 与现有 I24 reader。依据是现存 [W8-QDQ 因子](../../open_fusion_execution/major_operator_fusions_20260913/decomposition_owned/results/spatial_r16_w8.npz) 和 [原 adapter](../../open_fusion_execution/major_operator_fusions_20260913/decomposition_owned/adapter.py)，不是 W32 的 AEE。所有精度只选一次，无搜索、训练或 GPU 工作。

| 端点 / 操作 | 冻结表示与边界 |
|---|---|
| 实际输入 | AT-LIF `{0,θ}`，当前 `θ=1`；硬件读 C96×4×4 的原始低10bit source words，分离 T10。物理图像外 source=0。输出原点按 2×2 tile 步进，卷积本身仍 stride1。 |
| Q1 / Z | `q1[r,c,ky]` signed8，形状 `[16,96,3]`，范围 ±127。`Z[t,r,y,x]=Σ q1·gate_bit`，纵向3tap；没有 latent RNE、缩放或饱和。静态全输入界 `[-8562,7427]`，signed15，文件存 int16。每个 2×2 output tile 需 Z `[10,16,2,4]` 横向 halo。 |
| rank 尺度对齐 / Q2 | 从 W8 保存的首级 `s1[r]` 恢复整数 q1；`E[o,r,kx]=W2_QDQ[o,r,kx]·s1[r]·θ`，`sout[o]=max(abs(E[o]))/4095`，`q2=RNE(E/sout[o])`。θ与不同 rank 的尺度一次静态吸入 signed13 Q2 `[96,16,3]`，范围 ±4095；运行时不读取 s1。该系数选择使用既有 19×13 乘法器，Z 符号扩展到19bit。 |
| P | 横向3tap，`p[t,o,py,px]=Σ q2·Z`，signed32，无中间 RNE/饱和。完整输出界 `[-125666413,159593676]`；任意 Q2 归约顺序的部分和绝对值 ≤504162009，故不会靠截断保存。因子近似的浮点表示为 `p·sout[o]+bias[o]`，当前 bias 全0。 |
| 原 BN + identity | `a[o]=RNE(sout[o]·BN_gain[o]·2^40)` signed32；`b[o]=RNE((BN_offset[o]+BN_gain[o]·bias[o])·2^20)` signed32。`J=sat32(RNE(identity_FP32·2^20))`，`wide=p·a+((J+b)<<20)` signed64，`I24=sat24(RNE(wide/2^26))`，I24 的小数位14。负数 tie 按偶数舍入，先舍入再饱和。有限 FP32 identity 超界在 J 处饱和；NaN/Inf 不在此合同内。全有限 identity 的静态 wide 绝对界 2283715053563665，小于 2^52。 |

真实输入来自 [source capture](../../r8_consumer_fusion_20260914/data/first_source_words.npy) 与 [原 identity FP32 capture](../../r8_consumer_fusion_20260914/data/identity_fp32_full.npy)。[gold_tiles.npz](gold_tiles.npz) 收录原8个位置及128..191、4000..4063两套64（tile159重叠，去重为135），完整518400输出；输入只有 `source_words [135,96,4,4]`、`output_origin_yx` 和 `identity_fp32_bits [135,10,96,2,2]`。`z_halo_int/p_int/J_q20/wide_int64/i24` 均为 checker oracle，禁止喂给 DUT。P/I24 的单 tile lane 顺序为 `row=(o//8)*40+(py*2+px)*10+t, lane=o%8`。详细字段见 [manifest.json](manifest.json)。

资源事实：Q1 4608B；Q2 packed13 为7488B（NPZ用 int16 保存9216B）；consumer a/b 共768B。完整 Z tile 使用2560B 的 int16槽位；两个 R8 stripe 各需1280B，不能套用原平坦 R8 的520B结论。输出 psum 仍15360B。`expanded_int32` 仅供独立 gold 复核，不是硬件驻留矩阵；host-only s1/sout/BN 等解释字段不意味着硬件必须驻留，但真实选定的权重、系数及端口配置必须由 RTL 分支计费。

运行 [export.py](export.py) 可重建 [factors.npz](factors.npz)、全部 gold 与 [stats.json](stats.json)。135 tile 的整数两级计算与独立展开卷积、J 与现有 capture、I24 与精确范围内 FP64 RNE 全部一致。相对“保存的 FP32 W8-QDQ 因子在 FP64 中执行、latent 不量化”，新整数 raw 的 relative L2 为 **0.0001315764**、RMSE **8.22537e-5**、最大误差 **0.00121794**；I24 有296634/518400项不同、最大42 LSB，其中 consumer 本身量化的差异7994项、最多1 LSB。W8-QDQ 对 W32 的 raw relative L2 为0.00470295。Q1 的 `q1·s1` 转回 FP32 精确恢复保存因子；FP64 中两者存在保存舍入的微小差异，已包含于上述比较。这些是同一帧选定位置的局部误差，不是 CUDA FP32 累加序一致性或网络 AEE。

[torch_integer.py](torch_integer.py) 提供 `SpatialR16Integer.raw_p(source)`、`consume(p, original_identity)` 及完整 `forward`，支持 `[N,C,H,W]` 和 `[T,B,C,H,W]`；`tile=True` 从真实4×4 halo计算2×2输出。FP64卷积执行的乘积及任意部分和均为精确整数，后段显式 int64 RNE。`reader_value(i24)` 产生精确可表示的 FP32 `I24/16384`，供现有 reader 重读。[CPU 实测](torch_cpu_stats.json) 使用 `/opt/anaconda3/envs/pytorch310_cpu/bin/python -B torch_integer.py`、Torch2.7.1+cpu：135 tile 的 Z/P/J/wide/I24、reader再量化和 T,B 形状全部零差。完整网络质量与实际周期由根代理、RTL分支另测；本页不继承任何 W32/QDQ AEE。

执行约束：FP64可精确表示整数点积，不代表 cuDNN 的变换算法保持整数中间值。根代理 A800 全图诊断发现默认路径有10995072个非整数Z、最大舍入偏差2.728e-12、静态界零越界；只在本 factor oracle 内禁用cuDNN后，135个 live source/identity/Z/P/J/wide/I24 全部零差。因此接口现在用局部 `torch.backends.cudnn.flags(enabled=False)`，不靠事后round掩盖执行误差，也不修改网络其他算子设置。完整网络质量结果由根代理质量报告给出。
