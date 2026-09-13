实际 GPU 前向入口已准备：`adapter.install_conv(module, filename)` 返回 `(factor, restore)`。目标为 `sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0`，原生 3×3/stride1/pad1/96→96，支持 T,B,C,H,W 或 N,C,H,W。外层真实 BN、PSN、残差仍由已安装的 matched dense ParentNetwork 执行。源为 θg，当前 θ 静态；导出权重按实际 x=θg 拟合，不能再重复乘 θ。

```python
from adapter import TARGET, install_conv
factor, restore = install_conv(net.modules[TARGET], path)
try:
    # root 现有 evaluate_axis(...), same diverse10 / NB0 protocol
    ...
finally:
    restore()
```

已完成相应真实十帧数值前向的文件（AEE 见 README/root 汇总）：

| 文件（本目录相对路径） | 用途 |
|---|---|
| count_results/signed_selected_pair_w32.npz | 新的有符号二输入 count 字典＋两项原值 |
| count_results/unsigned_selected_pair_w32.npz | 同字典/同存储，只有无符号 count 的强 A |
| count_results/unsigned_h8_shared_pair_w32.npz | 一个 pair-ID 在物理 H8 输出组共享 |
| count_results/plain34_h8_shared_w32.npz | 同 H8 共同 omitted-index 的普通 3:4 |
| results/flat_nm_r32_w32.npz | 相近局部误差的普通平铺低秩＋2:4 残差 |
| results/structure_3of4.npz | 普通结构零强控制，误差更大但算术更少 |
| results/spatial_nm_r8_w32.npz | 第一轮空间分解候选，通过 NB0，服务仍须闭合 |
| results/flat_nm_r8_w32.npz | 空间 R8 的普通平铺同秩控制 |
| results/flat_svd_r8_w32.npz | 无残差普通矩阵分解，AEE 1.3479650409 |
| results/activation_svd_r8_w32.npz | 无残差激活加权分解，AEE 1.4030194175 |
| results/spatial_r16_w32.npz | 无残差 3×1→1×3，AEE 1.2691297866 |
| results/tucker_r8_w32.npz | 无残差 Tucker-2，AEE 1.4170152223 |

count 文件里的 `weight` 是实际分解所定义的卷积核。GPU bridge 对它调用原生稠密卷积，服务于同协议网络 AEE；没有宣称 GPU 在运行整数 count 稀疏内核。`count_basis.execute_pair` 在 CPU 对全部真实采样点以实际 {-2,-1,0,1,2} 两位面执行，和该重构卷积的 Float64 结果差小于 1e-10。原捕获的 CUDA TF32/FP32 归约与这个 Float64 数值定义仍有约1e-4相对差异，故网络评价不等于位精确硬件部署通过。

现有 install_conv 的 `trainable=True` 可用于对稠密重构核的普通恢复。它不是维持 pair/count 结构的训练；结构恢复必须单独约束 a、两个保留权重及离散 pair/sign，不能把该开关当作已实现的字典 QAT。

所有 AEE 验收只使用用户当前 NB0 同协议门：825 1.44535253468097、diverse10 1.45460286107。不继承旧文本的 ordinary +0.005。
