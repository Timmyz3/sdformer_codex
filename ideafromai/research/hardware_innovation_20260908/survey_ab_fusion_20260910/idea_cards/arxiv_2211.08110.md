# arXiv:2211.08110 · HeatViT

- uid/来源: `MAIN-R081`｜arxiv_2211.08110+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers
- 精读深度: 方法级（仅方法摘录窗口）+依据：多头信息融合的 token selector（插在 block 前）、非信息 token 打包成一个保留信息 token、稠密拼接避免稀疏算子、复用 GEMM、延迟感知多阶段插入训练、8bit+多项式近似非线性

## 可继承 A
自适应 token 剪枝 +「打包保留」而非硬丢；插入位置可学；硬件复用骨干 GEMM——可作 F1 删字时「保留通道表达」与 F5 生存期的算法/实现对照（借入≠X）。

## 强对照 B
静态 token 剪枝；完全丢弃非信息 token；独立 CONV 选择器不复用 GEMM；只剪 head/channel。

## 可差分 X线索
ViT token 打包≠r1 物理源字剪枝 X；差分须双 PED 损失+lifting 源活动。

## 与 F1–F7 / Stage B 关系
F1/F7；Stage B 后可挂。不抢 Stage B。

## 不可搬用边界
ImageNet ViT；打包 token 语义≠源字；仅方法摘录窗口；勿搬 FPGA 3–5×。

## 可复用 idea 点
- 「打包成一信息 token」启发：删字时保留汇总通道而非硬零
- 延迟感知插入位置搜索作 F1 挂点选择模板
- 稠密拼接避免稀疏硬件——对照本地更愿改供数面而非加稀疏控制器
- 多项式近似非线性作普通实现底座，非 X

## 杀门建议
打包保留后并集读仍满或 AEE/ΔAEE 破门，或不胜 HiNM → 停该选择器布局。
