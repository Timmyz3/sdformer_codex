# arXiv:2608.19238 · SLIACF

- uid/来源: `MAIN-R216`｜arxiv_2608.19238+本地excerpt（`p0_excerpt_batches/batch_04.json`）+Methods 窗补读（SLI/ACF 公式）
- 题名: Spiking Local Interaction and Adaptive Complementary Fusion for Spiking Transformer
- 精读深度: 方法级（方法摘录窗口+Methods: SLI/ACF）+依据：SSA QK 共发放离散关系稀疏（非零响应常<40%）；SLI=SN(BN(PWConv(DWConv(X)))) 拓扑邻域通路；ACF: X̃=X+γ_ssa⊙A(X)+γ_sli⊙S(X)，γ 初值0.5 按层通道学习；保留原注意力形式

## 可继承 A
「注意力无关局部通路 + 通道自适应互补融合」结构：在稀疏离散 QK 共激活之外，用 DW/PW 邻域交换补局部空间上下文，再用层/通道 γ 标定 SSA vs SLI——可作 F1「删暴露关系字时保留表达通道」与 F7 互补融合对照底座（借入≠X）。

## 强对照 B
仅 SSA/QKFormer 注意力通路；固定等权/相加融合；改注意力表示本身（α-SSA、双极/三元、SEMM 路由等）而无独立局部支路。

## 可差分 X线索
SLI/ACF 算法增益≠ lifting 结构化 T10/源字物理删字 X；差分须落到 r1 暴露端口与双 PED 损失，或可共同删物理字上界，而非复述 DWConv 支路或 γ 融合。

## 与 F1–F7 / Stage B 关系
F1/F7 结构与互补融合旁路对照。不抢 Stage B；非主岛 RTL 候选。

## 不可搬用边界
Spiking Transformer 精度表（ImageNet/CIFAR/ADE20K）≠本地 same-port 净服务；无硬件 PPA 可搬；方法摘录窗，训练超参/附录配置非本卡范围。

## 可复用 idea 点
- QK 共发放稀疏 →「可共同删关系字」上界叙事对照
- 注意力外独立局部通路 ↔ F1 删字时保留邻域表达
- γ_ssa/γ_sli 通道标定 ↔ 多消费者权重/共享组贡献旋钮（非搬模块）
- 负结果只停「SLI+ACF 挂 ep34」布局，不杀局部互补家族

## 杀门建议
映射后暴露端口/服务不降或破 AEE，或相对 SSA-only 无差分 → 停该挂接，保留文献结构对照。
