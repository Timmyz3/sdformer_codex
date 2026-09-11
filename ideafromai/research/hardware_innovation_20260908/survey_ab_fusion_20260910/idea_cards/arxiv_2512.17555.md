# arXiv:2512.17555 · CFMP / ConvFormer chip

- uid/来源: `MAIN-R348`｜arxiv_2512.17555+本地excerpt（`p0_excerpt_batches/batch_05.json`）
- 题名: A 28nm 0.22μJ/Token Memory-Compute-Intensity-Aware CNN-Transformer Accelerator with Hybrid-Attention-Based Layer-Fusion and Cascaded Pruning for Semantic-Segmentation
- 精读深度: 方法级（仅方法摘录窗口）+依据：HAPU 混合线性/稠密注意力（KTV-first 把存储 O(N²)→O(C²)）；LFS/DR-LFS KV→权替换的层融合复用；CFMP 级联 Fmap 剪枝（W0 扩中间 Z→预训练 mask 剪→W1/DRU 密度恢复）；非重叠 LF+零填边界；ISSCC 摘要体例，图注级架构

## 可继承 A
（1）多数 tile 线性注意力 + 少数 VA 保全局野的混合注意力；（2）层融合调度先耗尽 on-chip KV 再换卷积权，消冗余 EMA；（3）级联「注入冗余→结构化 mask 剪中间→恢复密度」使本不稀疏的 Seg.Head 可剪——剪枝/融合/暴露面压缩强工程底座（借入≠X）。

## 强对照 B
纯 VA 大 TL 外存注意力图；无 KV-权复用的朴素 layer-fusion；Seg.Head 上常规零跳过（稀疏极低无效）；重叠 fusion 窗导致边界覆盖/重复 EMA。

## 可差分 X线索
CFMP/HAPU≠ lifting 源活动×双 PED 删字 X；差分须落到 r1 物理源字与非因果 T10 消费者，而非语义分割 μJ/token。

## 与 F1–F7 / Stage B 关系
F1/F2/F7 强相关（结构化剪枝、tile 级调度、融合复用）；第二队列/Stage B 后可挂对照。不抢 Stage B。

## 不可搬用边界
28nm 0.22μJ/token、TOPS/W 勿搬；Cityscapes ConvFormer ≠ 光流 AEE；ANN 注意力/卷积剪枝假设≠ SNN 源∩W；仅方法摘录窗口（ISSCC 短文+授权页噪声）。

## 可复用 idea 点
- 「先扩中间再 mask 剪」使稠密段可结构化稀疏→对照 F1 在敏感段制造可共同删单元
- FMS 早停解码次一 mask→元数据开销掩盖杀门叙事
- DRU 列偏移→行切片重排作「稀疏存储与稠密消费」边界合同
- 非重叠 FC + 注意力补野 + 边界零填→对照删广播域时的边界误差证书（F4 弱）
- LFS「KV 用尽再换权」作同端口计划复用序（F3/F7 对照）
- 负结果只停「CFMP 挂 r1/Seg 式头」布局，不杀剪枝家族

## 杀门建议
同槽不胜 HiNM/窄稠密或破 AEE，或 mask/DRU 开销吃掉并集收益 → 停该级联剪布局，保留文献对照。
