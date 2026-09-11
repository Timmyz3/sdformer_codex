# arXiv:2503.15986 · SpiLiFormer

- uid/来源: `MAIN-R212`｜arxiv_2503.15986+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.2 三级层次；§3.3 FF-LiDiff（Qs 分兴奋/抑制差后 SN→token mask）；§3.4 FB-LiDiff 双前传+可学习 prompt 反馈；α 平衡双 CE 损失

## 可继承 A
侧抑制差分注意力（兴奋−抑制）作 token 级 mask + 反馈通路二次前传——可作注意力路径活动整形与「有损共同完成/跳过冗余阶段」算法先验（借入≠X）。

## 强对照 B
标准 SSA/无抑制的 QKV；单前传无反馈；无 token mask 的全量注意力。

## 可差分 X线索
侧抑制/双前传≠ lifting 结构化 T10/源字删字 X；本地差分在 r1 消费者接口与并集费用，不在 ImageNet 注意力热图。

## 与 F1–F7 / Stage B 关系
F2/F7 弱相关（token mask/反馈跳过 Stage1–2 冗余）；旁路主岛。不抢 Stage B。

## 不可搬用边界
分类精度/mJ 表≠本地 same-port 净服务；双前传训练成本勿写成推理省；仅方法摘录窗口。

## 可复用 idea 点
- Ae−Ai→二进制 token mask 作结构化跳过模板（对照 F1 可共同删）
- 反馈通路「第二遍只跑深阶段」作有损共同完成叙事对照（非搬 FB 环）
- α 双损失作多出口一致性先验
- 负结果只停「侧抑制挂 r1」布局

## 杀门建议
token mask 不能落到物理源字/门服务下降或破 AEE → 停该注意力整形布局，保留文献对照。
