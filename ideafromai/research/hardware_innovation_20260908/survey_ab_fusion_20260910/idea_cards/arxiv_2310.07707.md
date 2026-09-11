# arXiv:2310.07707 · MatFormer

- uid/来源: `MAIN-R162`｜arxiv_2310.07707+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: MatFormer: Nested Transformer for Elastic Inference
- 精读深度: 方法级（仅方法摘录窗口）+依据：FFN 隐层套娃嵌套 T1⊂…⊂Tg；每步随机采样粒度训练；Mix’n’Match 层间拼粒度得指数多子网；可扩到 attention head；相对 DynaBERT/OFA/HAT

## 可继承 A
嵌套弹性子网 + 少粒度训练换多部署点——作「宽度/计算弹性」对照；与 shared-Q 消融不同（借入≠X）。

## 强对照 B
为每约束重训；NAS 后重训；联合优化过多子网导致更新不足（DynaBERT）。

## 可差分 X线索
套娃 FFN≠lifting 结构化时间因子 X；弹性推理≠same-port 净服务证明。

## 与 F1–F7 / Stage B 关系
F1 减宽/窄稠密对照轴；条件。不抢 Stage B。

## 不可搬用边界
LLM/ViT 部署弹性；Mix’n’Match 不自动给出硬件周期；仅方法摘录窗口。

## 可复用 idea 点
- 嵌套隐层作「窄稠密 hidden50」谱系的可学习版对照
- 少粒度显式训 + 推断组合 → 降低多宽度训练成本的方法学
- 层间粒度单调 Mix’n’Match 启发式可作消融选点
- 负结果只停套娃减宽布局，不杀窄稠密对照族

## 杀门建议
嵌套子网同预算不胜独立窄稠密或破 AEE → 停该套娃布局。
