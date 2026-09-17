# arXiv:2607.26648 · Sparsity Ceiling Spiking Networks

- uid/来源: `ARX-025`｜arxiv_2607.26648+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: The Sparsity Ceiling: Where Spiking Networks Can and Cannot Trade Activity for Energy
- 精读深度: 方法级（仅方法摘录窗口）+依据：匹配架构ANN/SNN协议；双侧目标发放正则；Prop.1记忆负载发放地板；输入层地板；注意逃避/KV墙；能量代理分解；完整理论可能截断

## 可继承 A
稀疏天花板分析：双侧ρ⋆探针 + 命题1(记忆负载→发放地板) + 匹配ANN/SNN协议 + 注意逃避但付KV墙——相对「稀疏必自由」叙事的结构边界对照（借入≠X）。

## 强对照 B
单侧L1发稀疏至静默；默认感知/语言同等可稀疏；忽略输入重放地板；无信息论下界。

## 可差分 X线索
Sparsity Ceiling理论≠lifting X；分析旁路，差分不在FashionMNIST/bpc点。

## 与 F1–F7 / Stage B 关系
F2/F7相关（活动稀疏可交易边界、注意vs递归）。分析旁路。不抢 Stage B。f_candidates含F2,F7。

## 不可搬用边界
代理pJ/准确率≠valid825；仅方法窗；发放地板≠lifting删字合同但可作F2杀门先验。

## 可复用 idea 点
- Prop.1 Hb^{-1}(log M/H)作递归发放地板合同
- 双侧ρ⋆探针作可达成稀疏模板
- 输入重放地板作事件输入必要性旁证
- 注意自由稀疏∥KV墙作架构边
- 负结果只停该「稀疏必省」叙事挂接

## 杀门建议
若Stage B假设递归SNN可任意降活动 → 用本天花板作杀门先验。
