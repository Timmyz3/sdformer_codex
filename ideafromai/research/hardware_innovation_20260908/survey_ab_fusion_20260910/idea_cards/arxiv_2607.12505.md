# arXiv:2607.12505 · MD-SpMM Realizable N:M Search-Kernel

- uid/来源: `ARX-126`｜arxiv_2607.12505+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: Realizable N:M Sparse Transformer Inference via Search-Kernel Co-Design
- 精读深度: 方法级（仅方法摘录窗口）+依据：MD-SpMM=Weight Packing+Micro-Dense MMA+Adaptive SplitK；LUT延迟模型+NNLS；Algo1三阶段演化搜索；完整部署可能截断

## 可继承 A
可实现N:M推理共设计：Micro-Dense把N:M落成MMA规整瓦片 + 打包解码 + 规则驱动SplitK + 延迟LUT层间搜索——相对FLOPs代理与不规则SpMM的可部署对照（借入≠X）。

## 强对照 B
仅优化稀疏比/FLOPs；不规则索引驱动SpMM；统一层N:M；无延迟约束的准确率搜索。

## 可差分 X线索
MD-SpMM搜索≠lifting X；GPU内核旁路，勿搬端到端加速比当净服务%。

## 与 F1–F7 / Stage B 关系
F1相关（层间结构化稀疏配置）。内核旁路。不抢 Stage B。f_candidates含F1。

## 不可搬用边界
ViT延迟预测≠valid825；仅方法窗；Tensor Core MMA≠same-port合同。

## 可复用 idea 点
- Micro-Dense N:M→MMA瓦片作规整执行合同
- B_pack联合值/位作解码模板
- 规则SplitK作占用/归约权衡旁证
- LUT+NNLS延迟约束搜索作配置边
- 负结果只停该可实现N:M挂接

## 杀门建议
延迟LUT失准或搜索挤占 Stage B → 保持旁路。
