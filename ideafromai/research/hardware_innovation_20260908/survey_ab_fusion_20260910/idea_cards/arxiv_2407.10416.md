# arXiv:2407.10416 · SOFA

- uid/来源: `MAIN-R093`｜arxiv_2407.10416+本地excerpt（`p0_excerpt_batches/batch_02.json`）
- 题名: SOFA: A Compute-Memory Optimized Sparsity Accelerator via Cross-Stage Coordinated Tiling
- 精读深度: 方法级（仅方法摘录窗口）+依据：跨阶段协调 tiling；DLZS+SADS 低复杂度预测；SU-FA 解耦 softmax 行依赖；RASS 复用感知调度；挑战=预测开销/整行串行/跨阶段未协同

## 可继承 A
动态稀疏「预测→top-k→正式算」拆成可流水子阶段 + 前阶段排序信息引导后阶段（SU-FA）——作 F2 有损共同完成/跨阶段证书传递的工程对照（借入≠X）。

## 强对照 B
整行就绪才 top-k 的串行动态稀疏；无跨阶段引导的 FlashAttention2；各阶段独立优化无协同 tiling。

## 可差分 X线索
Transformer 注意力 top-k≠ r1 源字剪枝 X；差分须双 PED+lifting 源活动，而非抄 DLZS/SADS。

## 与 F1–F7 / Stage B 关系
F2/F7 对照；第二队列。不抢 Stage B。

## 不可搬用边界
LLM/注意力加速器 GOPS/W≠本地 finite_service；勿搬 9.5×/71.5×；仅方法摘录窗口（微架构后半截断）。

## 可复用 idea 点
- 跨阶段 tiling 打破「整组就绪」串行——对照 F2 半步检查点可否细粒度推进
- 低复杂度预测开销计入并集费用杀门模板
- 排序信息复用减少后阶段 EXP/COMP——证书传递对照
- 负结果只停该预测-正式流水布局

## 杀门建议
预测阶段功耗/延迟≥正式收益，或迁到源字后净服务不降 → 停该跨阶段预测布局。
