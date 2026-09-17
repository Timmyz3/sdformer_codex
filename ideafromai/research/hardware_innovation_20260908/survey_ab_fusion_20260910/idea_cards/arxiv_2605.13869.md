# arXiv:2605.13869 · NESTformer Elastic Spiking Transformers

- uid/来源: `ARX-043`｜arxiv_2605.13869+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: Elastic Spiking Transformers for Efficient Gesture Understanding
- 精读深度: 方法级（仅方法摘录窗口）+依据：§III弹性MLP/注意力/特征；Algo1行式注意力两级LIF；粒度g嵌套切片；GPU并行训→部署行式；尖峰能量分析；完整评测可能截断

## 可继承 A
弹性尖峰Transformer：Matryoshka嵌套宽度/头数/通道 + 行式QK→LIF→V两级映射免BMM——运行时无重训缩放与神经形态原生注意力对照（借入≠X）。

## 强对照 B
固定尺寸Spikformer/QKFormer；BMM/softmax注意力卸载CPU；仅MLP弹性的MatFormer；尺寸降但尖峰不降的特征提取瓶颈。

## 可差分 X线索
NESTformer弹性≠lifting X；模型侧旁路，勿搬EHWGesture/DVS准确率或µJ当净服务%。

## 与 F1–F7 / Stage B 关系
F7相关（行式注意力作打包友好算子）。模型旁路。不抢 Stage B。f_candidates含F7。

## 不可搬用边界
手势准确率/Loihi pJ模型≠valid825；仅摘录窗；GPU慢化行式≠芯片吞吐合同。

## 可复用 idea 点
- 行式两级LIF(Q·Kᵀ→LIF→·V)作神经形态注意力合同
- 粒度g嵌套切片作无重训宽度/头/通道模板
- 训用并行GEMM部署转行式作软硬桥
- 特征提取尖峰占比作能量缩放旁证
- 负结果只停该弹性挂接

## 杀门建议
弹性无线性降尖峰或映射开销挤占 Stage B → 保持旁路。
