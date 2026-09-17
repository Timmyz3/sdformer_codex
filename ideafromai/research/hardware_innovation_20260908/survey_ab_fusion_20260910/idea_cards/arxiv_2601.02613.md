# arXiv:2601.02613 · SAOCDS

- uid/来源: `ARX-047`｜arxiv_2601.02613+本地excerpt（`p0_excerpt_batches_gap/gap_05.json`）
- 题名: Sparsity-Aware Streaming SNN Accelerator with Output-Channel Dataflow for Automatic Modulation Classification
- 精读深度: 方法级（仅方法摘录窗口）+依据：§III SAOCDS；FC权掩码AND；卷积GOAP/EM；输出通道顺序流；Algo1时空稀疏无嵌套控制；完整AMC表可能截断

## 可继承 A
流式SNN：预压缩非零权遍历 + 输入尖峰门控累加 + 按输出通道固定顺序流——时空稀疏与层间直连吞吐合同对照（借入≠X）。

## 强对照 B
输入优先流式无视权稀疏；脉动阵列GOAP复制控制；仅时间稀疏滑窗；层间写回全局缓冲。

## 可差分 X线索
SAOCDS/AMC≠lifting X；FPGA流式旁路，差分不在调制分类准确率%。

## 与 F1–F7 / Stage B 关系
F1/F2相关（非零权/源活动选择、输出通道有序共同完成）。可挂第二队列线索。不抢 Stage B。f_candidates含F1,F2。

## 不可搬用边界
AMC准确率/吞吐≠valid825；仅摘录窗；输出通道顺序≠lifting半步RNE检查点。

## 可复用 idea 点
- 权掩码AND得fetch mask作FC空间稀疏合同
- GOAP按非零权+EM门控累加作卷积模板
- 输出通道顺序流免层间控制作直连旁证
- 时空稀疏无嵌套条件作负载均衡语言 | 负结果只停该流式数据流替换

## 杀门建议
稀疏度低时无增益或通道序约束冲突挤占 Stage B → 保持旁路/降为F1/F2线索。
