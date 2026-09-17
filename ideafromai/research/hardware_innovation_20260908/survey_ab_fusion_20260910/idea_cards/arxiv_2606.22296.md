# arXiv:2606.22296 · SCENIC Edge IoT Command Generation

- uid/来源: `ARX-130`｜arxiv_2606.22296+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: SCENIC: Semantic-Conditioned Edge-Aware Neural Framework for Structured IoT Command Generation
- 精读深度: 方法级（仅方法摘录窗口）+依据：紧凑中文骨干对比；C-SFT三元组辅助；多剪枝(幅度/梯度/WANDA/2:4)；ONNX FP16/INT8；TensorRT编码器剖析；摘录偏评测但方法链完整

## 可继承 A
边缘结构化指令生成：Instruct数据+紧凑Enc-Dec选型 + C-SFT三元组 + 剪枝/导出共评——压缩鲁棒与稠密精度对照（借入≠X）。

## 强对照 B
只看稠密SFT选模；无剪枝鲁棒性的解码器优先；无导出剖析的纯准确率排行。

## 可差分 X线索
SCENIC IoT指令≠lifting X；应用/NLP旁路，勿搬EM@1当净服务%。

## 与 F1–F7 / Stage B 关系
F1弱相关（结构/剪枝选型）。应用旁路。不抢 Stage B。f_candidates含F1。

## 不可搬用边界
EM@1≠valid825；仅摘录窗；ONNX体积降≠运行加速。

## 可复用 idea 点
- 稠密拟合≠高稀疏鲁棒作选模合同
- Enc-Dec作激进剪枝稳定模板
- C-SFT三元组作辅助目标旁证
- ONNX+TensorRT分报作部署边
- 负结果只停该IoT指令挂接

## 杀门建议
任务无关或与Stage B同分母冲突 → 保持旁路。
