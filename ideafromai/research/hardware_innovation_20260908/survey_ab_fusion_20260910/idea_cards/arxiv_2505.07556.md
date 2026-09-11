# arXiv:2505.07556 · SSER

- uid/来源: `MAIN-R238`｜arxiv_2505.07556+本地excerpt（`p0_excerpt_batches/batch_05.json`）
- 题名: Self-Supervised Event Representations: Towards Accurate, Real-Time Perception on SoC FPGAs
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.2–3.3 GRU/MGU 选型与 per-pixel 自监督事件编码；§3.4 QAT 混合精度、全并行 matmul/elementwise、LUT 激活、BRAM 每像素隐状态、16 CC/event 流水与同像素间隔约束；摘录止于 §4.1 实验开头

## 可继承 A
异步 per-pixel 递归事件表示（读 Hi−1→fe(ui,hprev)→写回）+ 自监督重建训练（掩码 MSE，偏重时间）+ QAT 硬件友好 GRU/MGU 流水——前端表示/状态驻留与「同地址未完成则不可再入」合同对照（借入≠X）。

## 强对照 B
固定窗直方图/体素再 CNN；LSTM（双隐状态加倍存储）；无 QAT 的纯浮点 RNN 表示；忽略同像素流水依赖的理想吞吐声明。

## 可差分 X线索
SSER/MGU 表示≠ lifting 结构化 T10/源字删字 X；本地主张在 r1 执行对象与消费者接口，不在 SoC FPGA 事件编码本身。

## 与 F1–F7 / Stage B 关系
F2 弱相关（同像素 16CC 完成屏障作「地址级共同完成」工程对照）；旁路主岛（前端表示）。不抢 Stage B。

## 不可搬用边界
Gen1/1Mpx 检测 ≠ valid825 AEE；160ns/event@100MHz 勿写成 same-port 净服务%；仅方法摘录窗口（实验/消融后半可能截断）。

## 可复用 idea 点
- 同像素「前一事件隐状态写回前不可再调度」作有限状态互斥合同对照（近 F2/背压）
- BRAM W×H×dout 驻留↔有限 RF/每址状态存活叙事（F5 弱对照）
- MGU 合并门仍同 CC、乘数维降到 2·dout → 算术并行≠状态费用下降杀门
- 掩码重建偏时间（α=1,β=0.1）作事件合同损失权重模板
- 负结果只停「SSER 前端替换本地」布局

## 杀门建议
替换前端后 AEE/延迟无增益，或同址屏障使吞吐被钉死且无余量 → 停该表示挂载，保留文献对照。
