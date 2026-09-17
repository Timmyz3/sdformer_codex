# arXiv:2605.00319 · PVT-Resilient Subthreshold SRAM-CIM SNN

- uid/来源: `ARX-173`｜arxiv_2605.00319+本地excerpt（`p0_excerpt_batches_gap/gap_05.json`）
- 题名: A PVT-Resilient Subthreshold SRAM-Based In-Memory Computing Accelerator with In-Situ Regulation for Spiking Neural Networks
- 精读深度: 方法级（仅方法摘录窗口）+依据：§II in-situ监测+分布式稳压；CIM/访问模式切换；电流阈ITH神经元；stride-tick批；variation-aware训练；完整芯片测可能截断

## 可继承 A
亚阈SRAM-CIM：片上监测电流闭环调VR抑PVT + 可编程ITH尖峰神经元免多比特ADC——PVT韧性存内SNN与校准/副本列对照（借入≠X）。

## 强对照 B
IDAC降RWL；事后校准/副本列；多比特ADC读出ANN-CIM；无variation-aware微调。

## 可差分 X线索
亚阈CIM SNN≠lifting X；模拟电路旁路，勿搬关键词检出准确率当净服务%。

## 与 F1–F7 / Stage B 关系
模拟CIM旁路；与F族弱挂。不抢 Stage B。f_candidates空。

## 不可搬用边界
KWS准确率/nJ≠valid825；仅摘录窗；模拟稳压≠数字same-port合同。

## 可复用 idea 点
- 监测单元+分布式EA闭环VR作PVT合同
- ITH电流阈代电压阈作尖峰检测模板
- 访问/CIM双模供电切换作噪声裕度旁证
- variation-aware微调恢复精度 | 负结果只停该CIM前端替换

## 杀门建议
稳压开销过大或精度不可恢复挤占 Stage B → 保持旁路。
