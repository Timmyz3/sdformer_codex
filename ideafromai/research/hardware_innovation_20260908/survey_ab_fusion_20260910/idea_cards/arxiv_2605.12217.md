# arXiv:2605.12217 · X-HEEP+ReckOn Heterogeneous SoC

- uid/来源: `ARX-172`｜arxiv_2605.12217+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: Heterogeneous SoC Integrating an Open-Source Recurrent SNN Accelerator for Neuromorphic Edge Computing on FPGA
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.1–3.3 X-HEEP/ARM×ReckOn；AER解码FSM；32b事件字(type/addr/tick)；SPI配置+BRAM/批卸载；e-prop TARGET_VALID；完整PPA可能截断

## 可继承 A
异构SoC挂开源RSNN：X-HEEP/ARM主机 + AER解码FSM供刺 + ReckOn共处理器(e-prop在线)——边缘神经形态集成与主机/加速器分界对照（借入≠X）。

## 强对照 B
单片纯加速器无主机；无AER时间戳的同步帧批；无在线学习的纯推理核；大BRAM一次性装全数据集。

## 可差分 X线索
X-HEEP+ReckOn集成≠lifting X；平台/SoC旁路，差分不在Braille准确率/资源%。

## 与 F1–F7 / Stage B 关系
F5/F7弱相关（事件字打包、批卸载供数）。平台旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
边缘准确率≠valid825；仅摘录窗；AER 4相握手≠lifting same-port合同。

## 可复用 idea 点
- 32b事件字(type∥addr∥tick)作供刺打包合同
- AER解码FSM作SAMPLE/TICK/epoch准确率旁证
- SPI参数库+AXI批卸载作主机/加速器分界模板
- TARGET_VALID触e-prop作在线学习边
- 负结果只停该SoC挂接

## 杀门建议
集成税或BRAM批卸载瓶颈挤占 Stage B → 保持旁路。
