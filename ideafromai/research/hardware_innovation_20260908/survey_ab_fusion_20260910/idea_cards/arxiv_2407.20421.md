# arXiv:2407.20421 · SENECA ANN vs SNN optical flow

- uid/来源: `MAIN-R231`｜arxiv_2407.20421+本地excerpt（`p0_excerpt_batches/batch_04_residual.json`）
- 题名: Event-based Optical Flow on Neuromorphic Processor: ANN vs. SNN Comparison based on Activation Sparsification
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.2 SENECA 核（SRAM 状态、8 NPE、RISC-V、event-driven depth-first conv、ac./sp. grouping=4）；§4 FATReLU 可训阈值+膜/阈值稀疏正则+ANN 阈值中位数初始化细调；FireNet/EV-FlowNet 同任务 ANN↔SNN 密度–AEE 表；摘录进入 §5 评测

## 可继承 A
同神经形态核上的激活/尖峰稀疏合同：可训阈值（FATReLU/LIF）+ 稀疏正则 + 同像素 grouping 降膜状态访存 + depth-first 跨核流水——作「活动稀疏→净服务」强对照（借入≠X）。

## 强对照 B
无阈值的稠密 ReLU/GRU FireNet；只压 DRAM 仍做零乘；无 grouping 的逐事件反复加载膜；单核串行整层完成后再交下一层。

## 可差分 X线索
FATReLU/SENECA grouping 本身非标题级 X；差分须落到 lifting 源字删字后同端口访存是否真降，而非复述 MVSEC Dens.(%)。

## 与 F1–F7 / Stage B 关系
F1/F2 强对照（激活删零、层间事件流水/共同推进）；同任务光流旁证。不抢 Stage B。

## 不可搬用边界
MVSEC AEE/Dens 表≠valid825 合同；SENECA 核容量约束分辨率（56/120）勿外推本地；权重不复用是原作机制边界；仅方法摘录窗口。

## 可复用 idea 点
- 同像素 group=4 一次加载膜作 F5/访存分母模板
- depth-first 收首个 ac./sp. 即启动下层 → F2 半步/流水对照
- 层密度不均成瓶颈 → λ_i 加重最密层的调度杀门叙事
- ANN FATReLU ↔ SNN 阈值的同核公平对照纪律
- 负结果只停「照搬 SENECA 映射」布局

## 杀门建议
同负载下稀疏升但服务周期/同端口费用不降 → 停该映射；保留「阈值+grouping」对照轴。
