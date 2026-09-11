# arXiv:2404.10597 · Seneca synaptic-delay HW-aware training

- uid/来源: `MAIN-R204`｜arxiv_2404.10597+本地excerpt（`p0_excerpt_batches/batch_02.json`）
- 题名: Hardware-aware training of models with synaptic delays for digital event-driven neuromorphic processors
- 精读深度: 方法级（仅方法摘录窗口）+依据：§III 延迟作空间并行突触+剪枝/细化；SCDQ（PRQ/POQ 双缓冲环）；Loihi ring vs Seneca shared queue；硬件感知量化微调；SHD 模型表

## 可继承 A
延迟时空展开→权值优化+剪枝的硬件感知训练；Shared Circular Delay Queue（双 FIFO 换缓冲）作稀疏事件延迟供数底座——时间依赖供数/生存期对照（借入≠X）。

## 强对照 B
无延迟浅网；每神经元 ring buffer（随神经元扩）；纯软件 RISC-V 仿真延迟；无剪枝的满延迟突触。

## 可差分 X线索
可学习突触延迟≠ lifting 结构化因子/半步图 X；本地主张在执行对象与消费者接口，不在延迟原语本身。

## 与 F1–F7 / Stage B 关系
F1/时间轴旁路对照；与 MAIN-R205 成对读。不抢 Stage B。

## 不可搬用边界
SHD/Loihi/Seneca 能效≠ valid825 AEE/本地 VCS；延迟原语≠ T10 合同；仅方法摘录窗口。

## 可复用 idea 点
- 延迟空间化+剪枝作「时间依赖稀疏化」训练模板（换损失）
- SCDQ 双缓冲作多时延消费者共享队列工程对照
- axonal-only vs per-synapse 剪枝作粒度杀门
- 负结果只停某延迟队列布局，不杀时间建模家族

## 杀门建议
延迟队列元数据/占用吃掉并集收益，或精度不胜无延迟结构化 PSN → 停该延迟布局。
