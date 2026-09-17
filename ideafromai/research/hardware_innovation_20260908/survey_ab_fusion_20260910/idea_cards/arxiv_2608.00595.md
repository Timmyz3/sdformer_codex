# arXiv:2608.00595 · Time-Mux SNN FPGA Accelerator

- uid/来源: `ARX-162`｜arxiv_2608.00595+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: A Time-Multiplexed Spiking Neural Network Accelerator with Pipelined Readout for FPGA Inference
- 精读深度: 方法级（仅方法摘录窗口）+依据：集中FSM十态；1-bit广播总线；每突触本地权缓冲；无乘法右移泄漏LIF；编译期杂质函数装权/模式；流水读出argmax；完整评测可能截断

## 可继承 A
FPGA时分SNN：集中FSM协调层窗 + 1-bit尖峰广播∥本地权查找累加 + 无乘法泄漏LIF + 流水读出——相对全空间扇出互连的布线/时分对照（借入≠X）。

## 强对照 B
784×64全空间权线扇出；AER地址事件；无集中时序屏障的竞态；运行时片外装权。

## 可差分 X线索
时分SNN FPGA≠lifting X；教学/小网旁路，勿搬MNIST周期当净服务%。

## 与 F1–F7 / Stage B 关系
加速器旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
MNIST仿真准确率≠valid825；仅方法窗；时分广播≠lifting多消费者并集合同。

## 可复用 idea 点
- 1-bit广播+本地权作消扇出合同
- 集中FSM十态作层/时间屏障模板
- 右移泄漏LIF作无乘法旁证
- 编译期装权/模式作部署边
- 负结果只停该时分SNN挂接

## 杀门建议
规模不可扩展或与Stage B同分母冲突 → 保持旁路。
