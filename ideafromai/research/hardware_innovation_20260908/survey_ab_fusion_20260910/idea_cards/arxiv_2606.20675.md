# arXiv:2606.20675 · VQ4SNN Vector Quantization FPGA SNN

- uid/来源: `ARX-168`｜arxiv_2606.20675+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: VQ4SNN: Vector Quantization for Memory-Efficient FPGA Spiking Neural Networks
- 精读深度: 方法级（仅方法摘录窗口+补arxiv HTML §III）+依据：权VQ指针+共享码本两级存储；交错层执行免多端口；分析选型(d,k)；相对Spiker+/ModNef BRAM；原摘录偏结论已补方法

## 可继承 A
FPGA SNN权向量量化：指针∥共享码本两级存储 + 交错执行消多端口 + 分析选(d,k)——片上BRAM压缩与稠密权表对照（借入≠X）。

## 强对照 B
稠密BRAM权表；重多端口码本；仅标量量化/剪枝；无硬件感知的纯算法VQ。

## 可差分 X线索
VQ4SNN码本≠lifting X；存储压缩旁路，勿搬BRAM%当净服务%。

## 与 F1–F7 / Stage B 关系
F5/F1弱相关（权驻留组织、有损压缩）。存储旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
MNIST/SHD准确率≠valid825；高尖峰密度时延↑；补HTML≠原gap窗全文。

## 可复用 idea 点
- 指针+共享码本作两级权存合同
- 交错层执行消多端口作时分模板
- 分析(d,k)选型作压缩旁证
- 码本复制换延迟作资源边
- 负结果只停该VQ挂接

## 杀门建议
高活动时延不可接受或精度掉点挤占 Stage B → 保持旁路。
