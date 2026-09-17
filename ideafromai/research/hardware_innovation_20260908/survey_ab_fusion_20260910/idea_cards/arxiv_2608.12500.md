# arXiv:2608.12500 · Lonic INT4 Local Online SNN

- uid/来源: `ARX-161`｜arxiv_2608.12500+本地excerpt（p0_excerpt_batches_gap/gap_08.json）+补arxiv HTML方法（§3–4）
- 题名: Lonic: Algorithm-Hardware Co-Design for Energy-Efficient Fully Local Online SNN Training with INT4 Precision
- 精读深度: 方法级（方法摘录窗口+补HTML §3算法/§4架构）+依据：INT4全局部在线三GEMM量化流；sWCTT替sWS；三型无乘法整型PE；DCOM双优化零门控；TPA时序前缀加速；片上量化减片外权搬；完整评测可能截断

## 可继承 A
全局部在线SNN训练共设计：INT4/三GEMM量化流+sWCTT + 可重构无乘法整型PE(0/1/2) + DCOM尖峰∩权零门控 + TPA跨时步FWD/BWD/WU并行 + INT权片外搬运——相对BPTT/FP训练与稠密PE的训练侧效率对照（借入≠X）。

## 强对照 B
BPTT全局反传存状态；FP32 GEMM/QAT假量化；无零门控稠密PE；逐步串行FWD→BWD→WU；仅推理加速器。

## 可差分 X线索
Lonic训练加速≠lifting X；训练侧旁路，勿搬相对GPU/TPU×能效当净服务%。

## 与 F1–F7 / Stage B 关系
F7相关（训练/更新侧共设计）。训练加速旁路。不抢 Stage B。f_candidates含F7。

## 不可搬用边界
CIFAR/DVS准确率与ASIC能效≠valid825；方法窗+HTML；零门控对齐≠lifting并集合同。

## 可复用 idea 点
- INT4三GEMM量化流作训练低精合同
- DCOM尖峰∩权掩码作双稀疏门控模板
- TPA十RCE时步并行作共同完成旁证
- 无乘法bit-slice PE作整型累加边
- 负结果只停该训练加速挂接

## 杀门建议
映射后训练分母与Stage B冲突或仅搬PPA无差分 → 保持旁路。
