# 唯一普通结构源强对照

B：当前 lifting 对照已有完整 dense-source 常量编译，但缺同一普通学生的结构稀疏源，尚不能排除收益主要来自普通时间矩阵删系数。强对照固定为当前 ordinary `original_ordered24 + onepass` 父臂；保留真实 I24、As 原定点非零系数及 exponent15、source τ/θ、consumer gain/阈值、Conv2/F、PED 与 BN，不恢复训练。

A：旧 dependency 的 contiguous3/3/4 块支撑。旧实验修改 r1.sn2 并重新拟合系数与 bias；本次仅迁移通用 T10 支撑 `0..2 / 3..5 / 6..9`，共34项，套到当前 r1.sn1 的原 As_q16。旧拟合矩阵、bias 和 AEE 均不继承。这是一次新干预，不是旧学生重放，也不是按当前精度挑结构。

X：本臂没有独占新意；它用于检验 lifting 结构相对普通结构化 PSN 的必要性。两方享同一 da4ml 整矩阵分解/CSE、门阈值折叠和96×8×48 RF、两级回写、同 SRAM/DMA 的编译与供数权限。

先交唯一矩阵、原 I24 的新门及最小 helper adapter，交给算法代理运行 ordinary diverse10；不自动825、不扫支撑、不训练。随后编译同一 CSE 可执行程序并计源阶段服务、端口字节和状态。新源改变后级输入，不能借旧 sn2/消费者 golden 或费用当新的完整链结果。去留按新同帧 AEE 与实测费用比较，不沿用已取消的 +0.005 门，也不宣称强接收。
