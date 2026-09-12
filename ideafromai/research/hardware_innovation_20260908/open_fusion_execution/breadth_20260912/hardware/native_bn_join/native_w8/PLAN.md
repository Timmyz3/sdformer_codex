# 完整 native projection 的 W8 驻留接口

B：普通两块目录复用已经将同一完整后段服务从 457,055,251 降到 427,507,285 槽，但为轮流装三份 H32 权重，多读了 198,733,824 B 权重。96×864 的展开 FP32 权重无法放进 128 KiB coefficient SRAM；这是本次选择 W8 挂点的原因，与此前 PED-U 低位草图不同。

A 与强对照：原 ordinary R24+onepass 的真实 native W 固定逐输出行 dyadic INT8，scale=2^ceil(log2(maxabs/127))，RNE/clip。两轴执行完全相同的新 W8 函数：展开 FP32＋已测最强两块目录复用；实际 packed code＋scale 全权驻留、同两块目录复用。两侧相同 96×8 RF、128 KiB state/coefficient、SR64/SW64/CR256、32 B/5 槽外传，完整 K 升序与原 onepass 树、真实 PED 后继不变。

候选接口与归因：code 与 96 个 scale 共 83,328 B 可以驻留。实际读打包码，sign8 解包、暂存、地址、恢复 scale 和实际输出均计费。每个输出的 gate 为 0/1、所有 K 使用相同 dyadic scale，整数和幅值有确定上界；逐位对照展开 W8，不能只凭该推理宣称等价。量化、整权驻留、普通 blocking 均为借用的共同底座，本实验即使净正也不自动成为标题 X。

只做固定 W8、一个完整 ready 帧，不扫描位宽/块大小，不运行训练/EDA。本目录导出实际展开 W8 和模块名供唯一 ordinary diverse10 新函数评价；不继承旧 825，也不把两张旧 Engine 表相加。
