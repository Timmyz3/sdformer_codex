# 当前 PED V96×24 Kronecker：已完成局部 RTL 筛选

[完整报告](REPORT.md) · [误差](fit_results.json) · [真实周期](rtl_results.json) · [收益表](benefits.csv) · [资源合同](resource_contract.json) · [来源](source_table.csv)

当前 dense 参数、首帧两真实窗口320向量。固定1/2项与同参数普通rank1/2比较；原 U/V RNE位置保持。原直接、两个Kron及其各自同函数展开矩阵，10个正常/背压case合计307,200输出零差。

同8 MAC、共同静态整词跳零与同端口/状态下，Kron1同函数direct为93824→33394拍（−64.4078%），Kron2为105344→61572拍（−41.5515%），含装入和最后输出；旧未跳零控制留在old_control。两项局部V误差77.48%，大于普通rank2的73.65%。AEE未跑；新X=0，不能把已有分解移植称新架构或整网加速。

重跑：`/opt/anaconda3/bin/python3.12 reproduce.py`。仅需预装numpy、Verilator和make；无GPU、训练、EDA或hash。
