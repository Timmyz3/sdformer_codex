已实现四位置低位打包和付费高位修复。真实64块冷/无背压时，**相对静态范围证明的三位置10bit强控制，1136507→1088854拍，省47653拍（4.1929%）**。本项只用既有八条32bit producer加法链；没有借用消费者64bit链。数值函数、完整Q1/Q2、FP32 identity→J20→I24均保留。

|实际范围|dualP13诊断 mode0|tripleP10强控制 mode1|四P8＋修复 mode2|
|---|---:|---:|---:|
|真实8块分别冷/无BP|134060|133916|131301|
|64连续冷/无BP|1138331|1136507|1088854|
|64连续暖/无BP|1136483|1134659|1087006|
|64连续冷/有BP|1218372|1216377|1168292|
|64连续暖/有BP|1216447|1214452|1166367|

所有模式具有同520B z/416bit向量口、Q1/Q2/source/psum/consumer状态及公共加法链。mode1在同一30bit链内计算前三个P的signed10更新，第4P另发；配置时用同八条链在864个已付Q1配置beat中计算各rank正负和，验证所有二值源及前缀的z都在signed10范围内。新增证明状态26B、修复delta状态8B，由三臂共担；范围不合的极值fixture精确回退dualP13。完整预算与费用见 [resource_contract.json](resource_contract.json)。

mode2把每个13bit z表示成 `signed8(low)+256*signed5(high)`。低位signed overflow产生−1/+1修复义务，随后额外读取z并用同32bit链的四个5bit段更新high，多付2拍和1读/1写/1ALU issue。Q1结束后还对10个z行付20拍，把high改为 `high-low[7]` 并原位变成标准13bit，Q2直接读取拼接值。证明、低位更新、修复、规范化及Q2 MAC均绑定到 [modular_core.sv](modular_core.sv) 第117–131行的唯一八条链；没有32个独立高位加法器。z读在第54–70行使用明确共享地址表达式，写回整52bit字。

16fixture×3模式×有/无BP×冷/暖共**192命令**，另64连续12命令，raw p、实际J20、I24各核对**3686400值**，全部通过。配置时RTL得到的正负界逐rank与TB独立累加一致；无reset暖命令复用证明和静态权重，重新支付source/origin/identity。新的mode0无BP小fixture周期与旧phase_borrow dualP13逐项一致。

[verify.py](verify.py) 从源bits和Q1逐K重建真实部分和、centered低/高位、每次修复义务和最终13bit，逐fixture核raw与全部事务/周期；[verify_stream.py](verify_stream.py) 对64原生块独立重建同样义务。没有用分析值驱动RTL。

|64块独立核算|三P10|四P8|
|---|---:|---:|
|Q1更新issue|76424|60113|
|高位修复issue|0|0|
|规范化issue|0|640|
|Q2 MAC|198720|198720|

净差恰为 `3×(76424−60113)−2×640=47653`。真实8块/64块没有触发高位修复，不能据此宣称任意输入都无修复费。all-one fixture实际触发150个修复事务/600个字段，padding poison为270/270，正负极值各100/3200；这些非零义务均与独立模型一致。全零fixture反而固定多20拍，候选不是所有输入都获益。

可保留为执行接口候选。carry分段、模数表示、低精度累加本身有强先验，本项不认定新颖性，也不把这约4.19%的组件周期收益称ASIC速度或能耗。mode0/1/2在同模块共担选择器/证明/修复硬件不构成综合面积证据；不能与旧208bit z口跨核比较同面积。尚未与direct/halo或双context组合，后续多context必须保留D3普通RR强控制，不能用seq单独作分母。本项没有跑19200块或重新评价网络质量。

复现顺序：`python3.12 run.py`，`/opt/anaconda3/bin/python3.12 verify.py`，`python3.12 run_stream.py`，`/opt/anaconda3/bin/python3.12 verify_stream.py`。使用Verilator4.028 `-Wall --cc --exe`；`build_glue.py`只在本目录生成wrapper/TB，完整消费者与输入fixture来自旧只读树。计划见 [PLAN.md](PLAN.md)，原始收据见 [results.json](results.json)、[results_64.json](results_64.json)、[checks.json](checks.json)、[checks_64.json](checks_64.json)。

本轮后续已另外完成双context RR融合：[rr_modular结果](../rr_modular/README.md)，相对RR+triple10在64块省3.281%；仍未与RR+借consumer64比较，不将单核4.1929%外推。
