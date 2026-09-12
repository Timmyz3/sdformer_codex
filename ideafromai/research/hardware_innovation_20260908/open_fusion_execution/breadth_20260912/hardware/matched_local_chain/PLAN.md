# stage320新常量的局部同机绑定

B：新dense/contiguous34/lifting40已重新训练并导出字面参数，现有局部8%服务属于旧参数，不能沿用；源RTL本身不是双消费者净收益。

A：保留同一IntegratedMachine、96×8×48 RF、128KiB state/coef、共用H4目录/驻留MAC、两级source ALU与512×128 ROM。原corner/interior真实I24和冻结preview不变。每个端点从新program.json和新literal constants读取，不从旧数学readout/历史cutoff重构训练阈值。

固定执行：I24→新source→完整K864 preview U32/V/固定BN1/非因果sn2→驻留gate→新Conv2 U16/F＋BN2＋原I24→新consumer gate及PED U24/V96。单Machine接续，没有source/sn2中间DMA退回/重新灌入。真实输入/系数冷填/所有RNE/sat、原I24重读与端点写出保留。

先独立重建source字面整数函数；冻结preview用独立C++逐项FP32 fmaf参考，不从Machine抽值当gold；再由该sn2及新q独立重建整数消费者。所有候选gold只在新view中，旧capture/程序不改。当前尚无新GPU窗口捕获，尤其preview GPU归约与scalar硬件函数可能不同；只以CPU同函数检查作绑定，不继承其AEE端点逐位等价。

只跑三个新端点×两原窗口ready。候选X仍需相对相同训练/编译权限dense与结构稀疏控制考察完整服务；不预支lifting小RF所允许的未实现跨层交织，不开新网络/native实验。
