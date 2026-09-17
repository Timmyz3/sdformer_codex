# 活动前沿预取：整链接口补齐

B：上一版 source 已转为四前沿，但child PF仍只针对旧最小rank通道，甚至可能不属于当前选中集合。叶级修复带来小幅净服务，必须在已统一root-bank的joined链复测。

A与对照：直接继承上层五臂、同两权重函数和同34个P32输入（32训练+2诊断）；106独立MAC、256KiB参数池、额外独立Y/routes/U/门桥、128B X及128B图cache全部相同。只开启source子目录已测试的active-frontier child PF，其他状态、选择与容量不改。static和one-channel路径应逐字段复现原整链，作为直接回归控制。

无新增算法训练/精度 claim；普通code共享同一改动。TB另独立统计每次producer批实际不同channel数，补齐joined的source_channel_refs对账，而不是仅打印核内计数器。未取得DC/PT/FM/PPA前，周期只属于隔离Verilator。
