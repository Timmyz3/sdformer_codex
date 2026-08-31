# M370：G7 bottleneck Conv 激活幅度门 fast-kill

四份冻结身份（H67 ep35 S10、PAFT ep4 S10、paired control ep4 S10、H67
ep35 train-only S32）的全部 248 条记录均显示：四个 bottleneck Conv 输入
逐层严格只有 `{0, a_l}` 两个 float32 值，且同一层的 `a_l` 跨身份不变。

最小非零幅值是 `0.999964952`。因此
`theta={0,1/64,1/32,1/16,1/8}` 的 G7 网格新增 drop 恰为 0；当 theta
越过某层 `a_l` 时，又会从 0% 直接跳到该层 active source 的 100%，不存在
中间幅度 Pareto。

结论：对这四个昂贵 Conv，G7 不进入 RTL，也不占 A800 accuracy 队列。
这不否定按 `|w*x|` 预算的 G11，也不外推到尚无值 trace 的 FC、patch embed
或 attention。无 cycle、accuracy、系统倍速或 headline 准入。
