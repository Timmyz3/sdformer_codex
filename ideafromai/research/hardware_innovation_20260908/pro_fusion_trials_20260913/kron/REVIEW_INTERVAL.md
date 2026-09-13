# interval 独立只读审阅

2026-09-13，kron 代理审阅 [interval 报告](../interval/REPORT.md)、[RTL](../interval/interval_tile.sv)、[TB](../interval/tb.cpp)、[prepare.py](../interval/prepare.py)、[analyze_results.py](../interval/analyze_results.py) 与30行 [原始周期](../interval/rtl_results.csv)。未改对方文件，未重新运行其RTL；本次为源码/账本独立审阅，不充作第三方盲审。

**未发现推翻该固定 tile 负结果的功能或计费漏洞。** D 的 XOR/方向、加减、完整时间前缀及最终 RNE 都在 RTL；前缀在唯一末级 RNE 前发生，不改变整数函数。P2边界在 PREFIX 从组19跳到22，正确跳过第二空间点的 t0。直接/端点同用8条32bit加法链、同320词状态、同双读单写权限；没有为前缀另配数据加法器。直接模式已按 union-K 装一次16系数并复用全部20个T/P消费者，端点没有偷得减少读权重的弱分母。

周期账本闭合：真实 A=557、E=1088，直接/端点分别发射1114/2176向量加减，端点另有36前缀拍；故3502−2404=2×(1088−557)+36=1098。模式2的第一次864次读、额外SELECT、第二次读和等待均有计费。TB检查每命令分项总和、30命令间不reset、最后40个输出及背压稳定；不同调度遭遇同一周期阻塞规则而产生不同stall，是合理的时序差异。

需要保留的三个边界：

1. **快照命名。** 报告和prepare把 matched dense stage320 写作“较旧/legacy”，但当前 major profile 的父学生就是 matched dense stage320。建议用 `stage ordinary R24+onepass` 与 `breadth matched dense stage320` 中性名称。两者的实际门与配套系数没有混用；这是版本叙述修正，不推翻当前ordinary测值。kron采用dense，不能把两试验横向拼成同学生收益。
2. **接口与装入。** 入口是已打包P2×T10 mask，权重从请求端口ROM读出；真实capture→mask gather/打包、片外系数装入、生产者PSN均在范围外。报告已列明。该结果是所述入口至完整U16输出的局部执行，不能与包含系数预装的kron总周期直接比较，也不是旧full_chain SR64/SW64机器的完整迁移。
3. **物理范围。** 组合优先编码、popcount、两组状态读与8条显式carry chain有真实面积/时序代价；未综合不能称同Fmax/同面积。真实数据不触发sat24，报告已准确注明；不要把9,600值零差延伸为所有数值边界均穷举过。

结论支持保留这份完整强 A/失败边界：当前原T顺序的同一固定tile，付费端点和付费选择均不优；全一控制验证长区间时机制可获益。它没有提供新的跨帧相关性证据，也不能关闭其他真实长区间接口或恢复训练家族。
