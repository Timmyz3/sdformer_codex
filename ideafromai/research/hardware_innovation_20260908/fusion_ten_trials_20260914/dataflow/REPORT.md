# 三项数据流/消费者接口的完整RTL筛选

全部使用同一固定Q1/Q2和真实FP32 identity→J20→I24出口，C96/K864/R8/N96/T10；未改数值函数、训练或生产树。三项各有实际SV/TB与最强共同资源控制，不把stall/重启或第三调度控制另计新点。

|本项|实际范围|强控制→候选|结论|
|---|---|---|---|
|[D1完整rawp旁路](d1_forward/REPORT.md)|8真+8边界，64连续|968747→968747拍|共同STORE+forward消除周期差；候选少30720个p_mem写词。|
|[D2两列halo环形复用](d2_halo/REPORT.md)|8真+8边界，64与19200连续|324217349→309563909拍|全帧省14653440源装入拍，4.5196%；普通halo A。|
|[D3有限双context](d3_interleave/REPORT.md)|8真+8边界，3/64与19200连续|普通RR 233289744→阶段错位 260697004拍|普通RR更强；阶段错位失败。generic interleave胜seq，但不称X。|

各项内部同状态/端口/输出backpressure权限。D3真实外置共享producer八ALU/乘法和单Q1/Q2阵列，I24仍独立宽算术；两tile存储比D1/D2多，不能跨项宣称同面积。三项没有在同一模块叠加，所以不能把百分比相乘或许诺组合全帧收益。

所有checkpoints为真实rawp、SV实际转换J、最终I24；gold不用于决定动态门、地址或调度。静态冷配一次、新源/origin/identity每次实计，padding由RTL产生。背压、warm同参数重启、tile身份/末输出都在小块及连续测试；整帧是单go无外部stall。完整收据与作者审计见[SUMMARY.json](SUMMARY.json)、[audit_results.json](audit_results.json)。固定数值函数已有旧stage真实825任务评价，本轮不重跑质量，也不声称与原浮点消费者逐位无损。

本页是作者结果汇总；独立实现审阅见[root审核](../REVIEW_DATAFLOW.md)。没有EDA、Fmax/面积/能耗数据，仿真墙时只说明实际可执行范围。三项均在当前固定工作点收口。
