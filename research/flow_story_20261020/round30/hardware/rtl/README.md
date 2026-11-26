# R30 实际位平面运算数据通路

这里合成的是整数计算通路，不再只有选择控制器。`binary_plane_pe` 每拍处理 128 个 binary 激活与 16 组 128-bit 权重平面，通过 popcount、带符号平面系数和 16 个 int32 accumulator 更新输出。Q8 接收 bit0…7，Q4 首拍即从 bit4 开始；bit7 系数 −128。16 路普通 `parallel_gateadd_pe` 每拍处理 16 个 binary 激活与每输出 16 个 signed8 权重，使用同样 2048-bit 权重接口和 16 个 int32 accumulator；**激活接口分别是 128-bit 与 16-bit，并非所有端口等宽**。位平面激活重读/供数必须由存储系统计费。它是普通 Q8 强对照之一，不是 FP32 native 计算单元。

row-group 描述符与首个数据 beat 一同提供且在接收沿前稳定；`row_start/mode_q8/k_chunks` 决定该行的发射序列。背压保持输出和状态，旧输出退休与下一行首 beat 可以同拍发生。没有 stalling 时，连续多行的总拍数是实际接收 beat 总数加末尾一个退休拍。这个协议没有证明 SRAM/地址系统能每拍供足数据；后者仍是模型假设。

Verilator 4.028 的真实仿真通过 206 个 row-group：110 个合成用例覆盖 K=1/16/96/128/129/256/384/768/864/1536/3072、零/全一/负数/极值/随机权重；另 96 个事务来自冻结 R26 的真实 sn1/sn2 与原量化矩阵，来源/坐标/通道/SHA 在 `ACTUAL_VECTOR_MANIFEST_V1.json`。其中 16 条 Q8 重复，按层/坐标/输出块/模式去重是 80 条，不能称 96 个独立样本。FC2 输入来自原生 sn2，没有重新计算混合模式后的上游响应。这些真实操作数只验证层内整数函数，既不是新模型前向，也不是 R30 新质量。每个引擎分别无停顿/随机输入气泡和输出背压，累计检查 13,184 个 int32 输出值；另六个缺描述符、零 chunks、中途重启错误被拒绝。

| 实际仿真流 | 无停顿拍数 | 实际接收 beat | 输出 row-group |
|---|---:|---:|---:|
| binary128，206 个混合 Q4/Q8 用例 | 8,401 | 8,400 | 206 |
| ordinary signed8×16，同一整数目标 | 10,311 | 10,310 | 206 |

这张表是测试流验收，不是部署速度比；Q4/Q8 例数与 K 分布是测试选择。逐 K 无停顿 arithmetic beat 为 Q8 serial `8 ceil(K/128)`、Q4 serial `4 ceil(K/128)`、普通 signed8 `ceil(K/16)`，结果每组 16 输出。K=96 时分别 8/4/6，K=384 时 24/12/24，K=864 时 56/28/54，K=3072 时 192/96/192；因此 Q8 serial 不天然强于普通并行。

两路都由 DC V-2023.12-SP3 在同一 TSMC28 TT0.9V/25°C 库、4ns 时钟、0.05ns uncertainty/transition、0.2ns I/O delay 和 0.01pF 输出负载下独立综合：

| 模块 | cell area（库面积单位） | worst setup slack |
|---|---:|---:|
| binary128×16 | 9,930.829984 | +1.78296ns |
| signed8 gate-add16×16 | 9,890.062045 | +2.00306ns |

面积比约 1.00412；两者在这组约束下均满足时序。这给原模型 binary128 对普通 int8×16 的**局部整数资源比**提供了一个实物参照，但不证明统一 FP32/BN/PSN/访存机器的面积或频率，更不回调已冻结配额。保留 high-fanout TIM-134、DesignWare 未用端口 LINT-28、共享常量 LINT-33 和输出命名 VO-4 等工具提示；不称零 warning。没有布局布线、功耗、门级或形式等价结果，cell area 也不能与芯片测量能效直接比较。

RTL 是单行上下文、权重外部流入；没有实现模型假定的多行 accumulator RF、跨行权重 RF 复用、SRAM、DMA、行打包、tile 选择器、int→FP32 scale、current BN 或网络。其权重端口流必须由未来存储系统兑现。BN 累加次序仍未证明与 GPU 一致；这些结果只证明真实低平面工作可取消及此局部整数函数可综合，不能将主成本表改称 RTL 测得的整网周期。

`build_v1.log` 保留一次工具版本不支持 `--build` 的失败；随后采用 `verilator --cc --exe` 加 `make`，没有为此改变 RTL。原始展开、编译、DC 日志及映射 netlist 在 `../rtl_results/`。模拟 wrapper `pe_compare_top` 只用于测试；DC 分别选择两种真实 PE 为顶层。
