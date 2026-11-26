# BN 局部归约：地址与 carry 控制器 RTL

本目录把 [前轮地址模型](../bn_address_validation/README.md)中的一个明确下层做成可综合 RTL：**单个未退休操作的 ping-pong 地址/carry 控制器**。Verilator 的 47 组检查和本机 DC 综合已完成；这是实现者自测，另由根代理审查。它不实现浮点 BN，不复现前轮 32 操作并发日程，不给完整 BN 或整网周期。

## 接口与有限容量

`rtl/bn_reduce_addr.sv` 接收 samples、channels（均 13 bit），用完整 26 bit 乘积检查 `0 < samples×channels ≤ 4096`。两个 scratch 区固定为 `[0,4095]`、`[4096,8191]`。输入必须已按 `sample*C+channel` 放入第一区；初始装载和结果读出在本模块外。samples=1 直接返回第一区，不发虚假归约命令。非法几何返回 fault，不发命令。

每个 pair 命令给出 `src0=source+2p*C+c`、`src1=source+(2p+1)*C+c`、`dst=destination+p*C+c`。奇数尾项单独发 `carry=1`，需要实际 read→copy→write；其未使用的 src1 明确置零。bank 为地址低 4 bit，row 为其余 9 bit。RTL 输出 bank/row 是连线映射，不包含 bank 仲裁器。

`cmd_valid/cmd_ready` 接受命令后，控制器等待 `retire_valid/retire_ready`。**consumer 只有在该命令所有读和目标写实际退休后才可确认**。确认之前无第二个命令；最后一项确认之后才交换 scratch 区并进入下一级，防止读前覆写。命令在 backpressure 下保持稳定；完成结果同样有 finish ready/valid。一次只预留一个操作的接口，其消费者可用 3 个 word 保存两个操作数与结果，但这些数据寄存器不在本控制器面积内。

这是前轮数学 jobs 序列的串行实现，具有相同 src/dst/carry 公式；并不与前轮可配置多 lane、32 活跃操作的 tick 日程逐周期相同。

## 已测范围

`case_manifest.json` 绑定前轮实际形状结果与源码 SHA。前轮 54 条 BN 记录产生的 15 类 full/tail chunk 全覆盖：`10×384, 1200×1, 12×192, 18×192, 1×384, 1×768, 21×192, 2816×1, 300×1, 3072×1, 39×96, 4096×1, 42×96, 5×768, 704×1`。另加 27 个奇偶/多通道 toy 和 5 个非法容量例，共 47 例。不是遍历全部 976,584,960 个生产图输入元素。

`tb/tb_addr.cpp` 用独立 pair/channel 数学循环生成期望地址，不调用 Python 调度器或 RTL 的递增公式。有限 8192-word 行为 RAM 保存整数值、通道号和样本区间/数量；每次 pair 要求同通道、区间相邻，最终与逐通道独立整数求和比较。行为 consumer 严格 read0→可选 read1→add/copy→write，每周期最多一个读，即使两源同 bank 也序列化；这些延迟和随机停顿只用于协议压力测试，不是 SRAM/浮点标定。

已接受 41,309 个操作，其中 4,369 个显式 carry；随机 command backpressure 13,666 次。retire 延迟、finish backpressure、无复位连续处理 47 个 chunk，以及配置被接受后任意改变输入均通过。详细每例计数见 `results/verilator/cases.csv`；其中 cycles 是该行为 consumer 的合成测试周期，不能外推为 BN 延迟。

## DC 综合实测与约束

| 项目 | 本次值/范围 |
|---|---|
| 工具 | 本机 Synopsys DC V-2023.12-SP3，2 CPU cores |
| 库 | TSMC `tcbn28hpcplusbwp7t35p140tt0p9v25c.db`，TT 0.9 V / 25 °C |
| 时钟约束 | 4.000 ns；uncertainty、clock transition 各 0.050 ns |
| I/O | 输入/输出 delay 各 0.200 ns；input transition 0.050 ns；每输出 load 0.010 pF |
| total cell area | **850.444003（DC 库面积单位）** |
| worst setup slack | **+2.41639 ns**，该 4 ns 约束下 |
| 报告宏/black boxes | 0 |
| 约束报告 | no violated constraints |

库和约束沿用光流 date12 的 `hardware/scripts/synth.tcl`，没有读取通用线文件。原始 `area.rpt`、`timing.rpt`、`units.rpt`、`constraints.rpt`、mapped netlist/DDC 和日志均保留在 `results/dc/`。面积仅包括配置容量判定、计数器、地址生成、ready/valid 与屏障状态机；其中容量乘法检查占有明显逻辑面积，未宣称最优。面积不含 8192-word SRAM、读写仲裁、操作数/结果数据寄存器、FP32 add/copy 单元、跨 chunk 累加、mean/variance/rsqrt/normalize、DMA 或全引擎。

**不是零 warnings。** 综合保留了：全覆盖 2-bit FSM 的 unreachable default（ELAB-311）；Verilog assign 与未使用输出网的导出提示（VO-4、VO-11）；`check_design` 的优化算术输入/进位未连接（LINT-28）、算术直通（LINT-29）、地址与 bank/row 的共享输出连线（LINT-31）、乘法部分积共享网（LINT-33）、result_base 的固定零位（LINT-52）。完整分类计数见 receipt；本次没有把这些报告删去。零宏/black boxes 和无违反约束并不等于做过形式等价、门级仿真或布局后时序。

## 边界与公平性

这是全部 native、固定布局、普通评分和光流评分都可使用的公共控制组件，不是光流专属算法增量。未证明 PyTorch 原生 BN 的 FP32 归约顺序、数值误差、整网 PE 净节省、功耗或能效；也不能将本控制器面积配上此前合成参数的 32-op 周期声称一个已实现的完整 BN 引擎。RTL 对真实 SRAM/计算 consumer 的退休承诺有依赖，后续仍需实现该接口及有限数据 buffer，再接入完整 BN 和同图 PE 分母。

```bash
bash scripts/run_verilator.sh
bash scripts/run_dc.sh
```

这些命令会更新本目录 build/results；要保留已封存证据，应先复制目录或改输出路径。冻结 receipt 绑定的是本次已有文件。
