# HIT-Flow事件生命周期Router RTL实现、验证与审阅

**日期**：2026-07-13  
**状态**：首个稳定RTL切片通过本地仿真、断言、静态lint和通用综合检查；不是完整LR-HTT签核  
**对应规格**：`docs/63_HIT-Flow_LR微架构与RTL前接口规格.md`

## 1. 本轮实现范围

本轮实现的是ATLIF事件离开DP-TME之后的生命周期路由前端，不是完整encoder，也不是最终SRAM系统：

```text
ATLIF event + static consumer_class
    ├─ single：一项弹性直通缓冲
    ├─ fanout：一份数据，Q/K两个独立消费状态
    └─ pair：按tag收集Q0/Q1/K0/K1并组成128-bit时间对
```

RTL文件：

| 文件 | 功能 |
|---|---|
| `hitflow_single_event_buffer.sv` | 单消费者一项弹性寄存器，可同拍retire/replace |
| `hitflow_fanout_event_buffer.sv` | Q/K独立反压和独立消费位，可同拍完成旧项并接收新项 |
| `hitflow_qk_pair_assembler.sv` | 四slot乱序组装、tag检查、重复slot拒绝、同拍切换下一tag |
| `hitflow_event_lifetime_router.sv` | 静态路由、非法route拒绝、五类事务计数 |

当前路由编码为`single=0`、`fanout=1`、`pair=2`，`3`为非法输入。`pair_data`固定按
`{K1,K0,Q1,Q0}`输出。所有出口采用ready/valid，反压期间payload和tag必须稳定。

## 2. 代码审阅中发现并修复的问题

### 2.1 已修复：fanout退休气泡

初版只有在内部项完全无效时才拉高`in_ready`。当Q先消费、K后消费时，K完成后的下一拍才能接收新项，每个fanout事务最多引入一个无意义空拍。

修复后以两个pending位的消费后状态计算`entry_done`：最后一个消费者握手时可同时装入下一项。新项不会复用旧pending位，下一拍重新向Q/K各发送一次。

### 2.2 已修复：pair上下文切换气泡与误报

初版完整pair即使`out_ready=1`也拒绝新slot，并会把下一tag误判为`tag_mismatch`。修复后完整pair退休与下一pair首slot可同拍发生，首slot直接建立新tag和one-hot存在位。

### 2.3 已修复：异常缺少可观察性

新增三个组合错误输出：

- `pair_tag_mismatch`：未完成pair收到不同tag；
- `pair_duplicate_slot`：同一tag重复写已存在slot；
- `route_unsupported`：收到未定义路由。

三类错误均拒绝握手，不静默覆盖旧数据。第一次补断言后，bind文件漏接四个端口，Verilator报告`PINMISSING`；本轮已修复并重新执行完整回归。未接线的那一轮结果不计入最终验证证据。

## 3. 验证内容与结果

### 3.1 定向仿真

Icarus Verilog和Verilator分别执行相同自检testbench，覆盖：

1. single连续反压时valid、data、tag稳定；
2. fanout的Q先消费、K后消费，已消费出口不重复发送；
3. fanout最后一个消费者完成时同拍接收下一事件；
4. pair四slot乱序到达，输出拼接顺序正确；
5. 不同tag、重复slot和非法route均拒绝；
6. 完整pair退休时同拍建立下一tag；
7. 11个输入event、1个single、2个fanout Q、2个fanout K、2个pair的计数器结果正确。

结果：Icarus与Verilator均打印`PASS: HIT-Flow event lifetime router`。

### 3.2 SVA

当前绑定断言覆盖：

- single、fanout-Q、fanout-K、pair四个出口在`valid && !ready`期间保持valid、data和tag稳定；
- pair tag错配或重复slot时`in_ready=0`；
- 非法route时`in_ready=0`。

Verilator断言仿真通过，且最终日志无缺失端口警告。

### 3.3 静态与综合检查

| 检查 | 结果 |
|---|---|
| Erie独立静态lint | 4个RTL文件均0 error、0 warning |
| Verilator lint | 通过 |
| Yosys `hierarchy/check` | 0 problem，无锁存器 |
| Yosys通用结构 | 113 generic cells、0 memory |

113个generic cell只用于发现结构异常，不能换算为标准单元面积、功耗或频率。与修复前相比增加的组合逻辑换来了fanout/pair无气泡切换和显式错误检测，是否处于关键路径必须由目标工艺DC判断。

Icarus关于`unique case`被忽略的消息是仿真器能力提示；Verilator和Yosys均识别该语义。远程Vivado/FPGA依赖未配置，本轮按项目目标只声称本地Icarus、Verilator、Erie静态lint和Yosys结果。

## 4. 审阅结论

### 4.1 已证明

- 三类静态生命周期路由的基本ready/valid语义正确；
- fanout不会因两个消费者反压不同而重复或提前释放；
- 一个tag内的Q0/Q1/K0/K1可乱序组成固定128-bit pair；
- 同拍retire/replace消除了前端人为气泡；
- 当前RTL可综合，不含锁存器、仿真专用语句或未解析模块。

### 4.2 尚未证明

- 只有一个pair上下文，尚未实现规格中的`N_CTX=1/2/4`和context选择；
- 当前是寄存器切片，没有resident SRAM、ping-pong bank、spill或epoch/sequence保护；
- single的80.13%只是静态直通资格上界，尚无ordered trace证明真实直通率；
- 未验证随机长反压、计数器溢出、复位中途到达和百万事务稳定性；
- 未与软件真实event/tag trace逐事务差分；
- 未做目标标准单元库DC、STA、门级仿真、SAIF功耗或Formality；
- 未实现DP-TME、TESSA/SCS、CCSP/FGP、RPI和descriptor scheduler。

因此本切片可以进入下一阶段集成，但不能称为“完整LR-HTT”，更不能单独支撑DATE架构贡献或芯片PPA。

## 5. 对架构设计的直接指导

1. **保持静态路由而非动态预测**：consumer类别来自冻结执行图，硬件不需要分类器，错误route必须在仿真和系统计数器中可见。
2. **fanout保存一份payload和两个消费位**：这比复制两份event更符合12个`proj_sn`点的真实消费者关系；最终SRAM也应保存一次payload、维护两个读状态。
3. **pair上下文必须扩展但不能盲目堆叠**：单context功能已稳定，ordered trace给出p99并发tag和等待周期后再决定2或4 context。
4. **错误检测进入descriptor审计，不进入正常数据路恢复**：固定部署中tag错配、重复slot和非法route应为零；若发生说明调度器或地址生成错误，不应覆盖后继续运行。
5. **下一RTL优先级是DP-TME lane和多context pair bank**：前者决定44.244亿时间MAC的吞吐/面积，后者决定128-bit时间对供数是否成立；两者都比继续美化单项buffer更影响架构结论。

## 6. 下一轮签核门槛

- 加入随机反压scoreboard，至少10万输入事务、零丢失/重复/乱序；
- ordered profile完成后重放真实tag序列，报告single直通率、resident读写、pair p50/p99等待和最大并发上下文；
- 实现`N_CTX=1/2/4`同一RTL参数化版本，并在相同trace下比较周期、寄存器/SRAM容量和DC面积；
- DP-TME lane与整数golden逐位差分后，才把router接入计算通路；
- 目标库DC前不引用113 generic cells作为PPA结果。

