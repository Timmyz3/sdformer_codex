# Bias提交即输出的Banked Accumulator RTL审阅

## 1. 本轮完成范围

本轮实现`hitflow_banked_accumulator.sv`，接收分段多播给出的逐bank更新，将signed 17-bit产品
符号扩展到signed 32-bit累加域，并维护每个token、每个output tile独立的累加状态。

完成的功能包括：

- 2-bank、每bank一项在途read-modify-write；
- 首次更新按零初始化，后续更新读取旧accumulator；
- product与bias使用相同更新接口；
- 每个token的bias只能提交一次；
- bias更新的和直接作为最终输出；
- final输出逐bank独立反压；
- 全162个token完成bias后才允许group finish；
- signed 32-bit溢出sticky检测和活动计数器。

尚未完成NMF、product、多播和accumulator的同一顶层连线，也未加入真实trace、输出requant、
RPI、SRAM宏模型、DC和SAIF。

## 2. Bias-Commit Output Drain候选

传统流程为：

```text
product累加 -> bias写回acc SRAM -> 再读acc SRAM -> 输出
```

当前数据流改为：

```text
product累加 -> bias read-modify-write -> final握手时同时写回并输出sum
```

即bias是每个token/output tile最后一次更新时，`old_acc+bias`已经是最终整数结果，可直接送给
requant/RPI，不需要再读一次accumulator。暂命名为Bias-Commit Output Drain（BCOD）。

BCOD是exact时序重排，不改变加法顺序或数值。它目前只是条件架构点，只有动作计数和目标库结果
证明减少最终读操作、输出反压没有显著拖住bank、完整projection EDP达到门槛后，才可写入DATE
贡献。对照组必须保留“bias写回后单独read-out”的传统流程。

## 3. Bank与SRAM合同

默认配置：

```text
TOKENS   = 162
BANKS    = 2
OUT_TILE = 8
ACC_W    = 32
```

每bank逻辑容量为81×256-bit，bank地址为`token_id/BANKS`，bank号为`token_id mod BANKS`。
每个bank当前只允许一个在途更新：

1. 接收更新并发起同步读；
2. 下一拍得到旧值并形成32-bit和；
3. 普通product自动写回；
4. bias更新等待final ready，握手后写回并释放bank。

这个两拍、无旁路版本是正确性基线，不是吞吐最终版。若ordered trace显示bank busy反压占周期超过
10%，再实现连续流水与同地址旁路；否则不为理论峰值增加控制和比较器。

## 4. 验证结果

### 4.1 定向仿真

Icarus和Verilator均通过以下场景：

1. bank0/bank1同拍接收相同product、不同token；
2. token0连续两项正负更新；
3. token4单独更新，其他token无product；
4. 六个token统一提交bias，bias-only token正确从零开始；
5. bank0 final阻塞时bank1可先完成，bank0数据保持稳定；
6. 每个token最终两个lane逐值比较，无checksum替代；
7. 重复bias报告协议错误且不接收；
8. 全部六个bias完成前group不能结束；
9. 更新、写回、bias、bank stall与final stall计数正确。

生产RTL Verilator `-Wall`为0 warning、0 error；绑定SVA验证final反压稳定、非法更新不握手和
finish时无在途final，构建日志0 warning、0 error。

### 4.2 Yosys错误路径与正确路径

第一次使用完整`memory`把accumulator强制映射为寄存器/mux，得到24,064项undriven问题。根因是
SRAM数据本来不复位，只有valid位定义其可见性；强制展开后Yosys把未初始化物理字逐位报告。
这一路径明确判定为失败，不能引用其cell数，也不能声称通过。

修正读模板后，使用`memory -nomap`保留宏边界：

```text
Found and reported 0 problems
wire bits = 8,261
cells     = 258
$mem_v2   = 2
```

两块memory均为：

```text
SIZE          = 81
WIDTH         = 256
RD_PORTS      = 1
WR_PORTS      = 1
RD_CLK_ENABLE = 1
WR_CLK_ENABLE = 1
RD_CLK        = clk_core
WR_CLK        = clk_core
```

因此当前RTL给出了两块同步1R1W SRAM的可替换合同。Yosys外围258个generic cells仍不是目标库
面积；SRAM本体面积、读写能量和时序必须来自目标memory compiler或明确的宏模型。

### 4.3 Erie

Erie内置Verilog-2001启发式对4个参数化`for`循环报告4 error和4 warning；这些循环的上界均为
SystemVerilog参数`BANKS/OUT_TILE`，生产RTL已由Verilator完整展开且Yosys保留宏检查为0问题。
该结果记录为方言不匹配，不伪报Erie全绿。若DC前端不接受当前SystemVerilog子集，需要生成受控
Verilog-2001展开版并做等价验证，而不是手工改写后跳过回归。

## 5. 已知风险

| 风险 | 当前处理 |
|---|---|
| 每bank两拍一项，峰值利用率仅50% | 先测真实stall，再决定流水旁路 |
| bias final反压会占住bank | 独立bank ready；统计final stall；必要时加输出弹性FIFO |
| 32-bit是否充分尚未冻结 | 保留overflow sticky；等待真实投影权重和valid825量化 |
| SRAM读写同址语义依赖宏 | 当前不允许同bank第二项在途；宏wrapper明确read-first合同 |
| 全量valid位同步清零 | 仅162 bit；后续可比较epoch tag，不能把acc数据本体复位 |
| BCOD可能只是常规融合 | 做传统read-out同约束对照，只按增量动作和EDP表述 |

## 6. 下一步

1. 增加G1 projection集成顶层，连接product、多播与accumulator；
2. direct active-entry与NMF term共用同一后端，做随机整数逐token等价；
3. 增加传统bias写回再读出模式，形成BCOD公平消融；
4. ordered trace完成后统计bank busy、final stall、每段交付效率和SRAM动作；
5. 用真实int8折叠权重确认ACC_W、overflow、requant和valid825 AEE/AAE；
6. SRAM wrapper、SDC和目标库就绪后才能进入DC。

当前结论是：banked accumulator的精确接口和同步SRAM合同已经成立，但完整G1 projection仍未
闭环，BCOD也尚未达到论文贡献冻结标准。
