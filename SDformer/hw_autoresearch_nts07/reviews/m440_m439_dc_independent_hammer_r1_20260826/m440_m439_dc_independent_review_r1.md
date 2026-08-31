# M440：M439 serial-vs-dual adapter DC 独立打铁

结论：**89/100，PASS standalone DC，严格限制为 adapter-only proxy。** 无 P0；4 个 P1、4 个 P2。可以继续做各自 RTL→mapped netlist Formality、独立 PrimeTime STA 和 full-population 集成，但目前不能升级为 Conv/系统 PPA、功耗或能效结论。

## 不信 receipt 的独立复核

独立脚本没有读取 M439 receipt；它重验 M433/M434/M439 内外封，并从两套原始 `dc.log`、area/QoR/setup/hold/check/constraint 报告和 mapped netlist 重解析。结果如下。

| 指标 | M405 serial | M433 dual co-read |
|---|---:|---:|
| Cell area | 18,984.797641 µm² | 8,351.405814 µm² |
| Cells | 15,832 | 7,139 |
| FF | 3,340 | 1,348 |
| Logic levels | 34 | 52 |
| Setup worst slack | +1.5583 ns | +0.8411 ns |
| Hold worst slack | +0.0251 ns | +0.0251 ns |
| Macro / blackbox / latch / loop | 0 | 0 |
| Constraint violations | 0 | 0 |

两点使用同一 TSMC28 HPC+、3 ns、0.1 ns clock uncertainty、0.2 ns I/O delay、0.01 pF output load、25 ps mapping hold guard；均为 ideal clock、ZeroWireload、0 macro。mapped FF 实例数分别精确等于报告中的 3,340 和 1,348。

独立复算得到：dual/serial 面积 `0.4398996488`，即 `-56.0100%`；FF 比 `0.4035928144`；宽块原始服务率 `2×`；因此 **adapter-only wide-block throughput/cell-area proxy** 为 `4.5464914683×`。逻辑峰值输入带宽增至 `1.5×`，物理接口带宽增至 `1.6666666667×`。

## 公平性判断

serial 宽块是 low/high 两拍接收，经 assembly 和两深度 completed FIFO，输出两次 1152-bit partial contribution；dual 同拍接收 low+high，经一深度弹性输出，输出一次完整 signed12 delta。FIFO2 与 elastic1 的差别不是漏综合：FIFO2 是 serial 在拆分输出下维持其服务率所需的机制内缓冲，而 dual 的一项弹性寄存器可在 pop+push 下保持 II=1。但这仍不是 iso-depth 对比，只能说“各机制达到宣称服务率的 standalone adapter 成本”。

更重要的是，两个 top 都没有后级 `old_psum + delta` 累加器；也没有 SRAM 宏、读端口、160 B 输入布线和互连。因此：

- `4.546491×` 不能叫 Conv throughput/mm²、加速器能效或系统倍速。
- `-56.01%` 不能叫系统面积下降。
- dual 的 144 logical / 160 physical B/cycle 不能叫免费 SHARED96 升级。
- `2×` 是宽块 adapter 原始服务率，不是含 narrow、correction、matcher、DMA、commit 的混合 population 倍速。

M430 的 `517,041,352 cycles / 1.4353753005× vs strong zero` 仍是四个冻结 H67 bottleneck Conv3x3 的全 population 周期仿真点。M439 只说明 adapter logic 在 3 ns 下可综合；它没有把 M430 重新准入为 mapped RTL、资源归一化或系统结果。

## 问题与下一门

- P1：缺各自 RTL→mapped Formality。serial 与 dual 接口及状态机不同，禁止互相做等价；应各自对自己的 mapped netlist。
- P1：缺 PrimeTime；下门必须显式恢复 slow-max / fast-min library，而不能只依赖 mapped SDC。
- P1：缺 full-population adapter+accumulator 集成及 160 B port/SRAM/interconnect 成本。
- P1：M430 1.435375× 仍非 matched-resource PPA 点。
- P2：dual 52 层、setup margin 比 serial 少 0.7172 ns，但两者在当前 3 ns DC 下均满足。
- P2：当前没有 input transition/driving-cell，且 ideal-clock/ZeroWireload；只能称 prelayout。
- P2：dual 唯一 postcompile design warning 是 `busy` 与 `debug_output_full` 有意别名的 LINT-31；Formality 中应显式处理。

下一步决策：**GO** 到各自 mapped Formality；**GO** 到独立 prelayout PT STA；**GO** 到保留 old-PSUM 语义且显式计入 160 B 端口的 full-population integration；**NO-GO** 到 paper PPA、power/energy、Conv throughput/mm² 或系统 headline。
