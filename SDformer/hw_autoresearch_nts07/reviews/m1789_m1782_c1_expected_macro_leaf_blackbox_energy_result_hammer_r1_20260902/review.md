# M1789：M1782 C1 expected-macro-leaf black-box energy 结果打铁

## 裁决

**PASS，99/100；P0=0，P1=0，P2=0。** M1782 的 canonical result、永久 consumed attempt、M1782 contract、M1783 source hammer 与 M1784 release 均通过完整人口和内外 seal 复核；当前七个 source SHA 与 M1782 冻结身份逐一一致，M1784 也绑定 exact M1782/M1783。docs/359 仍为 `dedde7ce...`。

本审阅只开放下述有边界的组件字段。它不把 candidate 的 pending token 改写成系统或流片结论，也未运行 VCS、PrimeTime、许可证查询或任何 EDA。

## 独立复核

Mapped UNIT_DELAY VCS 日志只有一个 PASS，双 bank 公开端口 warmup 覆盖为 epoch 5943/5944，测量 epoch 为 5945。唯一公开计数为：253 cycles、96 issue accepts、48 parent edges、46 macro reads、34 macro writes、2 forwards、30 dead-write elisions、64 commits 和 64 row completions。

TB 的时钟是 `always #1.5 clk_core = ~clk_core`，所以周期为 3 ns。SAIF 的 `DURATION=759.00 ns` 与 `253 × 3 ns = 759 ns` 精确一致。T0/T1/TX/TC/IG 五类记录各 117,690 条，所有 TX 均为 0，逐记录 `T0+T1+TX` 与 759 ns 无不一致。作用域为 `tb_m1772_c1_m1701_two_bank_public_warmup_energy.dut`。

PrimeTime 链接后的 black-box 集合精确等于九个 `TS1N28HPCPHVTB128X128M4S` SRAM leaf：无缺失、无额外对象、无错误 ref、无 hierarchical black box。SAIF 注释为 115,377/115,377 nets 和 107,371/107,371 leaf cells，均为 100%，且 inconsistent annotation 为 0。这里的 100% 是**注释覆盖率**；真正至少翻转一次的网络是 71,439/115,377，即 61.92%，不得把两者混写。

## 可引用的组件功耗

whole mapped C1 top（包含九个 macro Liberty）的报告值与独立 Decimal 重算为：

| 分量 | 功耗 |
|---|---:|
| Cell internal | 26.660183 mW |
| Net switching | 1.74465036 mW |
| Cell leakage | 0.671468437 mW |
| Whole component total | **29.0763016 mW** |

三个显示分量之和为 29.076301797 mW，与报告 total 相差 0.000000197 mW，仅为打印精度。测量窗口能量为：

`29.0763016 mW × 759 ns = 22,068.9129144 pJ = 22.0689129144 nJ`。

层次报告中的 `u_parent_scratch` 为 10.5071545 mW，占报告四舍五入的 36.1%。whole-top power-group 表中的 `memory` 为 10.5068808 mW，两者只差 0.0002737 mW；这是 hierarchy 与 power-group 的归因差异，不是额外功耗，也不能相加。

按 46 reads、34 writes 和 759 ns 单独重算的 datasheet SRAM alternative 为 7,842.54498057 pJ。它只是**独立敏感性点**，绝不能再加到已经包含九个 macro Liberty 的 22,068.9129144 pJ whole-component PTPX 上。

## 论文允许的写法

可以写：在 ep34-density-conditioned 的 253-cycle directed C1 component window 中，3 ns、mapped-gate averaged prelayout PrimeTime PX 估算 whole mapped component（含九个 SRAM Liberty）为 29.0763016 mW、22.0689129144 nJ；internal/switching/leakage 分别为 26.660183/1.74465036/0.671468437 mW。可将 parent scratch 的 10.5071545 mW、36.1% 写成**层次诊断**。

必须同句或脚注限定：standard cells 为 TT 0.9 V 25 C、SRAM 为 SSG 0.9 V 125 C，属于 mixed-corner component estimate；ideal clock、ZeroWireload、无 SPEF。`check_power` 中 12,628 条 ramp 与 2,304 条 load 的 out-of-table-range 诊断原样保留，进一步说明这不是 single-corner signoff。

禁止写成：energy/frame、全网/系统能量、系统加速、total C1 schedule energy、logic-only power、top-minus-macro、PTPX+datasheet SRAM 组合、single-corner signoff、silicon measurement 或 paper-PPA-ready headline。

## 审稿执行边界

本次只读解析 seal、JSON、VCS/SAIF/PTPX 文本并做 Decimal 重算；canonical result、attempt、M1782/M1783/M1784、docs/359 均未写入。没有运行 EDA、simulator、许可证查询、GPU、远端或网络操作。
