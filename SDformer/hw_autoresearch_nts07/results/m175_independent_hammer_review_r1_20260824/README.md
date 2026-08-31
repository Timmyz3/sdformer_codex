# M175 96-bit vs M174 128-bit 独立打铁评审

结论：**84/100，模块里程碑 conditional pass；宽度终选与完整 FC2/论文性能 claim 不放行。**

封存输入、M174/M175 DC evidence manifest、M173 DSE manifest 与 docs/359 SHA
均独立校验通过。使用新目录重新编译 M175，并以 seed `1750824` 运行 VCS：
13/13 SVA coverpoints 全部命中，0 assertion failure、0 compile warning/error signature。
功能人口计数保持为 50 beats、7 tokens、367 events、148 groups、1519 replayed
source terms，并覆盖 1 次 same-cycle token rearm 和协议攻击。

## 独立物理报告解析

| 点 | 面积 (um2) | cells | sequential | levels | setup (ns) | hold (ns) |
|---|---:|---:|---:|---:|---:|---:|
| M175 / 96-bit | 1309.266002 | 1783 | 236 | 55 | +0.4731 | +0.0003 |
| M174 / 128-bit | 1530.648002 | 2145 | 264 | 60 | +0.3397 | +0.0003 |

两者使用同一 TSMC 28nm standard-cell NLDM、3.0 ns、ideal clock、
ZeroWireload、flattened flow，且都没有 macro。M175 RTL 与 M174 的有效结构差异是
128→96-bit、16→12 rows 及相应对齐/步长，协议、K4 层级 selector、prefetch、replay
和 same-cycle rearm 保持一致。

M174 的 correction overlay 也已核对：M171 setup report 的第一条、最差 slack 是
`0.0000 ns`，M174 是 `+0.3397 ns`，因此正确恢复量为 `+0.3397 ns`，不是旧字段的
`+0.3385 ns`。

## 数值复算

- 96-bit 面积节省：`(1530.648002 - 1309.266002) / 1530.648002`
  = **14.4632861187%**；回执 `14.463286%` 正确。
- 128/96 analytic throughput gain：`157504597 / 146423753`
  = **1.0756765468x**；回执 `1.075676547x` 正确。
- 128/96 logic-only throughput density：
  `1.0756765468 / (1530.648002 / 1309.266002)`
  = **0.9200983701x**；回执 `0.920098370x` 正确。

所以 128-bit 用 16.91% 更多 selector/control logic 换 7.57% analytic throughput，
logic-only throughput density 反而降低约 7.99%。这些都只是 frontend analytic +
logic-only synthesis 数字，不是 physical、完整 FC2、FFN 或系统倍速。

## M173 分 stage 核对与宽度判断

| width | stage0 | stage1 | stage2 | stage3 | 每层均 >2x |
|---|---:|---:|---:|---:|---|
| 96-bit | **1.994291680x** | 2.714875315x | 2.983965235x | 3.121286152x | 否 |
| 128-bit | **2.151304070x** | 2.898766685x | 3.160067513x | 3.295854900x | 是 |

独立建议是：把 **96-bit 留作效率/集成默认候选**，优先用 stage0 定向调度、有限
reservoir 或跨 beat 合并补回仅 `0.00570832x` 的差距；只有当“所有 stage 都超过
2x”是硬性论文规则，而且真实 bitmap memory 与组合 FC2 仍保留收益时，才选
128-bit 作为性能点。最终选择暂不 admission。

## 合同审计发现

数值精度全部正确，但有两个 P1 口径问题：

1. VCS 合同把 stall cycles 与 consecutive stream hits 写入 directed population，
   却没有显式绑定 seed=1。fresh seed 下这两项从 `150/516` 变为 `138/531`；所有
   功能计数、13 covers 和 0 failure 仍成立。应在 overlay 中绑定 seed，或把这两项
   从不变量人口中拆为 seed-dependent stress counters。
2. `physical A/B point` 对 macro-free、ideal-clock、ZeroWireload DC 来说措辞过强；
   当前只能称 **matched logic-only synthesis A/B**。

## P0 / P1 / P2

P0（阻塞完整 FC2/论文性能 claim，不阻塞本模块 conditional pass）：

- 缺少 96/128-bit bitmap SRAM 端口可行性、带宽争用和 memory energy。
- 缺少 3072-bit K4 weight response/routing、M169 arithmetic、2304-bit accumulator
  context、BN2 与 residual 的组合证据。

P1：exact M51/M173 payload 尚未穿过 RTL；需做匹配 memory A/B、SAIF/PTPX，且修订
seed 与 physical wording 口径。

P2：补 RTL↔netlist Formality、多 seed 回归，并把已验证的 M174/M171 setup correction
overlay 传播到后续汇总。

明确缺失：bitmap SRAM、memory energy、weight response、arithmetic、accumulator
context、完整 FC2、功耗及 P&R。因此本评审不 admission physical/system speedup、
paper-ready PPA 或 headline。
