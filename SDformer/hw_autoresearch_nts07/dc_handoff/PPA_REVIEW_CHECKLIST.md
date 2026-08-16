# Synopsys 结果准入清单

这份清单用于服务器运行后的人工签收。脚本 PASS 只代表工件与身份合同
完整，不自动代表 ASIC signoff。

## 共同身份

- `docs/359` SHA 为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- Fixed/RQTB 使用同一标准单元库、宏库、时钟、角点和活动激励。
- Local5 1R1W/1RW 使用同样的 matched 条件。1RW 只是端口敏感度基线。
- 只有实例化真实 SRAM/RF adapter 且 macro reference 通过审计时，
  才允许 `PPA_ADMISSION=1`。

## DC / Formality

- `check_design_postcompile.rpt` 无 unresolved reference、multiple driver 或 combinational loop。
- `check_timing_postcompile.rpt` 无 unconstrained clock/end point；
  `timing_unconstrained.rpt` 无有效路径。
- `constraint_violators.rpt` 分别审查 max transition/capacitance/fanout 与 timing。
- `references.rpt` 中宏实例与 `EXPECTED_MACRO_REFS` 一致。
- Formality `reports/formality_status.txt` 必须精确为 `PASS`。

## PrimeTime STA

- setup 和 hold 使用独立库角和独立 `PT_RUN_DIR`，不覆盖。
- 每个角的 `ptsta_check_timing.rpt` 无 unconstrained/error。
- setup 角 WNS/TNS 均不小于 0；hold 角 hold WNS/TNS 均不小于 0。
- `ptsta_constraint_violators.rpt` 无未豁免违例。
- 有 SPEF 时，`ptsta_scope.rpt` 必须是 `extracted_spef`，且寄生参数报告与该
  P&R 网表属于同一 run。

## PrimeTime PX

- SAIF manifest 的 VCD、trace、测量窗、strip path 和 SHA 全部通过。
- SAIF annotation coverage 达到门槛，且 DUT 未注释对象为 0。
- Motion 只比较同激励 Fixed/RQTB；Local5 只比较同激励 1R1W/1RW。
- 禁止将 Motion 10 ns VCD 和 Local5 2 ns VCD 直接横向比绝对功率/能量。

## 论文标签

- 无目标宏：`pre-macro logic DC/STA/PTPX`。
- 有目标宏但无 P&R/SPEF：`post-synthesis macro-aware estimate`。
- 有同 run P&R 网表/SPEF 和目标角 PT：`post-layout estimate`。
- 以上均不等于流片或 silicon measurement。
