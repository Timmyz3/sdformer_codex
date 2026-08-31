# M390：M389 Formality 独立打铁

结论：**92/100，P0/P1/P2 = 0/0/5**。M389 的 14 项 output manifest、
二层 seal、runner/TCL/contract、M387 r1b 与 M388 身份全部独立验证通过。
Formality 明确 `Verification SUCCEEDED`：

- `1963 = 793 port + 1170 DFF` passing compare points；
- failing / aborted / unverified / unmatched compare points 全为 0；
- BBPin / Loop / BBNet / Cut / LAT 全为 0；
- 227 条 SVF guidance 全接受，0 rejected/unsupported/unprocessed。

raw log 两次报告 375 个 power cell 缺少 FF/latch 的 power-down function。这是
无 UPF/CPF、always-powered 等价运行中的 technology-library low-power metadata
告警：reference/implementation 都成功链接，全部 port/DFF compare 通过，没有
black-box/cut/fail/abort/unverified/unmatched，因此不损伤本次功能等价。但它不能
支持 power-state、retention、isolation 或 PTPX 论断。

因此允许对 exact M387 r1b netlist/SDC 做明确标注为 prelayout 的 PT STA。
PTPX 继续 NO-GO：M384 r1b、M387 r1b 与 M389 中均无 SAIF/VCD/FSDB。physical
SRAM、physical timing、power/energy、system speedup、paper PPA 与 DATE headline
全部保持 false。`docs/359` 未修改。
