# M456 对 M448R4 PTPX 的独立打铁结论

评分 **92/100**，P0=0、P1=1、P2=2。R4 的 manifest、活动率、三点功耗、能量换算和 input-slew sensitivity 均独立闭合，可准入为 **M416 selected-slice、prelayout standard-cell、TT 0.9 V/25 C、3 ns ideal clock、ZeroWireload、无 SPEF、0 macro** 的窄口径功耗证据。它不是 paper-PPA-ready，也不能扩展为 SRAM、CTS、完整 Conv、全网或系统能量。

## R1–R4 身份

- R1：外层 exit 22，封存 marker 明确 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`；原 `check_power` 有 4,139 个 out-of-range ramp。
- R2：外层 exit 25；错误地期待 `ptpx.log` 出现 3 条 `update_power` 源码回显，实际只有 procedure 定义的 1 条。保持 `DO_NOT_CITE`。
- R3：manifest 只有一行，target 为 stdin `-`，内容只是空输入 SHA；seal 因而是 vacuous。保持 `FAILED_INVALID_VACUOUS_SEAL_DO_NOT_CITE`。
- R4：44 项全部为 `./` 相对路径，dash/duplicate/work/hash mismatch/missing required 均为 0；manifest 与外层 seal 在 R4 cwd 自验证 rc=0。审计前后 R4 47 个 regular file 的 SHA/大小快照完全相同。

没有从 R1/R2/R3 复制任何数值。

## 功耗与活动率

| input slew | internal (mW) | net switching (mW) | leakage (mW) | total (mW) |
|---:|---:|---:|---:|---:|
| 50 ps（sensitivity） | 5.59262276 | 0.626869917 | 0.0342430696 | 6.25373602 |
| 100 ps（primary） | 5.59269476 | 0.626869917 | 0.0342430696 | 6.25380802 |
| 200 ps（sensitivity） | 5.59313822 | 0.626869917 | 0.0342430696 | 6.25425100 |

三份 raw `report_power` 均唯一解析到 internal/switching/leakage/total，单位为 mW；三份 `check_power` 均成功，Warning/ramp/missing-table/missing-function finding 全为 0。9 行 runtime ledger 的三组 `check_power → update_power → report_power` 顺序完全一致，`pt_shell` rc=0。

raw SAIF 独立重数得到 22,800 项、非零 toggle 21,827 项、TX 非零 0 项、TX 总时长 0；精确覆盖率为 **95.73245614035088%**。PrimeTime annotation 为 nets 22,800/22,800、leaf cells 20,803/20,803。测量窗口为 6,288,008.5 ns、2,096,003 measured cycles，因此 100 ps 主点：

`6.25380802 mW × 6,288,008.5 ns / 2,096,003 = 18.76142256815862 pJ/measured-cycle`。

50/200 ps 相对 100 ps 的最大 total-power 偏差为 **0.0070833642251688644%**。

## 必须保留的边界

P1 是 reset：`check_timing` 明确报告唯一没有 clock-relative input delay 的端口为 `reset_n`；SDC 对它做 false path，SAIF 测量期间它静态为高。100 ps slew 只修复功耗查表 ramp，不是 recovery/removal、minimum-pulse 或 reset signoff。

两个 P2：PrimeTime 对 SDC 文件 2.1 与 requested 2.2 发出两次 SDC-2 warning；在 paper timing signoff 前应统一版本重跑。另一个是 `clock_network=4.58843946 mW` 带 `i` 属性，包含寄存器 clock-pin internal power，但在 ideal clock/no CTS/no SPEF 下不包含真实 clock-tree buffer 与提取互连，不能称为 post-CTS clock power。

允许引用：100 ps 主点的 selected-slice prelayout standard-cell power/energy，以及 50/200 ps sensitivity。禁止引用：R1/R2/R3 数字、reset signoff、CTS/SRAM/macro/full-Conv/full-network/system energy 或 speedup、paper-PPA-ready/headline。

`docs/359` 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

