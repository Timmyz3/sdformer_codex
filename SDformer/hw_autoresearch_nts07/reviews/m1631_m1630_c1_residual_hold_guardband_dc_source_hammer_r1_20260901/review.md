# M1631｜M1630 C1 residual-hold guardband DC source 不同作者审阅

日期：2026-09-01

状态：`PASS_M1631_M1630_C1_RESIDUAL_HOLD_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_DC_ATTEMPT`

评分：98/100；P0=0，P1=0，P2=1。

## 裁决

M1630 source 包通过不同作者 compile-free 静态与变异审阅。允许作者另行编写并封存 M1632 release；只有该 release 精确绑定 M1630 runner、source contract 与本 review，调用者再同时 pin runner/release SHA 后，才允许一次未来 DC attempt。本审阅没有运行 DC/VCS/PT/Formality/PTPX，也没有创建 attempt、result 或 release。

唯一 mapped 输入仍是原始 admitted M993/M1006 DDC。M1614 输出未被读取；其 sealed negative 只用于证明上一轮在 setup、DRC、面积均过门后仍剩 `hold WNS=-0.000353523 ns / TNS=-0.000401557 ns / 3 paths`。

Tcl 只把 hold uncertainty 从最终报告点 0.050 ns 暂时提高到 0.051 ns，执行唯一一次 `set_fix_hold` 和唯一一次 `compile -incremental_mapping -only_hold_time`，随后立即恢复 0.050 ns，再生成全部最终报告与输出。不存在第二轮 compile、generic optimizer、`compile_ultra`、频率放宽、timing exception、disabled arc 或 case analysis。

物理合同保持 3.000 ns、setup uncertainty 0.200 ns、最终 hold uncertainty 0.050 ns、ideal clock、ZeroWireload、standard/macro slow-fast min-library 配对以及九个 dont-touch SRAM 宏。成功状态必须同时满足 setup/hold `WNS>=0, TNS=0, violations=0`、面积不超过 154608.7116945 um²、宏数前后均为 9、DRC violating nets 为 0；否则只封存 negative 且不得 retry。

## Mutation hammer

CPython 3.6 与 3.10 各自重跑相同冻结锤，JSON byte-identical；95/95 攻击全部被拒。攻击覆盖原始 DDC/失败 DDC 边界、current-design 对象名、guardband/恢复顺序、optimizer 次数、六类路径隐藏、3 ns/两项 uncertainty/ideal clock、九宏/dont-touch/min-library/ZeroWireload、review-release-attempt-tool 顺序、caller pins、same-UID collision、clean DC flags、HOME、其他 EDA、Error/Fatal/LINK/loop，以及 setup/hold/area/macro/DRC 正结果谓词。

原作者 15 项测试也在 CPython 3.6 和 3.10 下各 15/15 PASS；runner `bash -n` PASS。M993、original quarantine、M1006、M1614 sealed negative、author handoff 与 contract 的内外封印全部复核通过，`docs/359` SHA 仍为 `dedde7ce...`。

## P2

runner 在早期 freshness、独占锁和已消费 attempt 后封存 work tree 并 `mv` 发布，但自身不在发布后重新枚举 canonical exact topology。它不阻塞一次未来运行；mandatory different-author result hammer 必须校验 canonical topology、全部 seal、输出 SDC 命令数以及 receipt/report 交叉字段后才可准入。

## 边界

本审阅只授权 M1632 release authoring，不直接授权 DC，也不证明 hold closure、setup/hold timing、面积结果、Formality、PrimeTime、power、energy、speedup 或 paper-ready PPA。即使未来 DC 是 positive，仍需不同作者 result hammer、M993 gate-to-gate Formality、direct-RTL Formality 与独立 PT slow/max + fast/min。
