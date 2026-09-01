# M1807｜C3 M1798 唯一生产失败与 M1806 X/Z 诊断独立审阅

结论：**能量准入 FAIL_CLOSED；P0=0、P1=1、P2=1。优先修复选 (a) 有界合法 reset-release settling TB，不选 (b) zero-delay 重跑。**

## 已核实事实

- M1798 attempt 已唯一消耗并双封；只有一次 VCS compile、一次 simv，`SAIF=0`、`PTPX=0`、无自动重试。
- canonical result 不存在；失败以双封 quarantine 原子隔离，raw build 留在 `private_build.unsealed_do_not_cite`。
- M1806 在 cycle 9 只看到六个 public debug counter 出现 X：
  `debug_config_beats`、`debug_tiles_loaded`、`debug_stage1_issues`、`debug_stage1_done`、`debug_product_pushes`、`debug_result_departures`。
- 同一个 cycle-9、`#0.1` 采样点，28 个 architectural/control output 全部 binary；另外五个 debug counter（raw beats、stage2 issues/done、product replacements、context cycles）也 binary。
- M1806 随后被继承的 aggregate fatal 在 cycle 9 终止，因此它只证明“首个 X 在哪里”，不证明这些 counter 在后续三个空闲周期是否会自行收敛。

## reset 与 delay 的判断

RTL 是 `always_ff @(posedge clk_core)` 内的同步高有效 reset。TB 让 reset 覆盖八个 posedge，并在 negedge 释放；到下一个 posedge 有半个 3 ns 周期，即 1.5 ns，不存在同沿释放 race。

M1454 mapped netlist 有 10,508 个 `DFKCNQD1`，其中 10,504 个 `CN` 接 tie-high，说明大多数同步 reset 已映入 D cone。当前 library 的相关 `DFKCNQ`/`DEL025` functional specify arc 是 `(0,0)`，netlist/library 中也没有消费 `UNIT_DELAY` 宏，且没有 SDF。因此当前 activity 仿真实际是 zero-delay functional gate simulation；单纯改成“zero-delay”不会构成因果修复。时序仍由独立 M1456 PT 的 setup/hold `+0.000299/+0.030474 ns`负责。

## 最小 additive successor

1. 新建 TB top，不覆盖 M1790/M1798，也不改 M1454 netlist/SDC。
2. 保留八个 reset posedge 与 negedge 释放。
3. 从第一个 post-reset posedge 起立即检查全部 28 个 architectural/control output 为 binary，并要求 quiescent 期间没有 accept/issue/retire。
4. 仅 debug counter 使用原 TB 已经预留的三个 quiescent post-release posedge；这三拍禁止 config/raw traffic。
5. 三拍结束时硬检查全部 11 个 debug counter 均 binary 且恰为 0；若仍有 X，立即失败。
6. 随后重新启用原完整 aggregate public-X/Z check，并保留 M1798 ordered tile-done tag scoreboard、结果/beat/tag/retire/counter/stall/conservation、warmup+8 measured tile、TX0/annotation 等全部门。
7. 禁止 `force`、`deposit`、`initreg`、`ignore-X`、删 public 检查或用 DUT case-equality 吞 X。
8. 先封 source-only successor，由不同作者静态打铁，再发新 release；M1807 本身不授权 EDA。

新的 activity 结果只能称 **zero-delay mapped functional component SAIF/PTPX**；不能称 timing simulation。M1456 时序证据不因本次功能失败失效，但 M1798 没有产生任何功耗或能量数字。
