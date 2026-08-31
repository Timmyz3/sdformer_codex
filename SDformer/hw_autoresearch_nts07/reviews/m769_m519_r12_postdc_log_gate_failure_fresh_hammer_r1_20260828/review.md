# M769：M519 R12 K1 post-DC 日志门失败独立打铁

结论：**PASS failure audit，100/100，P0/P1/P2=0**。R12 唯一 attempt 已消费，整体必须永久保持 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`；K1 虽有完整的 raw setup/area 产物，也不能单独进入论文表或被改名为成功结果。允许作者新建 additive R13 source-only 包；本评审不授权任何 EDA。

## 封存与执行边界

- R12 quarantine 的 manifest/outer-seal 文件 SHA 分别为 `c9ebdd82...` / `7b4cc093...`，目录内 manifest 与 outer seal 均独立校验通过。
- R12 attempt receipt 的 manifest/outer-seal 文件 SHA 分别为 `918297a1...` / `6a3540aa...`，双封印通过；attempt 于 `2026-08-28T20:50:21+08:00` 在第一次 K1 DC launch 时消费。
- failure receipt 精确记录 runner rc=44、child rc=0、monitor rc=0、signal=none、runtime resource latch=0、exact term rc=0。`descendant_identity_faults.log` 和 `runtime_latch.txt` 均为空；preflight/runtime 外部碰撞表只有表头。
- 只存在 K1 与 K1 preflight。K8、K1x8 及其 preflight 均不存在；R12 canonical 与临时 work identity 不存在。
- `docs/359_DATE终局冻结_20260813.md` SHA 仍为 `dedde7ce...`。

## rc44 的唯一原因

R12 runner 第 1534--1535 行用 broad gate 拒绝任意 anchored `^Error:`/`^Fatal:`。K1 `dc.log` 中这类行总数精确为 1，且整个日志中不分大小写的 `error:`/`fatal:` 也只有这一行：

`Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl`

其后固定 stack block 明确是 `::env(HOME)` 不存在时，Design Vision GUI 辅助启动脚本访问 `.synopsys_dc_gui` 失败；16 行 block 的 SHA 为 `3f0791c8...`。该错误发生在 DC 初始化阶段、项目 Tcl 第一条命令之前。之后同一进程正常执行项目 Tcl 到 terminal、child rc=0 并打印 `Thank you...`。没有第二个 Error/Fatal，也没有 TIM-209/OPT-150。

因此 rc44 是 **runner 日志分类器对固定工具 bootstrap block 的假阳性**，不是 RTL、项目 Tcl、filelist、SDC、标准单元库、license、资源、碰撞或 DC compile 失败。这个判断仅适用于该 exact block；禁止把 broad grep 改成忽略所有 Error/Fatal。

## K1 raw 产物完整性

- Tcl terminal 存在且匹配：TIM-209=0、OPT-150=0、`compile_ultra_count=1`、incremental/hold optimization count=0。
- `compile_receipt.rpt` 记录单次 compile，wall=1534 s；日志中只有一条 `compile_ultra` 命令回显、一次 `Optimization Complete`、一次 Verilog 写出和一次 DDC 写出。
- 3.000 ns setup：QoR WNS/TNS=0.00/0.00、0 violating paths；详细 setup 报告的最差 slack 为 +0.0020 ns，100 条路径均 MET。
- max-delay、max-capacitance、max-transition、max-fanout 四份报告均为 no violated constraints；QoR 的 max transition/cap violation 均为 0。
- cell area=`124620.173180 um^2`，leaf/sequential=`153287/31160`，macro/black-box=0。hold 仍明确是 diagnostic-only，未在 DC 关闭，不能包装为 post-layout STA。
- mapped Verilog 23,014,689 B、DDC 6,610,944 B、mapped SDC 584,736 B、SVF 605,696 B，均已纳入 quarantine manifest。mapped top 为 `...ARCH_MODE0`。

这些事实证明 K1 的 DC Tcl 已完整结束；但它位于整体失败的 R12 quarantine 中，当前仍是 **内部恢复证据而非可引用 PPA**。在 K8/K1x8 缺失时，不允许计算等带宽面积、吞吐/mm²或三轴排名。

## 两种恢复路径

### A｜首选并授权 source-only：R13 全三轴重跑

新建 additive R13 runner/contract/candidate/release/canonical/attempt 身份，保持 R12 的 RTL、项目 Tcl、filelist、SDC、slow/fast DB、tool executable、license preflight、K1→K8→K1x8 顺序、每轴资源门与所有 PPA gates 不变。唯一功能修复是把日志门改为：

1. 允许且只允许启动阶段精确一次上述 `.synopsys_dv.tcl` + missing `::env(HOME)` fixed bootstrap block；固定首行、关键 stack token、出现区间和 occurrence count。
2. 除该 exact block 外，任意 `^Error:`、`^Fatal:`、TIM-209 或 OPT-150 仍 fail-closed。
3. 每一轴仍必须 child/monitor rc=0、resource latch=0、terminal/compile receipt/report/netlist/DDC 完整、setup/design-rule gate 全过。
4. R13 必须双封印绑定 R12 quarantine、R12 attempt、本 M769 review，并经 source/static hammer 与 final-release hammer 后才可获得一次 EDA attempt。

这是最小且最 sound 的路径。代价是重跑约 27 分钟 K1，但不会引入跨 attempt 拼接、选择性复用或复合 canonical 语义。

### B｜原则上可构造、当前不授权：sealed K1 + 新 K8/K1x8 组合

若未来必须节省 K1 重跑，可另建独立 split-axis/composite 协议；至少需要：

1. quarantine 保持原位不可变，以其 manifest/outer seal 精确绑定 K1 的日志、报告、netlist、DDC、child/monitor/resource/identity证据；不得复制后冒充原生成功目录。
2. fresh independent K1 component-admission 明确验证本评审的 exact bootstrap whitelist、Tcl terminal、单次 compile、3 ns setup、设计规则、面积、netlist/DDC 和 macro=0；在复合结果完成前仍不可引用。
3. 新 K8/K1x8 runner 必须与 K1 在 tool version/executable、slow+fast DB、RTL、Tcl、filelist、SDC、clock、compile policy、environment 与 closed contract keys 上逐 SHA 相等；仅 `ARCH_MODE` 合法不同。
4. K8、K1x8 各自必须在 launch 前重新通过 license、commit-headroom、MemAvailable、SwapFree、cgroup、same-UID EDA collision 门，并保存 child/monitor/runtime/final-gate完整回执；不能继承 K1 的资源通过状态。
5. 两个新轴必须采用同一个 exact bootstrap whitelist 和全部原 PPA pass gates；任何轴失败时不得发布 composite canonical。
6. 最终 composite manifest 必须同时绑定原 K1 quarantine 双封印和两个新轴双封印，显式标注跨 attempt 组合、无数值挑选/重综合 K1，并再做一次独立结果 hammer。

B 在证据论上可以成立，但实现/评审面明显大于重新跑一次 K1，且更容易被质疑为跨 attempt cherry-pick。因此 M769 **不授权 B，也不授权跳过 K1 的 EDA**；除非 A 因可复现外部约束无法执行，才应另立 source-only proposal 并重新打铁。

本评审没有运行 runner、DC、VCS、Formality、PT/PTPX 或 remote；没有修改 R12 quarantine、attempt、runner、RTL、Tcl、release 或 docs/359。
