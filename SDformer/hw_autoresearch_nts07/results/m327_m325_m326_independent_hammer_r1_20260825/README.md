# M327：M325 Formality 与 M326 PrimeTime 独立打铁评审

结论：`91/100`，`P0=1、P1=1、P2=3`。M321 RTL 对 M322 映射网表的**功能等价 GO**；3 ns 的 setup+hold 时序包因 fast-corner hold 失败而 **NO-GO**。本评审只读重放哈希与解析已有报告，未调用新思工具，未修改 RTL、合同或 `docs/359`。

## Formality：功能等价 GO

M325 的 exact-SHA 输入与 snapshot/output manifest 均重放通过。Formality `verify_return=1`、进程返回码 0，205/205 compare point 通过，其中 port 49、DFF 156；failing=0、aborted=0、unverified=0、reference/implementation unmatched=0。352 条 SVF guidance 全部 accepted，rejected/unsupported/unprocessed 均为 0。

这个 GO 只表示“exact M321 RTL ↔ exact M322 mapped netlist”。它不证明 M321 与 M311 的单周期 latency 合同等价，也不证明时序、SRAM、功耗、完整 Conv、accuracy 或系统性能。

## PrimeTime：setup GO、hold NO-GO

M326 使用 3.000 ns ideal clock、setup `ssg0p9v125c`、hold `ffg1p05vm40c` min library、ZeroWireload、无 SPEF、零 macro：

- setup：156/156 met，0 violated，最差 slack `+1.1047 ns`；仅可作为 prelayout setup 证据。
- hold：74/156 met，82/156 violated，最差 slack `-0.0071 ns`；global report 的 TNS 为 `-0.20 ns`。82 个违反端点均是 reg-to-reg，覆盖 52.6% 的寄存器 hold endpoint。
- `ptsta_timing_hold.rpt` 只展开 28 条违反路径，是 `-max_paths 100 -nworst 10` 的报告截面；权威人口是 coverage/global/constraint reports 一致给出的 82/156。constraint report 中正好有 82 个唯一违反 endpoint。

PrimeTime 返回码 0 只表示工具执行完成。M326 合同要求 hold 非负，因此 3 ns 时序包必须维持 fail-closed `NO-GO`；不能拿 DC slow-corner 的 `+0.0001 ns` hold 代替 fast-min corner 的结果。

## Reset 例外

`reset_n` 是唯一没有 clock-relative input delay 的端口，并在加载 mapped SDC 后执行 `set_false_path -from [get_ports reset_n]`。同步 setup/hold 检查没有 untested endpoint，但 156 个 recovery 与 156 个 removal 检查全部因 reset 没有 startpoint clock 而 untested。因此当前结论不包含 reset recovery/removal signoff。

现有 exact-SHA TCL 和 raw log 足以证明该 false path 被应用；但 `ptsta_exceptions.rpt` 使用 `-ignored`，没有枚举 active exception。下一轮应输出 active exception 对象及数量，并拒绝 reset_n 之外的 false/multicycle exception。

## Hold-fix 最小再准入条件

1. 新建、不覆盖原证据的 fixed/ECO netlist 与 SDC，封存新 SHA；若 RTL 变化，先重跑 VCS 并保持两拍 latency 与 ready-ready II=1。
2. 对 fixed netlist 重跑 Formality：`verify_return=1`，全部 compare point passing，failing/aborted/unverified/unmatched 全为 0。
3. 用同一 3.000 ns、同一 slow/max 与 fast/min library、同一 IO 假设和唯一 reset_n 例外重跑 PT；不得靠新增 false path/multicycle waiver 消除违规。
4. setup WNS 至少保持 `+0.5000 ns`（保留 M322 architecture gate），hold WNS `>=0`、TNS `=0`、156 个 hold endpoint 中 violation `=0`，同步 setup/hold 均 tested。
5. 报告 hold-fix 后面积、cell 与 delay-cell 增量；重新封存并重放 netlist/SDC/library/script/Formality/PT/receipt 全证据。

完成以上条件后，只能准入 prelayout/no-SPEF/ideal-clock/zero-macro 的 3 ns logic timing。post-route、propagated clock、真实寄生、paper PPA、系统加速比和 headline 仍需后续独立证据。

## 缺陷分级

- P0：82/156 fast-corner hold endpoint 违反，WNS `-0.0071 ns`，阻塞 3 ns 时序准入。
- P1：reset recovery/removal 被排除；物理 signoff 前必须定义 reset release 并补 recovery/removal 分析。
- P2：M326 receipt 没写 74/82/156 完整人口；active reset exception 未被报告枚举；SDC 2.1/2.2、Formality power-cell/hier-map 警告需闭合或书面 waiver。

`SHA256SUMS` 同时绑定本评审、M321/M322 exact 输入、M325/M326 合同/脚本/manifest 与关键报告；`SHA256SUMS.seal.sha256` 再封存该 manifest。
