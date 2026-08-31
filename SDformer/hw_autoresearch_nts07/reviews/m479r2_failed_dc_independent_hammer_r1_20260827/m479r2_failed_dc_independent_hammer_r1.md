# M479r2 失败 DC 独立打铁评审（receipt-blind）

日期：2026-08-27  
对象：`dc_handoff/runs/m479r2_lane_local_dc_3p000ns_r1_20260827`  
方式：只读 raw DC log、报告、mapped netlist、RTL、runner/Tcl/SDC 和输入 SHA；没有读取或依赖任何 M479r2 结果 receipt（该失败 run 也没有生成成功 receipt）。

## 裁决

**65/100，NO-GO。M479r2 当前 RTL 不得准入 logic-only DC/STA，也不得进入 Formality/PTPX。**

失败不是身份污染，也不是 setup/hold；失败原因是三类设计规则仍不干净。更关键的是，所谓 lane-local `keep` 在综合后完全消失，M479r2 mapped netlist 与 M477 除日期和模块名外逐字相同。这次改动对物理结构的效果严格为零。

| 评分项 | 分数 | 独立结果 |
|---|---:|---|
| 工具、输入与 top 身份 | 20/20 | exact-SHA 全部回验；top 是 M479r2，不是 M476r2 |
| 可复核性与 raw 证据 | 5/5 | DC 正常退出、报告和 netlist 齐全；runner 因 fail-closed 门退出 33 |
| 面积门 | 15/15 | 42,370.649130 µm²，低于 44,779.2 µm² 门 |
| setup/hold | 15/15 | setup +0.0000 ns、hold +0.0101 ns，均 MET |
| 五类约束 | 10/25 | max-delay/min-delay clean；cap/transition/fanout 失败 |
| lane-local 结构存活 | 0/15 | mapped netlist 与 M477 规范化后完全相同 |
| 合计 | **65/100** | 任一 DRC 失败即二元 NO-GO |

## 1. 身份与输入回验

- 工具：Synopsys Design Compiler V-2023.12-SP3，二进制 SHA `23a4101...`。
- slow/fast library SHA 与 run 的 `input_sha256.txt` 一致；slow corner 是 TSMC28 HPC+ `ssg0p9v125c`。
- SDC 是 3.000 ns、setup uncertainty 0.200 ns、最终 hold uncertainty 0.090 ns、ideal clock、ZeroWireload、`set_max_fanout 32`。
- RTL、filelist、SDC、Tcl、contract、上游 hammer、M477 失败证据和 docs/359 的所有列举 SHA 都对当前磁盘文件回验通过。
- `area.rpt`、DC log 和 mapped netlist 的设计身份均为 `m479_lane_local_backpressure_safe_parent_queue_pipeline`；不存在旧 top 被 elaborated 的迹象。
- `dc.rc=0` 且 log 有 `Thank you...`；`RUN_FAILED_OR_INCOMPLETE.txt` 中 runner exit 33 正好对应“5 个 constraint section 必须全部 clean”的 fail-closed 门。不是工具崩溃或超时。

因此这次 run 的失败数值是有效诊断，只是不能作为通过结果引用。

## 2. 独立抽取的物理结果

| 项目 | M479r2 raw DC | M477 raw DC | 判定 |
|---|---:|---:|---|
| Cell area | 42,370.649130 µm² | 42,370.649130 µm² | 完全相同 |
| Cells | 41,849 | 41,849 | 完全相同 |
| Comb / Seq | 36,340 / 5,508 | 36,340 / 5,508 | 完全相同 |
| Ports / Nets | 14,329 / 57,120 | 14,329 / 57,120 | 完全相同 |
| Logic levels | 55 | 55 | 完全相同 |
| Setup worst slack | +0.0000 ns | +0.0000 ns | MET，完全相同 |
| Hold worst slack | +0.0101 ns | +0.0101 ns | MET，完全相同 |

五类合同约束的准确结果：

1. max-delay：PASS，无 violation。
2. min-delay：PASS，无 violation。
3. max-capacitance：FAIL。`u_core/n17470` required 0.0446、actual 0.0776、slack -0.0330；`u_core/n1` actual 0.0479、slack -0.0033。
4. max-transition：FAIL。`u_core/n17470` required 0.5280、actual 0.7441、slack -0.2161；报告列出的负载主要是 `row_acc_q/psum_acc_q` 的 enable pin。
5. max-fanout：FAIL。`u_core/n17470` 80（slack -48）、`u_core/n16011` 61（-29）、`u_core/n1` 57（-25），合同上限 32。

`qor.rpt` 的 summary 与 raw constraint report 一致：3 个 violating nets、1 个 transition net、2 个 capacitance nets、3 个 fanout nets。这里的“1/2/3”是 violating net 计数，不应把 transition report 展开的许多 load pin误写成许多独立 violating nets。

## 3. `keep` 为什么没有形成 lane-local tree

RTL 在一个外层 `if (issue_accept_w)` 内，又用

```systemverilog
row_lane_enable_w = {LANES{issue_accept_w}};
psum_lane_enable_w = {LANES{issue_accept_w}};
if (row_lane_enable_w[lane]) ...
if (psum_lane_enable_w[lane]) ...
```

控制同一批更新。每个“lane-local”bit 与外层条件在布尔上完全相同，内层条件是冗余条件。`(* keep = "true" *)` 只贴在复制 wire 上，既没有创造独立 driver，也没有禁止 DC 做跨 bit 的等价合并。

三条相互独立的物理证据确认它被合并：

- DC 第三轮 compile 的 flow information 明确为 `Dont Touch Cells = 0`、`Dont Touch Nets = 0`。
- mapped netlist 中没有 `row_lane_enable`、`psum_lane_enable` 或可审计的 8/12-lane group hierarchy；enable pin仍由共享的 `n17470/n16011/n1` 等根驱动。
- 对 M477 和 M479r2 mapped netlist 去除生成注释/空白并规范化两个模块名后，Verilog token stream 完全相等；规范化 SHA256 为 `30ee5943d94c9ecab522b01020dc97e40d07bcb7b404edc7dbbdf162422d7023`。

这比“面积碰巧相等”更强：两次综合输出的逻辑与映射实例逐字一致。

## 4. 是否允许最后一次“8/12-lane 分段寄存 enable tree”

### 4.1 真正注册一级 enable：NO-GO，不值得做

直接把每 8 或 12 lane 的 accept 复制寄存一拍，会把状态提交推迟一拍；ready/valid 协议允许源在下一拍更换 `issue_*`，所以 enable 与组合 D 数据会错配。为了保持功能，必须同时寄存每 lane 的 13-bit `row_partial` 和 20-bit `psum_final`，至少增加 `96 × (13+20) = 3,168` bit 的 payload staging，再处理 metadata、final/first、fault 和 backpressure。

这相当于在现有 5,508 个 sequential cells 上至少再加 57.5% 的状态。按本 run 非组合面积/seq 的粗下界约 2.35 µm²/FF，光 3,168-bit staging 就约 7.44k µm²，远高于面积门只剩的 2.41k µm²余量；它还会改变 latency/协议和 M473 的 cycle 合同。半周期/负边沿寄存也会引入新的时序与 CDC 风格风险，不能当作同一设计的 DRC 修复。

因此：**不允许生产主线再做 registered-enable pipeline。**

### 4.2 唯一可允许的最后一试：受保护的组合 leaf-buffer tree

如果主线仍决定花最后一次短实验，允许的只能是**零周期、零协议变化**的物理实现修复：

- 选定一个粒度，不同时扫两个：建议 8 groups × 12 lanes；每个 group 覆盖 12 lane 的 row+psum accumulator enable，理论叶端负载约 `12 × (13+20) = 396` 个 enable pins，仍需在 group 内有多级 cell buffering；“一根 group wire”本身不能满足 fanout 32。
- 用明确的可综合 1-bit identity-buffer hierarchy或技术 buffer cell，并在 DC 中对 leaf/branch cell 执行可审计的 `dont_touch`/size-only 约束；仅在 RTL 上再贴 `keep` 不算实现。
- 不能注册 payload、不能增加架构周期、不能放宽 3 ns/uncertainty/fanout/cap/transition、不能改 external scratch/psum cut。
- 运行前用 elaborated/precompile 报告证明 8 个 group roots 和所需 branch cells存在，运行后 mapped netlist必须保留这些实例；若仍 `Dont Touch Cells/Nets = 0`，立即 fail-fast，不能再等两小时 DC。

这是一项 P&R/DRC hygiene，不是论文创新点；即使成功也只使 M473/M479 物理诊断更可信，不产生新 speedup。

## 5. 唯一 recovery 的 P0 门

若做上述组合树，必须一次同时通过以下门；任一失败永久关闭 M479 Conv 分支，不得继续 M480 式变体：

1. **结构门**：precompile 和 mapped netlist 均能审计到唯一预注册的 8-group × 必要 branch buffer tree；规范化 netlist不得再等于 M477。
2. **功能门**：精确 SHA 的全回归与 targeted Synopsys VCS/SVA 全过，tuple/numeric/protocol/stall/RAW coverage 不退化；之后 Formality RTL↔mapped 必过。
3. **时序门**：setup、hold slack 均非负，且不得用 uncertainty/clock/IO/false-path 放宽换通过。
4. **五约束门**：max-delay、min-delay、max-capacitance、max-transition、max-fanout 五段全部明确 `no violated constraints`。
5. **面积门**：cell area ≤44,779.2 µm²；macro count仍为 0，并继续标注 pre-macro logic-only。
6. **停止门**：若面积超门、任一 DRC 未清、Formality 不过、或树再次被合并，则永久 NO-GO；不允许第二种 12-group × 8-lane 变体。

## 6. 论文边界

- 当前 42,370.649130 µm² 和时序只能作为失败消融/工程诊断，不得进入通过的 PPA 表。
- 该 run 没有 power、energy、macro、physical clock、P&R 或 system speedup；不能外推任何 headline。
- 即使组合树修复成功，它也不改变 M473 的 fused/unfused 周期口径，不得与 M472、C2 倍率相乘。
- 当前正确路线是让 M479 失败结论尽快收口，释放 Synopsys 资源给 FC2 三点 matched-shell PPA，而不是再造 registered pipeline。
