# M928｜M917/M518 r5 Fixed-T10 DC 独立结果打铁

## 裁决

**PASS 99/100，P0=0、P1=0、P2=1**：`PASS_M928_M917_M518_R5_FIXED_LOGIC_ONLY_DC_RESULT_ADMITTED`。

M917 canonical result、唯一 consumed attempt 与嵌套 preflight seal 均通过全量重算；结果身份、runner/contract/admission/RTL/tool/库/约束/docs359 哈希一致。`dc.rc=0`、`runtime_monitor.rc=0`、Tcl terminal 与 `RUN_COMPLETE` 完整，未发现 runner 自碰撞、孤儿写入、数值报告污染或第二次 attempt。

唯一 P2 是元数据来源措辞：`structured_postcompile_gate.rpt` 写了 `source_declaration_tuple_count_authority=runner_prelaunch_parser`，但 M917 runner 没有执行该 parser。M928 对同一冻结 RTL 独立解析出 **50 个 source declaration tuple / 1175 个展开 bit**，且 DC 实测也是 **1175 bit-level ports**，所以结果计数闭合；但论文和后续收口不得引用那条错误的 authority 字符串，应引用 M928 的独立复核。

## 可引用的 DC 点

| 项 | M917/M518 r5 Fixed-T10 |
|---|---:|
| 工艺/约束 | TSMC 28 nm，3.000 ns ideal clock，ZeroWireload |
| cell area | **62,433.503388 µm²** |
| cells | **71,898** |
| combinational / sequential cells | **61,325 / 10,573** |
| macro / black box | **0** |
| 100 条 setup path 最差 slack | **+0.0003 ns MET** |
| QoR setup WNS / TNS / violating paths | **0.00 / 0.00 / 0** |
| compile_ultra / incremental / hold optimization | **1 / 0 / 0** |

五个 setup/design-rule 门均闭合：100-path setup report 无 violated path，max-delay、max-capacitance、max-transition、max-fanout 四份 constraint report 均为 no violation；`check_design=1`、`check_timing=1`，并执行了 unconstrained-endpoint 检查。

## hold 与 claim 红线

QoR 的 hold 仅为诊断：worst violation 约 **−0.02 ns**、total violation **−58.19 ns**、**9,741** 条 violation。本次没有 hold optimization，也没有独立 PT/STA。因此可引用范围严格限定为：

- Fixed-T10 component 的 logic-only、pre-macro DC setup/area；
- 3.000 ns ideal-clock/ZeroWireload 条件下 setup MET；
- 面积、cell、sequential-cell 和无 macro/black-box 数字。

不得写成 hold closed、STA completed、macro-inclusive PPA、功耗、能量、吞吐、加速比、全网或系统 headline。该结果本身没有性能分母，也没有 rank-3 对照。

## 运行时与日志审计

- preflight 3 样本、runtime 含 final 共 34 样本；runtime 最低 commit headroom / MemAvailable / SwapFree 分别为 `117,719,968 / 413,454,520 / 54,219,772 KiB`。
- 全部样本 cgroup fail/under-oom/oom-kill 为 0，external collision 为 none；33 个在线 gate 均为 none，final ACK 明确 `job_tree_empty_before_ack=true`。
- DC root 的 PID=pgrp=session，private HOME 在 canonical 中保持 0700；未出现 r4 的 descendant self-collision。
- `dc.log` 没有 Error/Fatal。925 条 Warning 只落在冻结的 LINT/VER/TIM/UISN/UID 类；裸搜到的 `TIM-209`/`OPT-150` 字符串来自 Tcl 命令回显，结构化 precompile gate 对真实输入报告计数均为 0。
- mapped netlist 只有一个 module，没有 DW/black-box 残留。

机器可重放入口为 `independent_hammer.py`；详细结构化结果见 `review.json`。
