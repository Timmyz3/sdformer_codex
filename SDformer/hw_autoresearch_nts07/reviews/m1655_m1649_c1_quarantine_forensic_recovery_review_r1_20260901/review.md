# M1655｜M1649 C1 quarantine forensic recovery 独立评审

日期：2026-09-01

状态：`PASS_M1655_M1649_C1_SEALED_QUARANTINE_FORENSIC__AUTHORIZE_SOURCE_ONLY_CANONICAL_RECOVERY__NO_EDA`

评分：98/100；P0=0，P1=0，P2=2。结论是：**DC 工具流实质完成，已封 quarantine 可做 forensic recovery；但当前目录仍不是 canonical，不可引用。**

## 为什么是可恢复，而不是 DC 失败

quarantine 的 39 个成员与 manifest/outer seal 全部逐字节通过，无 symlink、missing、extra 或 SHA mismatch。`dc.rc=0`，`TCL_INTERNAL_COMPLETE.txt` 完整，Verilog/DDC/SDC 已写出，SVF 已 `set_svf -off`，日志有正常 memory/CPU summary 和 `Thank you...` 退出。

日志仅有一条 `Error:`：第 32 行 DC Graphical 初始化 `dv.tcl` 时，`env -i` 没有 `HOME`。它出现在 `Current time` 和 M1630 Tcl 开始之前；进入实际 flow 后 Error/Fatal 数为 0。runner 在 DC 结束后用宽泛 `Error|Fatal` grep 扫描整份日志，因此自己返回 3 并封入 quarantine。这是 runner classification 假阴性，不是 Tcl/DC 中途失败。

## 已核数字

- post-restore setup WNS `+0.002221110 ns`，TNS 0，0 violating path。
- post-restore hold WNS `+0.000999451 ns`，TNS 0，0 violating path。
- area `152,898.625984 µm²`，相对 `147,246.392090 µm²` 基线增加 `3.838623%`，在 5% 门内。
- 9 个 `TS1N28HPCPHVTB128X128M4S` 宏，pre/post 数量一致，mapped Verilog 也精确出现 9 次。
- DRC violating nets = 0；输出 SDC 是 3.0 ns、0.200/0.050 ns uncertainty，0.051 ns 优化 guardband 没有泄漏，无 false/multicycle/path-specific/disabled-arc/case-analysis 异常。
- 输入仅为 original admitted M993/M1006 DDC，未使用失败 M1614 output；仅一次 hold-only incremental mapping。

## 恢复与后续门

可以从这份 sealed quarantine 恢复，且没有理由重跑 DC。正确步骤是先编写新命名 source-only recovery，exact 绑定本 tree 的 manifest/outer SHA 和 39 成员，只豁免已封日志中那一条精确的 pre-flow HOME GUI 签名，重算所有 timing/area/macro/DRC/SDC/artifact 门。它必须保留原 quarantine 不变，在 fresh recovered namespace 无覆盖发布，并生成新 receipt；本 review 没有做该复制/发布。

恢复后先做 M993 input 对 M1649 output 的 gate-to-gate Formality，再做 direct-RTL 链（或用已封 M993 RTL 等价 + 新 gate-to-gate 形成可审计传递链）。随后用精确 slow/fast 标准单元与 SRAM 宏库跑独立 PrimeTime max/min。DC 的 `PWR-428` 也意味着能量/PPA 表还需独立 power flow，不能从本结果推导。

## 边界

本评审只授权撰写 source-only canonical-recovery；未解封、未复制、未发布 canonical，未启动 DC/Formality/PT/PTPX。当前 quarantine 仍 `paper_citable=false`，不得写为 Formality/PT/power/PPA 或 headline 结果。
