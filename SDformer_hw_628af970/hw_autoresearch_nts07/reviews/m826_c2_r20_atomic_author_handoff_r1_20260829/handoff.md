# M826/C2 R20 final-authorization closure source handoff

M826 是 M823 唯一 P1 的 additive 修复。M803 RTL/SVA/TB/filelists、五档 exact 周期和 M822 已闭合的 attempt publication/receipt 逻辑均保持冻结。当前仍是 source-only package，不授权 VCS、simv、license 查询、正式 attempt/result、true release 或任何 EDA。

## 唯一修复

future final-hammer 的 `authorization` 现在必须与 15 键闭合集合在键名、值和 Python 类型上完全相等：只允许一次 VCS、一次 simv 和必要的 license query；DC/Formality/PT/PTPX、Icarus、Verilator、CPU/GPU、remote/network 全部显式为 false/0。额外键、缺键、`true`/`1` 或 `false`/`0` 类型混淆都拒绝。

M823 的五类攻击均已变成负例：`run_vcs=false`、`run_simv=false`、`query_license=false`、`max_attempts=0` 与 extra key。另对 15 个缺键和 7 个类型混淆逐一验证。合法 exact authorization 仍能通过合成 future chain。

## 保留验证

- Python 3.6.8 与主机 modern Python 3.12.13：原 12/12 atomic tests、新 8/8 final-authorization tests、closure 和 dry-run 全过。主机没有 Python 3.10 可执行文件。
- 四类实际 CLI failure receipt 仍为 `false/false/true/true`；pre-existing exact collision 两侧不被覆盖，postrename damaged 仍保守 consumed。
- wrong-SHA rc=3；source dry-run rc=86，并在 live VCS/license boundary 前停止，所有正式副作用为 0。
- 40 个 contract source SHA 和三组双封验证通过；`docs/359` 保持 `dedde7ce...`。

下一步只允许另一位 receipt-blind reviewer 完成 M827 source hammer。即使 M827 PASS100，也只能授权另行编写 true release 与 final-hammer request，不能直接运行 VCS。
