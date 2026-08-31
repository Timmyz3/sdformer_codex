# M1121：M1112r3 C2 DC 调用失败独立审计

裁决：**M1112r3 的唯一 attempt 已消费并正确隔离，永久禁止重试。根因唯一，是 `snps_shell` 启动选择器用错；并非 RTL、Tcl、许可或资源失败。这个 attempt 没有形成任何 DC、网表、mapped VCS、面积或时序证据。**

## 冻结状态与失败边界

本审计只读检查原 attempt 和 quarantine，没有调用 M1112r3 launcher/engine，也没有运行任何 DC Tcl/设计流、VCS 或 simv；仅做了不读取项目输入的 wrapper `-help` 只读探针。两个原目录的 manifest 和 outer seal 均逐字节重算通过：

- attempt 只有 `attempt.json` 一个 manifest 成员，状态为 `M1112R3_ATTEMPT_CONSUMED_AFTER_M1117R3_M1118R3`，记录 `dc_attempts=1`；
- quarantine 只有 `dc/dc.log` 和 `failure.json` 两个 manifest 成员，无符号链接；
- `failure.json` 状态为 `FAILED_DIAGNOSTIC_DO_NOT_CITE`，失败阶段是 `FRESH_DC_M1112R3`；
- canonical result 和残留 work namespace 均不存在；
- `dc.log` 恰好 37 byte、1 行，SHA256 为 `db8e7da6...`：`Error: The  script is not supported.`

该日志与先前 M522 r3 已独立审计的同类调用错误逐字节、逐哈希相同。attempt 与 quarantine 从创建到封存仅约 33 ms，且目录中没有 Tcl terminal、reports、netlist、DDC、SVF、mapped compile 或 simv 产物。因此这里的“DC failed”只表示启动 wrapper 返回非零，不能解释成 Design Compiler 已执行后失败。

## 根因：解析目标可冻结，但不能用解析目标的 basename 启动

M1112r3 engine 第 47 行把 `DC_TARGET` 固定为普通文件：

```text
/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell
```

第 569 行执行：

```text
snps_shell -f run_dc_m519_r8_setup_area_three_axis.tcl
```

本机安装的 `snps_shell` 是按启动路径 basename 分派产品的 POSIX wrapper，不是可忽略 `argv[0]` 的 DC 二进制。wrapper 第 11 行将 `script_name` 置空，只有符号链接解引用循环的第 33 行才把原始启动名写入；第 191–200 行只在该名字是 `dc_shell` 时构造 `common_shell_exec -shell dc_shell -r <install-root>`；否则第 398–400 行输出本次精确错误并退出。

因此，直接用普通文件名 `snps_shell` 启动时根本没有选中 DC backend。Tcl 虽出现在命令行中，但没有被 DC source；filelist、RTL、库、elaboration、compile 和 mapping 都未开始，后续 mapped VCS 更未开始。

## 为什么不是许可、资源或 RTL/Tcl

engine 的顺序是 collision gate、resource gate、license gate 全部通过后才创建 ATTEMPT，随后才进入 `FRESH_DC_M1112R3` 并启动 wrapper。ATTEMPT 已存在，证明这些前置门均已通过。错误又发生在 backend 分派之前，因此许可/资源不足、RTL 语法、Tcl、库以及 mapped-reset 逻辑均不是这次的失败原因。这个失败也不能证明那些后续阶段成功；它们根本没有被测试。

## 本机成功协议与唯一最小修复

本机已有三组正向证据使用以下入口并成功：M522 r4、M872 三轴和 M917 Fixed 都执行

```text
/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell -f <exact Tcl>
```

其中 `dc_shell` 是 raw link text 为 `snps_shell` 的符号链接，解析目标 SHA256 为 `23a4101c...`。M872/M917 的运行时捕获进一步证明，成功的真实子进程是 SHA256 `bf91e6ab...` 的 `common_shell_exec`，argv 精确包含：

```text
-shell dc_shell -r /opt/synopsys/syn/V-2023.12-SP3 -f <exact Tcl>
```

所以 additive r4 必须选择已经在本机成功过的协议：**用精确 `dc_shell` 符号链接路径执行 `-f`，同时冻结 link 类型、raw link text、resolved target 路径及 SHA，并捕获/核对真实 backend argv。**

不能把 `snps_shell -shell dc_shell -f` 当作本机已验证修复：安装 wrapper 在转发用户参数之前先按 basename 选择 case；`-shell` 参数不会补写空的 `script_name`。若未来想直接调用真实 backend，必须另做官方协议和本机成功证据；本次 r4 不应扩大改动面。

## r4 authoring 合同

r4 必须使用全新的 engine/contract/launcher/attempt/result/work/failure/lock namespace；M1112r3 永久 `DO_NOT_RETRY`。唯一语义变化只能是 DC invocation selector 及其身份/运行时核对。RTL、TB、filelist、SDC、DC Tcl、双库、异步 observation/reset-provenance 行为、VCS 命令、128-cycle 窗口和 claim boundary 必须保持原身份或被精确重绑定。

新 source 必须先经过不同作者打铁，再由不同作者封 zero-argument launcher 并打铁；之后最多一次 fresh attempt。即使成功，仍需独立 result hammer 才能准入 mapped 功能。不得复用 M1112r3 attempt，也不得把本失败目录中的任何内容写成面积、时序、功耗或性能数字。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
