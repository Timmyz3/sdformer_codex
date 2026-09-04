# M496 r3 resource-gated exact replay 独立预检 r2

日期：2026-08-27  
审阅范围：只读静态检查修订后的 M496 r3 contract、exact runner、冻结 TCL、r2 OOM 独立审阅及其内层 manifest、r2 冻结输入身份。未启动 DC、VCS、Formality、PT 或 CPU DSE；未修改任何生产文件或 `docs/359`。

## 裁决

**`STATIC_GO__ONE_EXACT_REPLAY_ONLY_WHEN_ALL_RESOURCE_GATES_PASS`（95/100）**。

r1 的两个 HIGH 阻塞项均已实质关闭：H1 现在于每个 point 的三个资源 sample 以及 launch 前最后一刻统一拒绝 DC/FM/PT/VCS/项目 M-series CPU-DSE；H2 现在锁死 canonical r3 run path，并用 `mkdir` 原子创建合同级 attempt marker，在第一次 K1 `dc_shell` launch 边界消耗唯一 replay。r2 review 的内层 `SHA256SUMS`、完整 `/proc/meminfo` 和冻结 compile body 也已闭合。

这个 GO 是**静态执行器准入**，不是“现在可以启动”。审阅时环境快照只有 `15,425,728 KiB` commit headroom，低于 `67,108,864 KiB` 门限，同时存在无关用户的 `simv`；因此当前状态会被 runner 正确拒绝。只有 runner 自己的三次、10 秒间隔检查全部通过，并在 launch 前再次无冲突时，才准许一次 exact replay。

## r1 阻塞项复核

| r1 问题 | 修订后证据 | 裁决 |
|---|---|---|
| H1：逐点漏检另一个 DC | `m485_forbidden_process_gate()` 同时匹配 `dc_shell`、`dc_shell-t`、DC `common_shell_exec`；该函数在每个 sample、`m485_resource_gate()` 返回前、point launch 前再次调用 | **CLOSED** |
| H2：可换目录重试 | `M496_DC_RUN` 明确禁止；run 固定为 `...r3_20260827`；run 目录与 attempt 目录均通过原子 `mkdir` 防并发/重放 | **CLOSED** |
| M1：r2 review 只验外层 seal | runner 先锁 seal 文件 SHA，再在 review 目录执行 `sha256sum -c SHA256SUMS` 和 seal 校验 | **CLOSED** |
| M2：CPU-DSE matcher 过窄 | contract 明确把 denylist 收口为 basename 带 `analyze/independent/sweep/dse/simulate_m<ID>` 的项目 Python entrypoint；代表性命令静态匹配通过 | **CLOSED TO CONTRACT SCOPE** |
| M3：未保存完整 meminfo | 每次 preflight/runtime snapshot 都把完整 `/proc/meminfo` 追加到独立 `*.meminfo.log` | **CLOSED** |
| M4：runtime 越线仅观察 | 保持 contract 的“环境证据、非 PPA、非 runtime kill gate”定义；根 evidence manifest 会封入全部 runtime log | **ACCEPTED AS SPECIFIED** |

## 身份与零漂移检查

| 项目 | 独立结果 |
|---|---|
| current runner SHA256 | `78fc6af6634c020bddfc51a27328d59475ff3ac23810fe34f785973dc9d8324a` |
| current r3 contract SHA256 | `e529aa8a5735fd25028b0c3325523167293be22c3c9760267c2f0397ff604f35` |
| frozen TCL SHA256 | `677aeecc1586ef5abceadeaf68c64700c6cd83178d8b222427673eec7ec72917` |
| r2 frozen input manifest | 24/24 当前均通过；排除 r2→r3 contract 身份替换后 drift=0 |
| runner 内全部显式 SHA pin | 工具、slow/fast DB、12 个 RTL、filelist、SDC、TCL、r3 contract、VCS/hammer/H67/r2 review/r1 preflight/docs359 全部匹配 |
| r2 OOM review inner manifest | `SHA256SUMS` 与 `SHA256SUMS.seal.sha256` 均通过 |
| r1 preflight review inner manifest | `SHA256SUMS` 与 `SHA256SUMS.seal.sha256` 均通过 |
| shell / JSON | `bash -n` 与 `jq -e` 均通过 |
| canonical r3 run / attempt marker | 审阅时二者均不存在 |
| docs/359 | SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

冻结 TCL 的综合体与 contract 逐字一致：

1. `ungroup -all -flatten`
2. `compile_ultra`
3. `compile_ultra -incremental`
4. `compile -incremental_mapping -only_hold_time`

未增加 `set_host_options`，点序仍为 K1→K8→K1x8，每点都重新 analyze/elaborate，未改变 top、ports、SDC、库、工具、RTL 或 effort。

## bash 语义与竞态审计

1. **并发 runner**：两个 invocation 即使同时通过最初的 `-e` 检查，也只有一个能原子 `mkdir` canonical run；另一个在写证据前失败。不存在两个成功 run 目录。
2. **one-attempt**：第一次 K1 launch 前原子 `mkdir` attempt marker；K1 一旦进入该边界，后续任意 DC/主机失败都不能换目录重跑。marker 还记录 runner 与 contract 身份。
3. **逐点资源门**：每点三个 sample 全部必须满足 64/128/32 GiB、`failcnt=0`、`under_oom=0`，任一失败累加并返回 40；第三 sample 后还执行一次进程 gate。
4. **launch 竞态窗口**：marker 后仍有一次进程 gate，再后台启动 DC。操作系统无法把“检查进程”与“启动 DC”做成跨用户全局原子事务，但此实现已经达到合理的 fail-closed 边界；runtime 资源证据会封存后续环境变化。
5. **process regex**：代表性的 DC/FM/PT `common_shell_exec`、`vcs`、`simv` 和 `analyze_m507.py`/`simulate_m600.py` 均命中。诊断命令本身若把完整 Synopsys 字符串写在命令行上，可能形成保守 false positive；只会阻止启动，不会放过冲突或制造有利 PPA。
6. **输出准入**：三点任何一项失败都会留下根 `RUN_FAILED_OR_INCOMPLETE.txt`；只有三点报告、网表、五类 constraint、setup/hold 和最终 manifest 全部通过才置 `m485_complete=1`。

## 非阻塞保留项

1. **preflight-before-first-launch 的运维语义比 contract 更严格。** runner 在 K1 preflight 前已创建 canonical run 目录；若 K1 preflight 失败，attempt marker 不存在，但原 run 目录会阻止直接重启。marker 后的最终 process gate 若失败，也会在未真正 spawn DC 时把 attempt 标为 consumed。两者都只会减少重试机会，不会允许选择性重复。为避免人工清理/解释，启动者应在调用 runner 前先确认资源明显高于门限且无 VCS/EDA/DSE；若仍在 preflight 阶段失败，不得自行删除目录或绕过 canonical path，必须重新独立裁定。
2. **CPU-DSE denylist 不是任意高 CPU 进程的完备证明。** 它只证明 contract 明示的项目 M-series Python 命名族；三次资源阈值、完整进程表和 runtime meminfo 负责暴露其他负载。不得把它扩写成“主机绝对空闲”。
3. **runtime 阈值不是 kill gate。** 完成后的 receipt-blind hammer 必须检查 `resource_runtime.log` 与 `.meminfo.log` 是否出现 OOM/cgroup 异常或严重 headroom 崩塌，再决定 PPA 是否可接受。

## 唯一准许动作

仅在以下条件同时成立时，准许执行当前 frozen runner **一次**：

- runner SHA=`78fc6af6...8324a`、contract SHA=`e529aa8a...604f35`、TCL SHA=`677aeecc...2917`；
- canonical run 与 attempt marker 仍不存在；
- runner 内每个 point 的 3×10 秒资源门全部通过；
- launch 前 DC/FM/PT/VCS/contract-defined CPU-DSE 再检查为零；
- 不修改 compile body、RTL、点序、阈值、工具或输出路径。

本审阅不解锁 Formality、SAIF/PTPX、SRAM macro、paper-ready PPA、完整 FC2/FFN、系统倍速或 DATE headline。只有完整 r3 三点通过原 logic Pareto gate，再通过 receipt-blind 独立打铁，才可进入后续物理/功耗收口。
