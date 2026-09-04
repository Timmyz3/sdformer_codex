# M496 r3 resource-gated exact replay 独立预检

日期：2026-08-27  
审阅范围：只读静态检查 M496 r3 contract、exact runner、冻结 TCL、r2 OOM 独立审阅与 r2 输入身份。未启动 DC、VCS、Formality、PT 或 CPU DSE；未修改生产文件或 `docs/359`。

## 裁决

**`NO_GO_PATCH_RESOURCE_GATE_AND_RESEAL_R3`（82/100）**。

r3 已正确实现三点各自的 `3 × 10 s` 资源采样，64/128/32 GiB 阈值、cgroup 状态、10 s runtime 采样、K1→K8→K1x8 点序、默认单核与冻结 compile body 均对得上 r2 恢复合同；全部 RTL、filelist、SDC、库、工具和 TCL 也未漂移。

但当前 runner 还不能执行：它没有在**每个 point** 的资源门里拒绝另一个 `dc_shell/dc_shell-t/common_shell_exec -shell dc_shell`，而且 `M496_DC_RUN` 可把同一合同重放到任意新目录，未执行 r2 审阅明确限定的“一次 exact replay”。这两个缺口都能让资源选择或重复试跑影响三点比较，属于准入阻塞项。

## 已验证通过的项目

| 项目 | 独立检查 | 结论 |
|---|---|---|
| runner / contract SHA | runner `6f5d915a47a3067eab3d47c1ca694d69450393b3bc67be2a3e7e829110f12a64`；contract `825330757e76442756b014a2b0c518469ca8aae18ac3171283f8072471d23895` | 与委托身份一致 |
| shell 语法 | `bash -n` 通过 | PASS |
| 三次采样 | 每点 `for 1 2 3`；前两次各 `sleep 10` | PASS：采样时刻为约 0/10/20 s |
| 资源阈值 | `67108864 / 134217728 / 33554432 KiB` | PASS：分别精确等于 64/128/32 GiB |
| 三样本合取 | 每次失败累加，`failures == 0` 才继续 | PASS |
| cgroup | `memory.failcnt == 0` 且 `under_oom == 0` | PASS，且本机路径存在 |
| runtime 采样 | DC 运行期间独立 monitor 每 10 s记录，结束后 final snapshot | PASS；它是环境证据，不是 PPA，也不是 runtime kill gate |
| 进程清单 | 每次 preflight 保存 `ps -eo pid,ppid,etime,stat,pcpu,rss,vsz,args` | PASS |
| point 顺序 | `ARCH_MODE=0` → `1` → `2`，即 K1→K8→K1x8 | PASS |
| compile body | flatten；两轮 `compile_ultra`；hold-only incremental mapping | 与冻结 TCL SHA `677aeecc...` 逐项一致 |
| compile 资源/effort | 无 `set_host_options`、无分层/降 effort、每点重新 analyze/elaborate | PASS |
| 输入身份 | r2 input manifest 中除 contract 外的每个文件重新 SHA；0 drift | PASS |
| 输出 fail-closed | DC fatal/非零、缺报告、setup/hold/electrical、unresolved/multidrive/latch 均阻断整次结果 | PASS |
| 成功证据封存 | 根 manifest 会包含 input/runner、三点 resource log、reports/netlist 和点 manifest | PASS |
| claim boundary | logic-only、0 macro、非 system speedup、非 paper-ready | PASS |

## 阻塞问题

### H1：逐点资源门漏检另一个 DC

初始检查只在 runner 第 35–36 行执行一次。之后会进行输入 SHA、manifest 验证、建目录以及每点约 20 秒预检；`m485_resource_gate()` 只检查 FM、PT、VCS/simv 和一类 CPU analyzer，不检查 `dc_shell`、`dc_shell-t` 或 DC 的 `common_shell_exec`。

因此另一个 DC 若在初始检查之后启动，K1 仍可能并发；另一个 DC 若在 K1 期间启动并持续到 K8/K1x8，后续点也会照常启动。这直接违反 r3 contract 的 `concurrent_dc_fm_pt_vcs_or_cpu_dse_allowed=false`，也违反 r2 审阅要求的“每点之前无其他 dc_shell/common_shell_exec”。

**必须修复：**把 DC 三个 matcher 加进 `m485_resource_gate()`，至少在每个 preflight sample 和 DC 启动前最后一刻各检查一次。任一命中均非零退出并留下 incomplete marker。三点都必须走同一门。

### H2：`one_exact_replay_only` 未被 runner 执行

contract 冻结 `one_exact_replay_only=true`，但 runner 允许环境变量 `M496_DC_RUN` 指向任意不存在目录。`[[ ! -e run ]]` 只防覆盖单个目录，不能防同一 r3 合同被重跑到 `...r3b/...r3c` 后挑选有利结果。

**必须修复：**r3 runner 应锁死 canonical r3 run path，或使用一个合同级 attempt marker/全局唯一锁，确保资源门通过后只允许一次 DC attempt。若 preflight 尚未启动 DC，是否消耗 attempt 必须在 contract 中明确；DC 一旦启动 K1，就必须消耗这次 replay，不得换目录重试。

## 中等问题与收口要求

1. **r2 OOM review 只 pin 了 seal 文件本身，未验证其内层 manifest。** 当前目录此刻能通过 `SHA256SUMS` 和 seal 双校验，但 runner 只 `m485_expect` seal SHA；报告/JSON 后续漂移不会阻止执行。执行前应增加该 review 目录内 `sha256sum -c SHA256SUMS` 与 `sha256sum -c SHA256SUMS.seal.sha256`。
2. **CPU DSE matcher 过窄。** 它只匹配 `/system_simulator/scripts/(analyze|independent)_m[0-9]`，不能覆盖其他高 CPU Python/DSE 名称。至少应把项目当前允许并发清单/禁止模式冻结进 contract，或在 preflight 的进程清单上由启动者确认；不能把当前 regex 描述成“证明无任意 CPU DSE”。
3. **未保存完整 `/proc/meminfo` 原文。** runner 保存了所有实际参与门控的字段，足够重算阈值；但 r2 审阅文字要求保存每次 `/proc/meminfo`。建议每个 sample 另存原始 snapshot，或把 r3 contract 明确收窄为冻结字段集合。
4. **runtime 阈值失败被 `|| true` 保留。** 这与当前 contract 把 runtime log 定义成“环境证据”相容；最终 receipt 必须声明 runtime 是否曾越线。不得用一份有 runtime OOM/under_oom 的 run 宣称资源环境稳定。

## 最小解锁条件

只需修 runner/contract，不得改 RTL、TCL、compile effort、点序或门限：

1. 每个 sample 与 DC launch 前重新拒绝 DC/FM/PT/VCS/CPU-DSE 冲突；
2. 执行合同级 one-attempt 语义，不能靠换 `M496_DC_RUN` 重试；
3. 验证 r2 OOM review 内层 manifest；
4. 更新 runner/contract SHA 后做一次新的静态独立预检。

修复后可给 **GO to one exact resource-gated r3 replay**。在此之前不得启动 M496 r3。即便之后三点 DC 全部通过，也仍只是 28-nm、3 ns、ideal-clock、ZeroWireload、0-macro 的 matched logic-only 证据；必须再过 receipt-blind hammer 才能进入 Formality/SAIF/PTPX/common-SRAM。

## 身份边界

- 冻结 TCL SHA：`677aeecc1586ef5abceadeaf68c64700c6cd83178d8b222427673eec7ec72917`
- r2 runner SHA：`21158dfe54efd153e69ba8fdfe0900fbb569b722b24fd80e2618664ed8255274`
- r2 OOM review seal SHA：`6c45826b06f502625efd648d20777a716e4582d7b530b0d2f894c550ac397cd6`
- `docs/359` SHA：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

