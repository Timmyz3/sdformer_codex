# M577｜M518 r4 Fixed / rank3 单点 DC final release 独立打铁评审

日期：2026-08-28  
模式：fresh independent、read-only、final-release only  
结论：**FAIL__NO_M518_R4_POINT_DC_LAUNCH**  
评分：**96/100；P0/P1/P2 = 0/1/0**

本评审没有调用 DC、VCS、runner、远端或大 CPU 任务，没有创建 Fixed/rank3
result、attempt 或 paired-comparison admission，没有修改已有合同、源码、结果、
失败隔离包或 `docs/359`。唯一新增物是本 M577 评审及其 member manifest / outer
seal。

## 1. 裁决

两个 true release 的 payload、member sidecar、outer seal、point/top、权限预算、
候选 SHA、候选评审 SHA、候选评审 manifest/outer seal、runner/contract 身份及
result/attempt 缺席性均闭合；M568 source-static 与 M572/M575 candidate hammer
也都保持 PASS100。

但是两个 true release 对它们绑定的 candidate hammer **写入了不存在的 status
字符串**。这使 final-release provenance 自相矛盾，不能以“SHA 已绑定”为理由
忽略一个明确的状态断言。因此本轮 final hammer 失败，两个当前 release 都不得
交给 runner。该问题不污染 candidate、runner、RTL 或 r2 封存链；最小修复是以
新 release 身份逐字绑定真实 candidate-hammer status，双封后重新做一轮 fresh
final-release hammer。

## 2. P1 finding

### P1-1｜两个 release 的 candidate-hammer status 与所绑定 review JSON 不一致

两个 release 都通过以下 SHA 绑定同一个独立候选评审：

- review：
  `reviews/m572_m518_r4_per_point_dc_launch_admission_candidate_hammer_r1_20260828/review.json`
- review SHA256：
  `df459336391ead6372999de1e68b78439fdd5e225662646b64761dc10c389e3b`
- review manifest 文件 SHA256：
  `97f75d77dd12517fb971ae37fd3d9e7aa25a6e8f4fedd84cc2f8ebfab1c165c0`
- review outer-seal 文件 SHA256：
  `52b29d31cb89ce632f012acfbccadacdea090a98a2ef5ccc35970473aa350788`

该 review JSON 的真实 `.status` 是：

`PASS_M572_M518_R4_TWO_POINT_LAUNCH_ADMISSION_CANDIDATE_HAMMER`

而 Fixed 与 rank3 release 的
`.release_chain.candidate_hammer_status` 都是：

`PASS_M572_M518_R4_PER_POINT_DC_LAUNCH_ADMISSION_CANDIDATE_HAMMER__NO_DC_AUTHORIZED`

后者不是被绑定 review 的 status。score `100` 与 P0/P1/P2 `0/0/0` 都一致，
但这不能修复 status 的逐字冲突。final release 是一次性 EDA 权限根，provenance
断言必须 fail closed。

最小修复边界：

1. 不覆盖当前两个 release；分别创建新的 Fixed/rank3 release 身份及新 sidecar/
   outer seal。
2. 将 candidate-hammer status 逐字写成所绑定 review JSON 的真实值，或删除这个
   冗余字符串、只保留 review path/SHA/score/severity 并由新合同明确其语义；
   两种方案都必须重新 fresh final hammer。
3. 不改 candidate、M568/M572/M575 评审、runner、Tcl、SDC、RTL、工具/DB 或
   r2 失败封存链。

## 3. 已通过的机械核验

### 3.1 true release 与双封

| point | release payload SHA256 | point/top | payload/member/outer seal |
|---|---|---|---|
| Fixed | `72e08fc809c149608f1b0701facc1dd41b433547dd6f36fe7e0f35ce1159bcb9` | `fixed` / `m518_matched_fixed_t10_atlif` | PASS |
| rank3 | `64b191789d4fc908b1c269d215f8bf905b08eaf61da9ff40f49d9c93f85550bd` | `rank3` / `m273_integrated_rank3_atlif` | PASS |

两份 release 与两个 candidate 都是普通文件而非 symlink；严格 JSON 解析未发现
duplicate key。两份 release 当前均为 `launch_now=true`、`max_attempts=1`、
`run_dc=true`，VCS/Formality/PT/PTPX/remote/paired 均为 false；claim boundary
仍把 DC/setup/area/hold/STA/paired/power/energy/PPA/headline 全部置为 false。

### 3.2 candidate、M568/M572/M575 与 r2 冻结链

- Fixed candidate live SHA：
  `e83e2a47319a5fca165fb918adfb64659d1d968022aa946c52e8788bd5aa82a4`；
  rank3 candidate live SHA：
  `7c6fb69062707f542e310b9bcf2ab227ec0ee9397ada3d891e8dd8aea82f2958`。
- 两个 candidate 的 payload/member/outer seal 通过；其 `point/top/result/attempt`
  与对应 release 一致且互不重叠。
- M568 source-static review、M572/M575 candidate hammer 均递归通过 member
  manifest 和 outer seal，且均为 100/100、P0/P1/P2=0/0/0。
- M555 failure review、r2 quarantine、r2 attempt marker 全部递归双封通过；r2
  状态继续为 `FAIL_MATCHED_DC__SEALED_QUARANTINE__DO_NOT_CITE`，Fixed 中间
  QoR 不得追认。

### 3.3 runner 所需字段、身份与逐点隔离

冻结 runner SHA256 为：
`5240712aeaf5dd3b50d68fb29389b1be5d27ba0611c7c50b9d744185c63a00c8`，
`bash -n` 通过。runner：

- 只接受 `M518_R4_POINT=fixed|rank3`，并把 point 分别硬映射到唯一 top、release、
  canonical result 和 attempt 路径；禁止 canonical override。
- 要求 caller 同时固定 runner SHA 与对应 release SHA，校验 release 双封，并用
  `jq -e` 检查 exact authorized status、point、`max_attempts=1`、`run_dc=true`
  及其余 EDA/remote/paired flag 为 false。
- 从 release 读取并校验 source contract SHA，校验 contract 双封、runner SHA、
  identity point、全部 exact files、工具与 slow/fast DB。
- Fixed result/attempt 与 rank3 result/attempt 当前全部不存在；paired-comparison
  admission 也不存在。runner 只生成所选 point 的 work/result/attempt，输出中
  固定 `paired_comparison_admitted=false`，不能消费另一点或生成 paired receipt。
- attempt marker 只在三次 preflight 全部通过后原子搬入；preflight reject 不消费
  attempt。attempt 后失败只进入所选 point 的双封 quarantine。

本评审也确认 runner **不会运行时重新读取 candidate-hammer status**；因此本次
P1 必须在 final-release 层修复，不能把矛盾留给 runner。

### 3.4 公平性、Tcl、SDC、RTL 与工具/DB

source contract live SHA256 为：
`fab51d46ddabff5254943cd1646be107f3fa173447f26cfd3f863b3657e65b5f`。
其七个 exact file SHA 全部命中：r4 runner、冻结 r3 Tcl、共享 filelist、3 ns
SDC、Fixed RTL、rank3 RTL 与 `docs/359`。

- filelist 只有两份冻结 RTL；两顶层各有 50 个有序
  `(direction,width,name)` source tuple，逐项相同；DC bit-level port 口径独立
  冻结为两边各 1175，两个命名空间没有混用。
- Tcl 由 `DESIGN_NAME` 选择唯一 top，共享同一 filelist/SDC/slow-fast DB/operating
  condition/flattening，只执行一次 `compile_ultra`，没有 incremental compile、
  hold fix、hold-only optimization 或 hold report；只准称 setup/area-only。
- SDC 为 3.000 ns、setup/hold uncertainty 0.200/0.050 ns、I/O delay 0.250 ns、
  max-fanout 24。
- DC wrapper/actual executable、slow DB、fast DB live SHA 均与合同一致；
  `dc_shell` realpath 仍为冻结 `snps_shell` wrapper。

### 3.5 资源、碰撞与 terminal 门

release 没有删除或放宽 runner 的任何 gate。runner 保留：64 GiB commit-headroom
三样本（10 秒间隔）preflight、48 GiB 连续三次 runtime soft gate、40 GiB
immediate hard gate、128 GiB MemAvailable、32 GiB SwapFree、cgroup OOM、同 UID
EDA collision、exact child PID/starttime/UID/parent/exe/NUL-safe cmdline，以及
runtime-final 计数/第三样本决策/ACK/monitor-rc 门。

当前只读 live observation 仍发现另一个 UID 的 `simv`：UID `1909`、PID
`580855`。和既有 M570 边界相同，runner 内建 collision 检查只覆盖同 UID；
所以 root 的 **full shared-host collision preflight 是不可省略的外层门**。该
foreign `simv` 存在时，即使修复了 release provenance，实际启动也必须 BLOCK。

## 4. 当前缺席性与 claim boundary

评审时以下路径全部不存在：

- `dc_handoff/runs/m518_r4_fixed_setup_area_logic_only_dc_3p000ns_r1_20260828`
- `dc_handoff/runs/.m518_r4_fixed_setup_area_attempt_consumed`
- `dc_handoff/runs/m518_r4_rank3_setup_area_logic_only_dc_3p000ns_r1_20260828`
- `dc_handoff/runs/.m518_r4_rank3_setup_area_attempt_consumed`
- `contracts/m518_r4_fixed_rank3_paired_comparison_admission_r1_20260828.json`

因此当前没有新的 DC、setup、area、STA、hold closure、paired throughput/area、
power、energy、system speedup、paper-PPA 或 headline 结果。M577 **不推荐执行当前
两个 release**。修复后的新 release 仍须 fresh final hammer PASS；之后也只能在
新的 live full shared-host collision/resource preflight 全部通过、且 result/attempt
仍唯一缺席时，分别考虑一次 immutable runner 调用。两点 raw result 仍须各自
独立 receipt hammer，paired comparison 继续禁止到两点都双封并审过。

`docs/359_DATE终局冻结_20260813.md` SHA256 保持：
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
