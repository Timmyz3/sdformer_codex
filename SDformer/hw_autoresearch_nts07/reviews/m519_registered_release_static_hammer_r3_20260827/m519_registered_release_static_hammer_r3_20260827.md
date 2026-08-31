# M519 registered-release r3 独立静态打铁

日期：2026-08-27  
结论：`STATIC_GO__EXACTLY_ONE_TWO_TEST_VCS_CAMPAIGN__DC_STRICTLY_BLOCKED`  
评分：**98/100**  
P0：**0**  
P1：**3（均不阻塞本次一次 two-test VCS campaign）**

本审阅是 receipt-blind 静态复核。未运行 M519 runner、VCS、simv、DC、Formality、PT、
PTPX、Verilator、iverilog 或其他 RTL/EDA 工具；未修改 M519 作者 RTL/SVA/TB/filelist/Tcl、
r2 失败树、旧 review/result 或 `docs/359`。本 review 只检查 superseding r3 contract、VCS/DC
runner、冻结输入身份、已封存 failure attribution 和控制/发布顺序。

## 1. 裁决

r3 是合格的最小 runner/contract 修复，**授权 exactly one 新 two-test VCS campaign，仍不授权
DC**。授权对象是 runner：

```text
dc_handoff/scripts/run_vcs_m519_fc2_registered_release_exact_sha.sh
SHA256 = eff58555da2153dfeef5bb9b3e849e15944ee40ef100a388f7f6bd04873add93
```

一次 campaign 必须按 runner 固定顺序完成：

1. caller pin 本 runner SHA 与本 review outer-seal file SHA；
2. 先以 `M519_VCS_MODE=negative_preflight` 执行同一 runner，内部 wrong-SHA child 必须 exit3，
   positive canonical path 不得出现，receipt 固定 `vcs_invocations=0`，并做 inner/outer 双封；
3. 再且仅再以 positive mode 执行 runner 一次；它在全新 canonical r2 path 中从头 compile/sim
   primary K1-vs-K1x8 与 equal-bandwidth K8-vs-K1x8 两套测试；
4. 只有两套 rc/PASS/assertion/cover/cycle-consistency gate 全通过，才生成 receipt r2、
   `RUN_COMPLETE`、member manifest 与 outer seal。

这里的 “one campaign” 包含两次 VCS compile 和两次 simv；不得把它误读成只运行一套测试，
也不得用 r2 失败树中的 primary diagnostic 补齐新 receipt。

## 2. 身份与零漂移

| 项 | 独立观察 |
|---|---|
| r3 recovery contract | `ed2c22eb...2b36`，JSON parse 通过 |
| VCS exact runner | `eff58555...dd93`，`bash -n` 通过 |
| DC exact runner | `6ce350ca...6f85`，`bash -n` 通过 |
| r2 recovery contract | `48b63e1a...077b1`，保留 |
| r2 consumed static authority | outer seal file `2b202d46...f73da`，verdict `475a109c...56e0a` |
| sealed VCS failure hammer | outer seal file `0e74bdb5...84e6`，verdict `e46277cb...4243`，双封验证通过 |
| r2 failed tree | 97 files；实时 fingerprint `8e66bed3...03f` 与 failure hammer 一致 |
| frozen author inputs | 6 RTL + 2 SVA + 2 TB + 3 filelist + DC Tcl，14/14 SHA 匹配 contract |
| `docs/359` | `dedde7ce...dfc4`，未漂移 |

r3 contract 的 `runner_only_delta.rtl_sva_tb_filelist_or_dc_tcl_changed=false` 与机械 SHA 检查一致。
新 positive path `results/m519_fc2_registered_release_k1_vs_k1x8_vcs_r2_20260827` 和 negative
path `results/m519_fc2_registered_release_vcs_r3_negative_preflight_r1_20260827` 审阅时均不存在；
没有旧输出复制或覆盖入口。

## 3. 两项 invalid cover 的最小修复

failure hammer 已封存两项且仅两项 primary runner cover-domain mismatch：

1. candidate external response attack 在 M499 adapter 截止，因此不能要求下游 candidate service
   `cp_protocol_fault_rise>0`；
2. M499 `cp_retire_then_slot_reuse` 的 old same-edge presentation antecedent 与 registered-release
   语义冲突，合法证明是 candidate service 的 next-cycle slot/context reissue。

r3 runner 从 **primary required gates** 精确删除上述两项。它仍然保留：

- top PASS line 的 10 clean、2 reset、4 protocol attack、0 numeric/tuple/weight mismatch、
  0 same-edge-release violation，以及所有 request/response/result/raw stall 与 next-cycle reuse；
- candidate service 的 `cp_same_cycle_distinct_release`、slot/context next-cycle reissue、result stall、
  done；
- 8 个 baseline service 各自的同组 5 项 cover（40 项）；
- M499 的 pending-request stall、out-of-order、cutthrough 与 protocol-attack cover。

对 sealed r2 primary diagnostic 的机械复查显示：上述 candidate/service/adapter 9 项和 baseline
40 项全部非零；被删的两项恰为 0。该复查只证明 gate 选择与既有故障归因一致，**不代替 r3
重新 VCS**。

equal-bandwidth 路径没有删除 cover。runner 仍要求 baseline K1x8 的 b1/b2/b4/b8、full-8、
backpressure、stall/done/protocol fault；candidate K8 service 的 request/replace/stall/done；8 个
baseline registered-release service 的 40 项 cover；以及 M490 的 full-8 response、OOO、
cutthrough、protocol attack 和 M490 语义下合法的 same-edge/retire reuse covers。这里保留的
`candidate.memory_adapter.m490_sva.cp_retire_then_slot_reuse` 不是被 failure hammer 判无效的 M499
property，不能误删。

## 4. fail-closed、拓扑与发布顺序

所有 authorizing identity gate 均位于 positive canonical existence check/mkdir 之前：caller-pinned
runner SHA、caller-pinned review outer seal、review inner/outer seal、verdict P0=0/
run_vcs=true/run_dc=false、evidence 对 contract/runner 的绑定、r3 contract、failure-hammer 双封与
r2 failed-tree fingerprint。任一失败均 exit3 且不会创建 positive canonical path。

negative preflight 在独立新目录中运行；wrong-SHA child 在 runner 首个 SHA gate exit3，控制流上
到不了 VCS 命令。negative receipt、`RUN_COMPLETE`、member manifest、outer seal 顺序明确，
positive mode 又逐项复验该 receipt 与双封。

positive 先原子 `mkdir` 全新 canonical path，之后两套 VCS 均从各自新 `csrc/simv` 编译；runner
没有 `cp`/`rsync`/`mv` r2 diagnostic output 的路径。primary 全 gate 通过后才创建 `equalbw` 并
启动第二套；两套都通过后才解析五行固定 shape，检查每行正周期、0 mismatch、打印 ratio 与
精确 ratio 一致，并交叉要求重复 K1x8 cycles 完全相等。

receipt 的浮点量只来自受 watchdog 约束的正整数周期相除，分母先检查大于零，因此在当前冻结
TB 范围内为 finite。成功发布顺序是 receipt → `RUN_COMPLETE` → member manifest → outer seal →
双封自检 → `task_complete=1`；异常路径保留 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`，不能仅凭内部
PASS 或 `RUN_COMPLETE` 升格。最终 receipt review 仍须严格复核 JSON parse、finite、完整 manifest、
无 failure marker 与 exact output topology。

## 5. DC 严格锁定

本 review 的 machine-readable authorization 明确 `run_vcs=true`、`run_dc=false`。r3 contract
同样写明当前 `run_dc=false`，并要求未来依次出现：

1. 完整 sealed VCS receipt r2；
2. 独立 receipt-blind VCS hammer r2 P0=0；
3. 新建且 sealed 的 `m519_fc2_registered_release_dc_launch_admission_r2_20260827.json`，status
   必须为 `AUTHORIZED_ONE_M519_DC_ATTEMPT`，并绑定 contract、VCS receipt/outer seal、本 static
   review、VCS review、final DC runner/Tcl 与 `docs/359`。

上述 future VCS review、launch admission、DC canonical run 和 attempt-consumed marker 审阅时
全部不存在。DC runner在创建 work/preflight directory 前要求 caller pin final runner 与 future
admission SHA，并逐个读取/校验这些未来身份。故本 review、r3 contract、negative preflight 或
VCS PASS 中任一个单独存在都不能启动 DC。

## 6. P1 与 claim boundary

1. receipt 字段 `authorized_vcs_invocations=1` 实际指 one runner campaign；该 campaign 内明确有
   两次 VCS compile。后续 receipt hammer/文档应使用 “one two-test campaign” 避免计数歧义。
2. Python builder 由正且 watchdog-bounded 的整数保证 finite，但未显式设置 `allow_nan=False`；
   receipt hammer仍应独立 strict-parse 并检查所有 numeric finite。
3. fresh canonical + complete member manifest 给出闭合身份，但 runner未写死 VCS 动态生成物的
   exact filename whitelist；receipt hammer应检查两套 compile/sim、receipt、complete marker、
   manifest/outer seal齐全且无 `FAILED_OR_INCOMPLETE`。

当前只准入“r3 静态授权顺序与 runner gate 修复已通过”。不准入 SV compile、functional VCS、
cycle 数字、完整 FC2/FFN、组合图无环、DC/PPA/power/energy、system speedup、paper-ready 或 DATE
headline。

