# M519 registered-release 独立静态打铁 r2

日期：2026-08-27  
结论：`STATIC_GO__ONE_EXACT_SHA_VCS_ONLY__DC_STRICTLY_BLOCKED`  
评分：**97/100**  
P0：**0**  
P1：**4（继承 r1、均不阻塞本次一次 VCS）**

本审阅是 receipt-blind 静态复核。未执行 M519 runner、VCS、DC、Formality、PT、PTPX、
Verilator、iverilog 或其他 RTL/EDA 工具；未修改 M519 生产文件、旧 M219/M496、封存结果或
`docs/359`。

## 1. 裁决

r1 唯一 P0 已关闭。新的 r2 recovery contract 把授权顺序明确拆成两个互不循环的阶段：

1. 本 r2 静态 hammer P0=0 且 verdict 明确 `run_vcs=true/run_dc=false` 后，只授权固定 SHA
   runner 的一次 positive Synopsys VCS；
2. VCS receipt/outer seal 必须再经过独立 receipt-blind VCS hammer，随后另建 sealed DC
   launch-admission，才可能授权一次三点 DC。

VCS runner 在创建 canonical result directory 前依次验证：caller-pinned runner SHA、
caller-pinned r2 static-review outer-seal file SHA、r2 contract 固定 SHA、r2 review 三个文件存在、
review inner/outer seal、verdict `P0=0/run_vcs=true/run_dc=false`、review evidence 对 r2 contract
和当前 runner 的精确绑定。任何一项失败均在 `mkdir` 前 exit 3。随后 canonical path 的
existence check 与原子 `mkdir` 使并发调用至多一个成功，并使失败/成功的 positive attempt
都不可通过同一 runner 重放。

DC runner 继续 fail closed：当前 launch-admission 状态仍为
`BLOCKED_PENDING_VCS_AND_INDEPENDENT_STATIC_REVIEW__DO_NOT_LAUNCH`，身份字段为空；修订后的
DC runner还强制读取未来 `vcs_review_outer_seal_file_sha256`，校验独立 VCS review 的 inner/
outer seal，并将其与 r2 contract、VCS receipt/outer seal、r2 static review、DC runner/Tcl、
`docs/359` 一并绑定。上述 gate 均在任何 M519 DC work/preflight directory 创建前执行。

因此本 review **只授权一次 VCS，不授权 DC**。它不准入 SV compile、功能、cycle、组合图无环、
PPA、power、energy、完整 FC2/FFN、系统倍速或 DATE headline。

## 2. r1 P0 关闭证据

| r1 P0 | r2 修复 | 独立裁决 |
|---|---|---|
| recovery contract `run_vcs=false`，又要求 post-VCS DC admission 在任何工具前存在 | r2 `one_vcs_transition` 明确 P0=0 review 是一次 VCS 的授权主体，且明确 `dc_authorized_by_this_transition=false` | **CLOSED** |
| DC admission 依赖尚不存在的 VCS receipt，形成循环 | `mandatory_order` 固定为 static review → one VCS → independent VCS review → separate DC admission → at-most-one DC | **CLOSED** |
| runner 未绑定 authorizing static review | VCS runner 在 mkdir 前验证 review outer seal、inner manifest、verdict 与 evidence 中的 contract/runner 行 | **CLOSED** |

r2 contract 自身仍写 `run_vcs_now=false` 是正确的 prepared 状态；本 sealed review 的
machine-readable verdict 完成合同规定的 transition，并明确限制为一次指定 runner invocation。

## 3. 冻结身份与零漂移

| 项 | 独立结果 |
|---|---|
| r2 recovery contract | `48b63e1a...077b1` |
| VCS exact runner | `8f80af8d...7b55` |
| DC exact runner | `3653d462...cd54` |
| 6 RTL + 2 SVA + 2 TB + 3 filelist + DC Tcl | r2 contract 14/14 SHA 全匹配 |
| r1 recovery contract | `0bec8d8c...894c`，保留未改 |
| r1 static review | inner/outer seal 全通过；outer-seal file SHA `458d9260...a52f` |
| 当前 DC admission | `602b51d8...356d`，仍 blocked |
| shell / JSON | 两个 runner `bash -n`、r2 contract JSON parse 通过 |
| `docs/359` | `dedde7ce...dfc4`，未变 |

r2 correction 没有修改 registered-release RTL、SVA、TB、filelist 或 DC Tcl。因此 r1 已审的
唯一功能语义仍是删除 M219 response-edge free-slot/context bypass；K1/K1x8 下一拍才可复用，
K8 保持 M218/M490；同 top/ports/SDC/3.000 ns/slow-fast DB/ideal clock/ZeroWireload/flatten/
K1-K8-K1x8 compile sequence 均未漂移。

## 4. exactly-one VCS admission 审计

当前唯一合法 positive 调用必须同时由 caller 提供：

```text
M519_EXPECTED_VCS_RUNNER_SHA256=
  8f80af8da28f5e76f1fa43440748f375c714710eaee57169d62a1102737f7b55
M519_EXPECTED_STATIC_HAMMER_R2_OUTER_SEAL_FILE_SHA256=
  <本 review 的 SHA256SUMS.seal.sha256 文件 SHA>
```

runner 还会机械确认本 review 的 `evidence_identity.sha256` 精确包含 r2 contract 与自身 SHA，
并确认 verdict 的 `p0_count=0`、`authorization.run_vcs=true`、`authorization.run_dc=false`。
这些检查全部位于 canonical result existence check 与 `mkdir` 之前。

wrong-runner-SHA negative preflight 允许在 positive 前执行：首个 gate 直接 exit 3，不创建
canonical directory，因此不消费 positive admission。正确 SHA 的 runner 使用固定 result path，
先做 `[[ ! -e ]]` 再原子 `mkdir`；两个并发调用即使同时观察 path 不存在，也只有一个 mkdir
能成功。首次正确授权调用一旦创建目录，无论后续输入漂移、compile、simulation 或 assertion
结果如何，第二次调用都会拒绝覆盖，不能选择性重试。

设计输入 SHA gate 位于 canonical directory 创建后；这只会在漂移时保守消费 attempt 并留下
fail-closed receipt，不会让错误输入进入 VCS。启动者不得删除 failed directory 后重跑；任何
异常都必须另行独立裁定。

## 5. DC 继续严格禁止

本 review 不能满足 DC runner：

1. 当前 launch-admission 不是 `AUTHORIZED_ONE_M519_DC_ATTEMPT`；
2. 当前 admission 不含 future `vcs_review_outer_seal_file_sha256` 字段；
3. VCS receipt/outer seal 尚不存在；
4. `reviews/m519_registered_release_vcs_hammer_r1_20260827` 尚不存在；
5. DC runner要求以上 receipt/review 的 inner/outer seal 全部通过，并要求 caller 同时 pin
   final DC runner SHA 与 future launch-admission SHA。

DC runner 在满足这些条件前不会创建 M519 DC work directory 或 attempt marker。即使有人只把
现有 admission 的 status 字符串改成 authorized，缺字段、非 canonical SHA、admission SHA pin
和缺失 review 文件仍会使 runner exit 3。

## 6. P1 保留项

以下 r1 P1 未在本次 contract-only 修复中处理，不阻塞一次 directed VCS，但必须由 VCS receipt
hammer/后续生产修订明确处理或界定：

1. service SVA 仍用 `mem_rsp_accept && !protocol_error` 近似内部
   `legal_response_accept`；并发非法 response/request 的断言语义可能偏宽。
2. wrong-runner-SHA negative preflight 仍需实际执行并由后续 VCS hammer cross-link；本 review
   只证明其控制路径位于 mkdir/VCS 前。
3. precompile 阶段可严格引用的是 `TIM-209=0`；历史 OPT-150 在 compile update 出现，最终仍需
   检查 dc.log，不能只把 precompile report 的 OPT-150=0 当独立证明。
4. TB 检出 malformed `result_slice>=6` 后仍可能继续索引 `[0:5]` 数值数组；合法路径不触发，
   但 malformed-output fail path 的日志可能带越界/X 噪声。

## 7. 唯一授权动作与后续门

本 review sealed 后，只准许：

1. 可选：用错误 runner SHA 做一次无副作用 negative preflight，必须 exit 3 且 canonical result
   directory 不存在；
2. 用上述两个 caller pin 执行当前 VCS runner **一次**；
3. 若且仅若 runner 生成完整 PASS receipt/outer seal，再由独立 reviewer 审阅原始 compile/sim/
   assertion logs、cycle rows、negative preflight 和 seal。

禁止运行 DC。只有独立 VCS hammer P0=0 后，才能创建新的 post-VCS DC launch-admission；该
admission 必须绑定 r2 contract、VCS receipt/outer seal、本 review outer seal、VCS hammer
outer seal、DC runner/Tcl 与 `docs/359`，并由 caller pin 后，才可能授权一次三点 DC。

