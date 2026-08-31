# M519 R5 channel-local fault decoupling 独立静态打铁 r1

日期：2026-08-27  
状态：`PASS_STATIC__R5_READY_FOR_ONE_VCS__DC_FORBIDDEN`  
评分：**96/100**；P0/P1/P2 = **0/1/2**。

## 裁决

R5 的源码结构足以授权根 agent 按精确 SHA **至多运行一次三阶段 VCS**。这项授权只表示：

1. 原 TIM-209 诊断中的 request-fault → response-valid 反馈边已在 RTL 结构上切断；
2. unit attack、K1-vs-K1x8 和 K8-vs-K1x8 三阶段 VCS campaign 的输入与硬门已冻结；
3. 当前不存在可引用的 R5 VCS/DC 结果，DC 仍明确禁止。

本审查**不证明 TIM-209=0**。只有后续三个 ARCH_MODE 分别通过 DC precompile gate 才能作该判断。

## P0 审查

### 1. 根因因果锥已静态切断

旧封存环为：

`adapter core_rsp_valid → service mem_rsp_valid/protocol_error → service mem_req_valid → adapter illegal_request → adapter core_rsp_valid`。

R5 中：

- `illegal_request = core_req_valid && !req_payload_legal`；
- `response_channel_open = !fault_q && !illegal_response`；
- `request_channel_open = response_channel_open && !illegal_request`；
- `core_rsp_valid = complete_found && response_channel_open`；
- `core_rsp_accept = core_rsp_valid && core_rsp_ready`。

`illegal_request` 的全部生产 RTL fanout 仅为 `protocol_error`、`request_channel_open` 和 sticky
`fault_q` 更新；不存在到 `response_channel_open`、`core_rsp_valid`、`core_rsp_accept`、
`rsp_hold_valid_q` 或 response retirement enable 的反馈。`illegal_response` 仍同时关闭 request 和
response 通道，符合 fail-stop 语义。

### 2. 同拍优先级与状态账本

- 非法 request 当拍：`request_channel_open=0`，因此 `core_req_accept=0`、全部
  `bank_req_accept=0`，pending/request ledger 不更新；
- 同拍独立合法 response：只要不是 illegal response，仍可按 `core_rsp_ready` 精确 accept；
  `slot_valid/expected/arrived` 和 bundle-response ledger 只在 `response_channel_open` 内、由
  `core_rsp_accept` 清除一次；
- 非法 response 当拍：`response_channel_open=0`，四类 accept 全零；只有 `fault_q/stale_q`
  更新，下一拍 sticky fault 使全部流量静默；
- normal path 未增加寄存状态或生产端口；两份 full workload TB、K8 和 K1x8 datapath SHA
  均保持冻结身份。

### 3. 验证 campaign 未弱化

unit TB 仅通过生产端口驱动，无 hierarchy write、`force/release/deposit` 或仿真专用生产 RTL
hook。静态可见 12 类 directed attack/recovery，包括 cut-through/held response 与非法 request
同拍、非法 response 与合法 request 同拍、pending drain、backpressure、sticky containment 和
reset recovery。46 个 bind-SVA 端口均能在 DUT 名字空间解析；runner 对 10 个关键 cover 做
非零硬门。

两份 r2 full workload TB 的 SHA 未变。runner 不仅检查 PASS 行、数值 mismatch 和 stall
非零，还逐行精确检查五档旧周期，并在 embedded Python 中按有序列表再次比较：

- K1/K1x8：`259/53, 737/133, 3153/499, 7569/1246, 14/14`；
- K8/K1x8：`51/53, 131/133, 486/499, 1231/1246, 14/14`。

### 4. 身份与封存

合同 exact files 19/19；作者交接、原 TIM-209 diagnosis、exit-130 failure hammer 的内外 seal
全部通过。两个 runner `bash -n` 通过，embedded Python 可编译。新 VCS result、新 DC
canonical、attempt identity 均不存在；runner 拒绝覆盖/override。`docs/359` SHA 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## P1：DC runner 成功路径会被 raw-log 自身回显误杀

`run_dc_m519_r5_channel_local_fault_three_axis_exact_sha.sh:265` 对整个 `dc.log` 搜索任意
`TIM-209|OPT-150`。封存的旧 `dc.log` 证明 `dc_shell -f` 会回显 Tcl 源行，其中仅
`regexp ... {TIM-209}`、`regexp ... {OPT-150}` 本身就包含这两个 token。因此即使
`precompile_loop_gate.rpt` 是 0/0，当前 runner 仍会返回 44。

这是**假阴性、不是假阳性**：不会把有环设计准入，但会让未来 DC 成功身份不可达。DC 本轮本来
就被禁止，所以不阻塞一次 VCS；在新 DC launch admission 前必须把 raw-log 门改成只匹配实际
warning/error 形式，或只信任独立生成并精确检查的 `precompile_loop_gate.rpt` 与 PASS terminal。

Tcl 本体的 fail-closed 分支是正确的：FAIL branch 有且只有 `exit 36`，无 ungroup/compile；
PASS branch 有 ungroup 1、compile_ultra 2、incremental-mapping 1、PASS terminal 1。

## P2

1. VCS runner 直接创建最终 result 目录；失败/中断时只写
   `RUN_FAILED_OR_INCOMPLETE.txt`，不生成失败树内外 seal，也不原子移入 quarantine。它不会
   形成假 PASS，且 collision gate 会阻止覆盖，但失败证据的原子性弱于新 DC runner。
2. source-count-out-of-range 与 slice-out-of-range 由 exact TB 的 directed checks 和总类计数
   证明，没有各自独立的 runner-gated cover 名。当前足以覆盖根因，但后续若修 runner，建议补
   两个 cover 以改善可诊断性。

## 唯一授权方式

根 agent 必须同时 pin：

- VCS runner SHA256：
  `e6d7160b47b4f49827dcf7c65ef7036bb9139911b64de2992a0daec350897dc0`；
- 本 review 的 `SHA256SUMS.seal.sha256` 文件 SHA256。

随后可运行**一次** `run_vcs_m519_r5_channel_local_fault_exact_sha.sh`。不得运行 DC。VCS 后仍须
独立 receipt-blind review P0=0、修复上述 DC P1、创建并双封新的 DC launch admission，才可
考虑一次三轴 DC。

## Claim boundary

准入：R5 源码/verification campaign 具备一次 VCS 静态准入。  
未准入：functional PASS、TIM-209=0、combinational-loop-free、DC、PPA、throughput/mm²、
power、energy、完整 FC2、system speedup 或 DATE headline。
