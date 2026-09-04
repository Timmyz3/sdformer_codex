# M519 R5 channel-local fault decoupling 独立静态打铁请求 r1

日期：2026-08-27  
作者身份：`contracts/m519_r5_channel_local_fault_recovery_contract_r1_20260827.json`  
合同 SHA256：`779180ed7ca889a92c83273476f6d70a970ed5f8a713e235fd18c4600919160a`  
本请求不授权 VCS、DC、PT、PTPX、Formality 或任何开源 EDA。

## 评审目标

只读验证 R5 是否同时关闭两个已封存 P0：

1. M499 的 request-side `illegal_request` 不再组合撤回独立合法 `core_rsp_valid`，因此
   `core_rsp_valid -> service protocol_error -> service mem_req_valid -> illegal_request -> core_rsp_valid`
   的结构环被切断；
2. 新 DC Tcl 在 TIM-209/OPT-150 非零时使用显式 `exit 36`，而且 `ungroup` 和三条
   `compile*` 文本与控制流都只在 PASS `else` 分支；runner 还要求 PASS-only terminal，捕获
   child/monitor rc，传播 INT/TERM，并在失败/中断后先双 seal 再原子 quarantine。

## 必查 P0

- `core_rsp_valid`、`core_rsp_accept`、slot retirement、bundle-response ledger 使用同一个
  `response_channel_open = !fault_q && !illegal_response`；不得残留 `illegal_request` 或
  `protocol_error` 反馈。
- 非法 request 当拍 `core_req_accept=0`、`bank_req_accept=0`，pending/request ledger 不动；
  同拍已拥有的合法 response 可按 ready 精确 retirement 一次。
- 非法 response 当拍 request/response 两边全部 0 accept；下一拍 sticky fault 全静默。
- normal path 不新增寄存器、端口或周期；K8/K1x8 源码、两份全 workload TB SHA 不变。
- unit TB/SVA 必须通过合法端口覆盖至少十类：同拍交叉攻击、count/mask/zero/channel/slice、
  pending drain、backpressure、held/cutthrough、sticky/reset；禁止 hierarchy write、force、
  release、deposit 或仿真专用生产 RTL hook。
- VCS runner 必须连续跑 attack unit、K1-vs-K1x8 全 r2 regression、K8-vs-K1x8 全 r2
  equal-bandwidth regression；所有旧 cycle 行是 exact gate，不是仅记录。
- 旧 M519 r1 DC attempt、r2 VCS identity、r2 launch admission 均不得复用/覆盖。
- `docs/359` 必须保持
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 静态机械检查

1. 校验 diagnosis 与 failure hammer 内外 seal，并把其 outer-seal 文件 SHA 写入
   `evidence_identity.sha256`。
2. 用合同 `exact_files` 对 19 个文件逐项 SHA；合同本体另行 pin。
3. `bash -n` 两个新 runner；Python 3.6 兼容性检查；禁止执行 runner。
4. 对 Tcl 做括号/分支审计：失败 branch 出现 `exit 36` 且 compile/ungroup 计数 0；PASS
   branch 恰有 `ungroup` 1、`compile_ultra` 2、`compile -incremental_mapping` 1、terminal 1。
5. 反向数据流列出 `illegal_request` 的所有 fanout，证明不存在到 `core_rsp_valid`、
   `core_rsp_accept` 或 response retirement enable 的路径。
6. 检查所有 response state mutation 仅由 `response_channel_open` 及
   `core_rsp_accept`/`bank_rsp_accept` 账本驱动，所有 request state mutation仅由
   `request_channel_open` 及 `core_req_accept`/`bank_req_accept` 驱动。
7. 统计 10 个必需 attack cover 名称和 TB 的 12 类计数；检查 runner逐一硬门。
8. 检查新 result/canonical/attempt 路径当前均不存在，且 runner拒绝 collision/override。

## 预期产物和授权语义

若且仅若 P0=0，创建并双 seal：

- `reviews/m519_r5_channel_local_fault_static_hammer_r1_20260827/`
- JSON 名：`m519_r5_channel_local_fault_static_hammer_verdict_r1.json`
- `status` 必须是
  `PASS_STATIC__R5_READY_FOR_ONE_VCS__DC_FORBIDDEN`
- `severity.p0=0`
- `authorization.run_one_vcs=true`
- `authorization.run_dc=false`
- `evidence_identity.sha256` 必须 pin 合同、新 RTL/TB/SVA/filelists/runners/Tcl、两份旧
  full TB、诊断/failure review seals 和 `docs/359`。

即使静态 PASS，也只授权根 agent 在 caller 同时 pin runner SHA 与本 review outer-seal 文件
SHA 后运行一次三阶段 VCS。DC 仍须等 VCS result 双 seal、独立 receipt-blind P0=0 和新的
`m519_r5_channel_local_fault_dc_launch_admission_r1_20260827.json`。

## claim boundary

静态 review 最多准入“R5 源码和 fail-closed runner 可进入一次 VCS”。不得准入 functional
PASS、TIM-209=0、DC、PPA、throughput/mm2、power、energy、完整 FC2、system speedup 或 DATE
headline。
