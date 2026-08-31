# M519 one-shot VCS 失败回执独立打铁 r1

日期：2026-08-27  
结论：`PASS_FAILURE_ATTRIBUTION__TWO_RUNNER_COVER_DOMAIN_MISMATCHES__PRIMARY_SIM_DIAGNOSTIC_ONLY__R3_REAUTH_CONDITIONAL`  
评分：**98/100**  
失败 campaign P0：**3**  
P1：**3**

本审阅只读检查 M519 one-shot 失败目录、原始 compile/sim/assertion 输出、冻结 runner、RTL、
SVA、TB、contract 与既有静态 review。未运行 runner、VCS、DC 或其他 EDA 工具；未修改
M519 候选、失败目录、旧结果或 `docs/359`。

## 1. 裁决

本次 outer runner 的直接 exit40 是**纯 runner cover-domain mismatch**，不是已观察到的
compile、simulation、numeric、transaction、protocol 或 registered-release 功能失败：

- `compile.rc=0`、`sim.rc=0`；
- compile log 有正常 VCS V-2023.12-SP1 identity，没有 anchored Warning/Error/Fatal；
- sim/assert report 中 `failed at`、`Offending`、anchored Error/Fatal、watchdog 全部为 0；
- TB 打印唯一 PASS line：10 clean、2 reset、4 protocol attack，numeric/tuple/weight/
  same-edge-release violation 全为 0；
- request/response-injection/result/raw stall、distinct same-edge request/response、next-cycle
  slot/context reuse、K1 request、K1x8 full issue、两侧 out-of-order 都有非零覆盖。

runner line154--158 把 candidate core service 的 `cp_protocol_fault_rise` 列为必需；实际
assert report 为 0，于是 exit40。外部 scalar-bank spurious response 在 candidate 拓扑中先由
M499 adapter 检查并 sticky fault，adapter `cp_protocol_attack=4`；错误响应不会再穿透成 core
service 的非法 response。baseline 无 adapter，只有实际被注入的 lane0 service
`cp_protocol_fault_rise=1`，其他七 lane 为 0。这个层级分布与拓扑一致，不能要求 candidate
service 复现 adapter 已截获的 fault。

独立继续审计 runner 还发现第二个尚未执行到的 domain mismatch：line168--172 要求 M499
`cp_retire_then_slot_reuse>0`，实际为 0。该 property 的 antecedent 要求 response accept 同拍，
同 slot `core_req_valid=1` 但未 accept，下一拍才 accept；M519 service 与 M499 都已删除同拍
released-slot reuse，正确行为是该同 slot request 到下一拍才首次出现。因此该 antecedent 与
registered-release 设计目标相矛盾。candidate service 的真正次拍证明已经非零：
`cp_release_then_slot_reissue=4144`、`cp_release_then_context_reissue=180`。

所以最小 r3 必须同时修正两个 runner gate，不能只删 `cp_protocol_fault_rise` 后重跑。

## 2. 失败目录的原始事实

| 项 | 独立观察 |
|---|---:|
| outer marker | `FAILED_OR_INCOMPLETE_DO_NOT_CITE` |
| outer exit | 40 |
| compile / sim rc | 0 / 0 |
| primary PASS line | 1 |
| clean/reset/protocol attacks | 10 / 2 / 4 |
| numeric/tuple/weight/same-edge mismatches | 0 / 0 / 0 / 0 |
| request / response injection / result / raw stalls | 1363 / 3143 / 47 / 4509 |
| distinct same-edge request-response | 11025 |
| next-cycle slot / context reuse | 7850 / 322 |
| candidate service slot/context next-cycle covers | 4144 / 180 |
| candidate service protocol-fault-rise | 0 |
| candidate adapter protocol attack | 4 |
| candidate adapter retire-then-slot-reuse | 0 |
| result tree files / bytes | 97 / 5,695,011 |
| snapshot tree fingerprint | `8e66bed3...03f` |

五个 primary cycle rows 也完整打印，K1/K1x8 分别为：

| B/events | K1 cycles | K1x8 cycles | diagnostic ratio |
|---:|---:|---:|---:|
| 1/20 | 259 | 53 | 4.886792x |
| 2/41 | 737 | 133 | 5.541353x |
| 4/90 | 3153 | 499 | 6.318637x |
| 8/110 | 7569 | 1246 | 6.074639x |
| 1/0 | 14 | 14 | 1.000000x |

这些数字只可作为 failure diagnosis 和 r3 回归参考；不得进入 paper、主表、system table、
energy/PPA 计算或与其他倍率组合。

## 3. 为什么是 cover-domain mismatch

### 3.1 candidate service fault cover 不属于外部攻击实际终止层

K1 wrapper 的 top-level `protocol_error` 是
`core_protocol_error || adapter_protocol_error || consistency_fault`。spurious scalar-bank response
首先进入 M499；M499 发现 bank response identity/shape 非法后，`adapter_protocol_error` 拉高，
同时 `core_rsp_valid = complete_found && !protocol_error` 把该非法 response 隔离在 core 之外。

因此 TB 的 top-level protocol check 通过、adapter cover=4、candidate service cover=0 是一致结果。
把 top-level attack 等同为所有内部层级都必须 fault，是 runner 的 coverage-domain 错误。

baseline K1x8 没有 M499 bundle adapter，spurious response 只注入 bank0，所以 lane0 service
cover=1、lane1--7=0。runner 原本没有要求 baseline 八个 service fault cover，这也说明正确
coverage 粒度应跟 fault injection endpoint 对齐。

### 3.2 M499 retire-then-slot-reuse property 属于旧 same-edge presentation 语义

M499 的 registered-open 条件是 `!slot_valid_q[core_req_slot]`，M519 service 的 free slot/context
也只观察 registered state。response edge 清状态后，同 slot request 下一拍才可以变 valid。
但旧 `cp_retire_then_slot_reuse` 先要求 response 同拍已经出现同 slot `core_req_valid`，再要求下一拍
accept；它要求的“同拍呈现但被挡”恰好被新设计消除。

正确证据是 service 层 `response accept ##1 request accept with same slot/context`，本次已分别
4144/180 match。runner 可以继续要求 adapter pending-stall、out-of-order、cutthrough、protocol
attack，但不能要求这个旧 antecedent。

## 4. primary 仿真的合法保留边界

可以保留为 **diagnostic-only primary simulation**：

- 证明当前 exact input identity 在 VCS 中可编译；
- 证明执行过的 K1-vs-K1x8 directed workload 上，TB oracle、conservation、stall、protocol、
  reset 和 registered-release checks 没有观察到错误；
- 为 r3 提供五行 cycle 与 cover-count 回归锚点。

不能把它升级成 M519 VCS PASS：

1. failure marker 明确 exit40；
2. runner 在 equal-bandwidth compile 之前退出，K8-vs-K1x8 完全没运行；
3. final VCS receipt、`RUN_COMPLETE.txt`、member manifest 与 outer seal 全部不存在；
4. 失败目录本身未封存，本 review 只封住所引用的关键文件 SHA 与审阅时 tree fingerprint；
5. r2 authorization 的唯一 positive attempt 已消费，不能删除目录、改名或重用旧授权。

## 5. 失败 campaign 的 P0

1. **Outer result 明确失败且未封存。** exit40/`FAILED_OR_INCOMPLETE_DO_NOT_CITE` 不能由内部
   PASS line覆盖。
2. **完整三轴 VCS 证据缺失。** equal-bandwidth K8-vs-K1x8 未启动，repeated K1x8 cycle identity
   无法比较，receipt builder 未执行。
3. **r2 one-shot 已消费。** 不能覆盖原目录或用同 contract/review 选择性重试；DC 所需 sealed
   VCS receipt 与 independent VCS review 均不存在。

## 6. r3 最小修复

允许的最小代码范围是 **runner/contract/authorization only**；RTL/SVA/TB/filelist 可保持原 SHA。

1. 新建 superseding r3 recovery contract，记录 r2 attempt path、exit40、failure hammer outer seal，
   明确 r2 永久 diagnostic-only，并只允许一个新 r3 VCS attempt。
2. 新 runner identity 与全新 canonical r3 result path；拒绝覆盖、删除、rename 或读取 r2 primary
   sim 当作 positive receipt 输入。
3. candidate service required cover 列表删除 `cp_protocol_fault_rise`；protocol gate 改为同时要求
   TB `protocol_attacks=4`、top PASS line 和 M499 `cp_protocol_attack>0`。
4. M499 required cover 列表删除 `cp_retire_then_slot_reuse`。继续要求
   `cp_pending_request_stall`、`cp_out_of_order_bundle_response`、
   `cp_cutthrough_bundle_response`、`cp_protocol_attack`；次拍 slot/context reuse 继续由 candidate
   M519 service 的两个 nonzero cover 与 TB zero-violation counters证明。
5. 静态 review 必须机械审计修订后每个 primary required cover 在本 diagnostic assert report 中
   已非零，避免再留下已知的第二个零覆盖；这只验证 gate 可达，不代替 r3 重新仿真。
6. r3 必须从头运行 primary 和 equal-bandwidth 两套 compile/sim，使用原 seeds 或合同冻结的新
   seeds；不得复制本次 compile/sim/cycle/cover 到新目录。
7. r3 runner 只有在两套 rc=0、PASS line、assertion failure token=0、所有分层 required cover>0、
   repeated K1x8 cycles一致、receipt生成并双封存后才能 PASS。

若选择修改 SVA、新增 adapter-compatible `core_rsp_accept ##1 core_req_accept` cover，则不再是最小
runner-only 修复：SVA/filelist/contract identity 必须全部刷新，并接受独立静态 review。

## 7. r3 重新授权门

任何新 VCS 前必须同时满足：

1. sealed r3 contract 绑定本 failure hammer、r2 contract/static review、冻结设计身份和新 runner；
2. independent r3 static hammer P0=0，明确 `run_vcs=true/run_dc=false`；
3. caller pin 新 runner SHA 与 r3 static-review outer-seal file SHA；所有授权 gate 在新目录 mkdir 前；
4. wrong-runner-SHA negative preflight 实际执行，exit3、无 r3 canonical directory，并封存/cross-link；
5. 新 canonical path 原子 one-shot，失败亦不可重试；
6. r2 failed directory 保持原位且本 review 引用的关键 SHA 不变；
7. DC runner/admission 不得提前改成 authorized。

r3 VCS PASS 后仍必须独立 receipt-blind VCS hammer P0=0。只有随后创建并封存一份新的
post-r3-VCS DC launch-admission，绑定 r3 contract、VCS receipt/outer seal、r3 static review、
VCS hammer、final DC runner/Tcl 与 `docs/359`，才可能授权一次 DC。

## 8. P1

1. r2 wrong-runner-SHA negative preflight 没有找到独立 sealed receipt；r3 必须补齐。
2. failure directory 未自带 manifest，后续不得依赖它保持不变；本 review 的 selected-file SHA/
   tree fingerprint 只是审阅快照。
3. service SVA 用 public `mem_rsp_accept && !protocol_error` 近似内部 legal response 的既有 P1
   仍在；它没有导致本次 failure，但后续若改 SVA 应一并修正。

`docs/359` 审阅结束时 SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

