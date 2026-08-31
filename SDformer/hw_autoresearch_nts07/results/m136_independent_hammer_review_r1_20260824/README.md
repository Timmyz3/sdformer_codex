# M136 latency-tagged 16-bank response bridge independent hammer

## Verdict

**90/100；P0=0、P1=1、P2=3。Fixed-one-cycle synchronous wrapper contract、rotation、token、two-entry FIFO 和 fail-closed 行为在声明范围内通过；真实 SRAM macro 与 physical one-cycle supply 未准入。**

Sealed commercial VCS 已 exact-SHA 独立重跑，原 pass line 与 covers 复现。另加两组 commercial VCS：

- 65,540 个连续请求/输出，跨过一次 16-bit token wrap；所有 512-bit rotation、metadata 和 token 精确，合法流量 zero-delay monitor 观察到 0 次 `protocol_error` rise 和 0 次 `request_ready` drop；
- FIFO 满/空与 fault 组合：full block、full+pop 同拍接受新 request、reserved response 回填、全部 response 字段 stall stability、live/full reset、stalled-full unsolicited response 和 3-cycle sticky fault 全通过。

## Fixed-one-cycle semantics

这里的 `latency=1` 仅表示 macro wrapper response contract，不是 request 到 consumer accept 的端到端 latency。ready-positive timeline 是：

```text
edge N    request_accept；macro wrapper 注册 request/token/address
cycle N→N+1  macro_response_valid/token/data 保持有效
edge N+1  M136 校验并写入 return FIFO
cycle N+1→N+2  response_valid/FIFO head 可见
edge N+2  response_accept
```

独立 65,540-request 测试对每个 transaction 都测得 request→`response_accept` 为 2 cycles。任何 accepted request 在下一 macro-response sampling edge缺 response、token 错误或出现 unsolicited response 都 fail-close。模块不支持 variable latency、response bubble、macro backpressure 或 CDC。

因此可引用“one-cycle macro return contract”；不可引用“one-cycle end-to-end response latency”。

## Exhaustive mapping and rotation

独立枚举全部 12-bit base：

| Item | Result |
|---|---:|
| Valid 16-word windows | 3,665 (`base=0..3664`) |
| Invalid bases | 431 (`base=3665..4095`) |
| Base-bank offsets | 16/16 |
| Row-crossing valid bases | 3,435 |
| Maximum bank row | 229 |
| Bank conflicts | 0 |
| Physical→logical rotation mismatches | 0 |

对 logical word `base+i`：

```text
bank = (base + i) mod 16
row  = floor((base + i) / 16)
     = base[11:4] + (bank < base[3:0])

logical_output[i] = physical_bank[(base[3:0] + i) mod 16]
```

RTL address/rotation 与上述整数模型完全一致。Stress sequence 的步长 37 与 3,665 互素，实际遍历了全部 base 多轮。

## FIFO projected-count proof

合法 fixed-latency environment 下，state 用 `(fifo_count, pending_response)` 表示。独立 BFS 只有五个 reachable states：

```text
(0,0), (0,1), (1,0), (1,1), (2,0)
```

`(2,1)` 不可达，因为：

```text
projected_count = fifo_count + enqueue_now - pop_now
request_ready   = projected_count < 2
```

即新 request 必须为下一拍返回预留一个 slot。最大 `fifo_count + pending` 是 2，不是 3。20 个 reachable-state/input transitions 全部无 overflow/underflow。

关键并发已在独立 RTL VCS 执行：

- `(2,0)` + stall：request blocked；
- `(2,0)` + pop：同拍 request 可接受，next state `(1,1)`；
- `(1,1)` + stall：pending response 回填到 `(2,0)`，新 request blocked；
- steady II1：enqueue+pop+accept 同拍，保持 `(1,1)`。

## Token wrap

`next_token_q` 按 16-bit modulo arithmetic 自然 wrap。reservation invariant 令最大合法 outstanding 为 2，远小于 65,536，因此只要 fixed-latency/FIFO contract 成立，不会有相同 token 同时 outstanding。

Production 只跑 128 requests；独立 VCS 跑 65,540 requests 并实际观察 `ffff→0000` 一次，65,540 responses 全部 token/order/data exact。

## Stall, reset and fault composition

- RTL 所有 FIFO payload fields 都是 registered head，独立测试对 `{words,start,last,width,tag,token}` 连续 stall 4 cycles，全部稳定。
- Production `ap_stalled_response_stable` 只包含 `{words,tag,token}`，遗漏 start/last/width；RTL 没出错，但 frozen proof tuple 不完整。
- fault priority 会当拍隐藏 output/request accepts，下一 edge 清空 pending/FIFO，并由 `fault_q` sticky 到 reset。独立测试在坏 response 消失后再检查 3 cycles。
- reset 对 request/response/protocol handshakes 有组合 gating；但 reset 是同步的，`pending_response`、`buffered_responses`、`busy` 在首个 reset edge 前仍可显示旧 occupancy。独立 live/full reset 确认 edge 后归零。这必须在接口合同中说明，或将 status ports 也按 reset gating。

## Delta-cycle and physical glitch boundary

Production wrapper 的 `macro_response_valid/token/data` 与 DUT `pending_q/token` 都在同一 posedge 通过 NBA 更新。独立 event monitor 在 65,540 个合法 transactions 上观测：

```text
zero-delay protocol_error rises = 0
zero-delay request_ready drops  = 0
```

这只证明当前 RTL/VCS scheduling 没有 delta pulse。`response_violation` 仍是 external macro-response Q 与 internal pending Q 的组合比较；映射后两条 clock-to-Q path 不可能绝对等延迟，所以不能从 zero-delay 结果声称 `protocol_error/request_ready/response_valid` 是 physically glitch-free。它们在单时钟同步采样前会稳定，当前 synchronous contract 可成立；若这些输出会被异步使用或作为 glitch-sensitive enable，则必须注册 fault decision 或做 gate-level SDF/glitch signoff。

## Actual SRAM feasibility

M136 是 response bridge，不是 SRAM macro implementation：

- foundry SRAM 通常不原生 echo 16-bit token；需要 wrapper 在相同 latency pipeline token、base bank 和 metadata；
- 需要 16 个可同拍读取的 230×32 organization 或等价 macro，并证明 address→macro→512-bit return timing；
- aggregate `macro_response_valid` 无法检测单 bank stale/skew；需要 per-bank valid/deskew 或 wrapper 保证；
- interface 没有 macro request-ready/response-ready，实际 macro 必须严格接受每个 pulse并按固定一拍返回；
- 当前 M136 尚未与 M135 mapper/assembler、M133 consumer 构成一个 sealed integrated timing/data path；
- 无 macro area、routing、CTS/PT、SAIF/PTPX、energy 或 matched 256/512 physical comparison。

Contract 对这些边界总体诚实，因此“宏未实现”列为 paper/physical P1，而不是当前 directed RTL functional failure。

## Scorecard

| Dimension | Score | Evidence |
|---|---:|---|
| Address/rotation correctness | 20/20 | 全部 3,665 valid windows exact。 |
| Fixed-latency/token correctness | 18/20 | 65,540 transactions、一次 wrap；仅支持严格单时钟1拍。 |
| FIFO/backpressure/fault | 19/20 | BFS + targeted VCS；production stall tuple不完整。 |
| Commercial reproducibility | 15/15 | sealed 与 exact-SHA independent rerun。 |
| Macro/physical realism | 8/15 | wrapper contract可执行；无 macro/per-bank-valid/integration。 |
| Claim discipline | 10/10 | macro/physical/system/headline 均 false。 |
| **Total** | **90/100** | **强 synchronous response-bridge milestone，不是 macro-complete frontend。** |

## P0

**0 个。** 未发现 sealed scope 内 rotation、token、fixed-response sampling、FIFO conservation 或 fail-closed 功能错误。

## P1

### P1-1 — 真实 16-bank macro supply 与集成路径未实现

缺实际 macro/wrapper、per-bank response validity、macro timing/area/energy，以及 M135→M136→M133 integrated VCS/DC/PT。它阻断 physical one-cycle bandwidth 与任何 speedup/PPA claim。

可执行修复：建立 target macro adapter，实例化/抽象 16×230×32，按实际 macro latency pipeline token/base/metadata，AND/deskew 16-bank valid，接入 assembler；用相同容量的 256-bit baseline 做 VCS conservation、DC/PT corner、SAIF/PTPX A/B。

## P2

### P2-1 — Production stall SVA 漏三个 payload fields

可执行修复：将 property tuple 改为：

```systemverilog
$stable({response_logical_words,
         response_start, response_last, response_width,
         response_tag, response_token})
```

并增加至少一次非零 start/last/width 的 long-stall cover。

### P2-2 — Zero-delay 无 glitch 不能替代 mapped timing

可执行修复：合同明确所有 handshake/error/status 只在 `clk_core` edge 采样；若需要 glitch-free output，注册 violation/fault 并明确同拍旧 transaction grace。至少增加 mapped-netlist SDF event monitor，检查合法 response start/end 周围无被异步消费的窄脉冲。

### P2-3 — Reset status 的即时语义未冻结

可执行修复：二选一：

- 合同写明 reset 同步，pending/count/busy 只在 reset edge 后清零；或
- 将三个 status outputs 也用 `!rst_core` gating，并扩展 `ap_reset_quiet`。

## Safe claim

> Exact-SHA commercial VCS and independent stress validate a single-clock fixed-one-cycle 16-bank response-wrapper contract with token/metadata alignment, exhaustive conflict-free rotation, and a conservative two-entry return FIFO. An independent 65,540-request run crosses one 16-bit token wrap with no mismatch; ready-positive request-to-consumer acceptance is two cycles, including the FIFO stage. Foundry SRAM macros, per-bank response validity, integrated macro timing/energy and physical/system speedup remain unadmitted.

## Prohibited claims

- 不得把 `latency=1` 写成 request→consumer accept 一拍；
- 不得声称 SRAM macro 原生返回 transaction token；
- 不得将 aggregate valid/token 当 per-bank stale/skew detection；
- 不得把 zero-delay 无 pulse 写成 gate-level glitch-free；
- 不得泛化到 variable-latency macro 或 CDC；
- 不得声称 macro-inclusive PPA、physical/system speedup、FPS 或 headline。

## Artifacts

- `audit_m136.py` / `m136_independent_hammer_audit.json`：全 base rotation、FIFO BFS、sealed/independent VCS 审计。
- `tb_m136_independent_wrap_rotation.sv`：65,540-request wrap、rotation、latency与 event monitor。
- `tb_m136_independent_fifo_reset_fault.sv`：projected-count、全字段 stall、reset/fault attacks。
- `production_rerun_vcs/`、`wrap_rotation_vcs/`、`fifo_reset_fault_vcs/`：commercial VCS 文本证据。
- `source_evidence.sha256`：production/sealed exact evidence。
- `immutable_manifest.sha256`：本 review 可引用文件清单。

未修改 production RTL/SVA/TB/contract。`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
