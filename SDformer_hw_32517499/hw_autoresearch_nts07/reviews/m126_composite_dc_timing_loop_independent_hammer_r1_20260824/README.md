# M126 composite DC timing-loop independent hammer

## Verdict

**M126 的 production functional seal 保留，但 physical/DC admission 失败。物理准入评分 42/100；P0=1、P1=2、P2=2。**

这次 DC 失败不是单纯的 Synopsys SP3 偶发崩溃。冻结 production RTL 在 `compile_ultra` 之前的 GTECH timing graph 已报告 `TIM-209`；独立的 `analyze/elaborate/link/uniquify + SDC + check_timing` 复跑同样得到 timing loop。原失败 run 随后两轮优化都出现 `OPT-150`，DC 自动断 arc 后 loop 又重现，最后 hold-only incremental mapping 明确因 design has loops 触发 error 263749 和 internal fatal。

因此 `dc_handoff/runs/m126_logic_only_dc_3p000ns_exploratory_r1_20260824/` 整体维持 `FAILED_DO_NOT_CITE`：没有 mapped netlist、可信 STA、area、power、energy、Fmax 或 physical speedup。不能从中间 compile 输出回捞任何 paper 数字。

现有 exact-SHA commercial VCS 仍是有效的合法 directed 功能证据：24,802 source contributions → 7,326 fold updates → 7,326 lane writes，3,072×96 commit 检查和 reset isolation 均不因本次审查被撤销。但零延迟功能仿真停在 error=0 的稳定点，不证明组合图无环，也不证明可综合或 PPA。

## Root cause

静态 RTL 和 DC GTECH loop table 对上了两组控制环；数值 fold/add datapath 本身不是根因。

### 1. Raw child error 的双向 sibling gating

```text
M125 fold_protocol_error
  -> M126 gates M123 update/start/end valid
  -> M123 illegal_request / protocol_error
  -> M126 gates M125 row/fill valid or update ready
  -> M125 illegal_request / protocol_error
```

`protocol_error` 是组合 `illegal_request` 的输出，却被立即拿去门控兄弟模块的 `valid/ready`。两边都这样做时形成 error → valid/ready → illegal_request → error 的闭环。

### 2. M125 redundant busy 经 wrapper audit 回灌

```text
M125 protocol_error
  -> update_valid
  -> busy = fill_active_q || row_active_q || update_valid
  -> M126 wrapper_illegal_request(fold_busy)
  -> M126 gates fold row/fill valid
  -> M125 illegal_request / protocol_error
```

这里 `update_valid = row_active_q && ...`，所以 `update_valid` 必然蕴含 `row_active_q`。把它再 OR 进 `busy` 在二态 Boolean 上完全冗余，却把组合 error cone 带回 wrapper audit。16 组 `fill_active/row_active/selected/error` 穷举的 before/after busy mismatch 为 0。

## Independent reproduction

同一台服务器、Synopsys DC V-2023.12-SP3、同一 TSMC28 max/min DB、3 ns SDC：

| Check | Frozen production | Review-only candidate |
|---|---:|---:|
| DC stage | precompile `check_timing` | precompile `check_timing` |
| `TIM-209` | reproduced | absent |
| DC exit | 0 | 0 |
| Commercial VCS compile/sim | prior sealed PASS | PASS |
| PASS line vs frozen production | reference | exact match |
| SVA cover matches | 160 / 5115 / 2211 / 384 / 1 | 160 / 5115 / 2211 / 384 / 1 |
| Physical admission | **false** | **false** |

Review candidate 有两处最小改动，且只存在本 review 目录：

1. raw child errors 仍立即暴露到 top `protocol_error`，但 sibling `valid/ready` 只由已有 sticky `wrapper_fault_q` 隔离；child error 在下一时钟边沿进入 sticky barrier；
2. M125 `busy` 去掉 Boolean 冗余的 `|| update_valid`，只保留两个寄存状态 `fill_active_q || row_active_q`。

第二项是二态组合等价；第一项保持已准入的无故障数据通路逐周期一致，但改变 fault 隔离的同周期/下一周期关系。因此 candidate 只能叫“可行的最小断环方案”，不能直接改 production，也不能宣称 M126 已准入。

## Score boundary

这个 42/100 只评 **M126 physical/DC admission**，不替代此前 92/100 的 directed functional review。

| Dimension | Score | Evidence |
|---|---:|---|
| 根因定位与独立复现 | 20/20 | Precompile `TIM-209`、mapped `OPT-150`、hold-only fatal 链完整。 |
| 功能证据边界 | 12/15 | 冻结 production VCS 仍有效；明确不能外推 synthesizability/PPA。 |
| Production 组合无环与可综合性 | 0/25 | Production 仍有真实 loop。 |
| 完整 backend/PPA | 0/25 | 无 mapped netlist/clean STA/area/power。 |
| Review-only remediation | 10/15 | Precompile loop-free + 原 directed VCS exact PASS；尚无 fault hammer/FM/PT。 |
| **Total** | **42/100** | **functional milestone 保留，physical milestone 拒绝。** |

## P0

### P0-1 — Production M126 有真实组合 timing loop

这是 physical admission blocker。production 在修复、重新 VCS、完整 DC、Formality/PT 前，任何 M126 area/Fmax/power/physical-speedup 都必须为空。

## P1

### P1-1 — 原 3 ns DC run 没有可引用输出

DC 自动断过 timing arcs，loop 又重现，并在 hold-only mapping fatal。即使 log 中有中间 cell/area/timing 文本，也不是完整、未篡改 timing graph 上的结果。

### P1-2 — Candidate 的 fault contract 尚未独立准入

原 directed TB/SVA 覆盖合法数据通路和 reset attack，不覆盖 child illegal_request 在 raw-error→registered-barrier 改动后的精确周期语义。production merge 前至少要新增：child fault 同周期不误 accept、下一周期 sticky quarantine、reset clear、held-valid/retry，以及两 child 同时 fault。

## P2

### P2-1 — Precompile loop-free 不是 physical closure

候选仍需 `compile_ultra`、mapped loop check、RTL↔netlist Formality、PT setup/hold，再做 SAIF/PTPX；同时仍缺 foundry weight/accumulator macros。当前 candidate 的 `paper_ppa_ready=false`。

### P2-2 — SP3 internal fatal 是次生工具问题

error 263749 可以提交 Synopsys SAR，但生产 precompile timing loop 已独立复现。先修 RTL，再用 loop-free reproducer判断是否仍有工具 bug；不能用 fatal 把 RTL loop 归咎为纯工具异常。

## Required next admission sequence

1. 由 production owner 审查并实现 raw child error 的 registered fault isolation；M125 busy 只保留寄存状态。
2. 新增 fault-contract adversarial VCS/SVA；重跑现有 M126 production、independent、boundary 三套回归。
3. 用最终 production exact SHA 先跑 precompile `check_timing`，要求零 `TIM-209/OPT-150`。
4. 顺序跑完整 DC，检查 mapped netlist、unconstrained path、setup/hold 和 area；禁止复用本失败 run。
5. 再做 Formality、PT STA、SAIF/PTPX。只有这些均通过，才能讨论物理 PPA；宏未绑定前仍不能称 macro-inclusive。

## Claim boundary

Safe statement:

> Commercial VCS functional evidence for the frozen M126 legal directed scope remains valid, but independent precompile DC reproduces real combinational timing loops in production M126. The 3 ns exploratory DC run is failed and supplies no citable PPA or physical speedup. A review-only registered fault barrier plus a Boolean-equivalent M125 busy simplification removes `TIM-209` and preserves the existing directed VCS PASS line; it is not production or physical admission.

Prohibited:

- 不得称原 failed DC 的任何中间数字为 area、Fmax、power 或 physical speedup；
- 不得把 VCS PASS 当成 combinational-loop-free 或 synthesizable 证明；
- 不得把 review-only delta 写成 production fix/admission；
- 不得把 `3.385476×` traffic compression 或 `3.172537×` simulator projection归因于本 DC。

## Artifacts

- `m126_composite_dc_timing_loop_independent_audit.json`：machine-readable verdict、cone、边界和 P0/P1/P2。
- `audit_m126_timing_loop_review.py`：exact-SHA、失败证据、DC A/B、VCS PASS/cover 的 fail-closed 审计。
- `m125_busy_boolean_equivalence_exhaustive.json`：冗余 busy term 的 16-case 二态穷举。
- `original_dc/`、`delta_dc/`：同环境 precompile `check_timing` 对照。
- `delta_vcs/`：原 M126 TB/SVA 的 candidate commercial VCS 回归。
- `m125_registered_state_busy_delta.sv`、`m126_registered_fault_barrier_delta.sv`：review-only，不是 production。
- `manifest.sha256`：文本与 source 证据清单；VCS binary/database 可由 runner 重建，不纳入 manifest。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
