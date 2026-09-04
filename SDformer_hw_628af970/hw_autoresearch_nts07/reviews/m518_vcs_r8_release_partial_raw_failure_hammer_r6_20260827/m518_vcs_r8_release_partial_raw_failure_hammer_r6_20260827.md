# M518 r8 VCS failure receipt-blind hammer（r6，2026-08-27）

## Verdict

`DIAGNOSTIC_CONFIRMED__R8_V16_REDUNDANT_NEGEDGE_BUBBLE__R9_TB_ONLY_READMISSION_REQUIRED`

- 审计完整性评分：**100/100**。
- 问题计数：**P0=0，P1=1，P2=1**。
- r8 不是可引用 PASS：`compile.rc=0`、`sim.rc=0`，但 testbench 在 21.906 us 主动 `$fatal`，runner 正确以 exit 23 写入 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`。
- 失败属于 **V16 testbench stimulus cadence P1**，不是 RTL/SVA/numeric-oracle P0。
- 唯一允许的 r9 修复是删除 `release_partial_raw_attack` 中多余的一整个负沿等待，并用已有 0.2 ns TB skew 在当前负沿之后驱动 raw/release。不得修改 RTL、SVA、`finish_context`、29-cycle oracle、V01--V20 场景集合或 cover 门。

## Receipt identity and fail-closed audit

审阅对象：`results/m518_matched_fixed_t10_atlif_vcs_r8_exact_20260827`。

| Evidence | Observed |
|---|---|
| runner identity | expected = observed = `fe457d7bbf93e72e913c55427696fb782dcc00dee80c74b1f4dba9c3edd01a52` |
| static admission snapshot | `e28022f96b6f0026905c796d977568e5ca69bd9c6d9ec9882be7bd3dc768f5ff` |
| compile / sim rc | `0 / 0` |
| result-level outcome | runner exit 23; `RUN_FAILED_OR_INCOMPLETE.txt` SHA256 `0b5e0cf33d68ff4c29b8cc7f237a2328c09123b5e5edd7c000e60582bc95d466` |
| compile log | no fatal/error marker; SHA256 `dcdf66aeddc9087a413bf2f557a1b361a40c41005dbf8b2a59bdad2ddf0249d0` |
| sim log | `V01 cycle mismatch N=1 got=30 expected=29`; SHA256 `5d7a2e49e5b54c53b8525d1bff8b6f570ec9d62b0ba15bb219f4f36cee0171e2` |
| assertion report | no assertion failure marker, but campaign terminated before all covers; SHA256 `79713cbaf2826081c48e306537c3379716be836c3e2d61b62dc7e1c2496a1d1c` |
| positive author receipt | absent |
| `RUN_COMPLETE.txt` | absent |
| result member/outer seals | absent |
| input snapshot | present, SHA256 `a7546ee6be31f102649531b95b983cbedf2dcf4c89369249de01318fce280e27` |
| docs/359 | unchanged SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

因此该路径虽是预定 canonical pathname，却没有 canonical positive receipt，也没有完成 manifest/outer seal；任何性能、周期、正确性或 PPA 消费均为 fail-closed 禁止。

## Why this is not the first V01 context

`finish_context` 对所有 `exact_cycle=1` 的场景都复用字符串 `V01 cycle mismatch`，所以 fatal 文案不是场景身份。

动态计数给出了唯一定位：

1. 在 targeted V08 之前已经完成 12 次 release：4 个基础 profile、2 个 rail case、4 个 random case、1 个 pressure context、1 个 V06 oldest-bank context。
2. r7 在 phase16 对齐检查处终止，`cp_release=12`。
3. r8 报告 `cp_release=14`、`cp_context_retire=13`。第 13 次是 r8 已经通过 phase16 并完成的 targeted context；第 14 次正是紧随其后的 `release_partial_raw_attack` 当前 release edge。`cp_context_retire` 少一次是因为 cover 在当前 posedge 的 pre-NBA 值采样，而 TB 在该 edge 后 `#1` fatal。
4. r8 的 fatal 时间 21.906 us 比 r7 phase16 fatal 的 21.370 us 晚 536 ns，也与“已越过 line 765 后进入下一 V16 场景”一致。

所以 `#0.2` 没有改变全局 10 ns clock 相位，也没有让开头 N=1 随机漂成 30；它只是修复 r7 的 active-region 观察并暴露了后续旧缺陷。

## Root cause

`send_config()` 的后置条件已经固定：完成 beat 4 后，它执行

```systemverilog
@(negedge clk_core); config_valid=1'b0; ...;
```

然后才返回调用者。普通 `send_tiles()` 就在这个已观察到的负沿直接拉起 `raw_valid`，第一拍 raw 在下一 posedge 接受，因此 clean N=1 是 29 cycles。

但 `release_partial_raw_attack()` 返回后又执行：

```systemverilog
@(negedge clk_core); release_valid=1'b1; raw_valid=1'b1;
```

这不是 settle barrier，而是等待**下一个**负沿，确定性地在 config 与 raw 之间插入 10 ns / 1 cycle 空泡。`release_valid` 仍然在 partial raw 全程保持，攻击语义成立；然而这个无关空泡被 `finish_context` 从首个 config accept 到 release accept 的公平周期账本计入，于是 RTL 和 TB 都得到 30。

证据进一步排除三个错误方向：

- `finish_context` 先检查 `context_retire_cycles==measured_cycles`，该检查没有 fatal，说明 RTL retire ledger 与 wall-clock measurement 同为 30；不是采样错一拍。
- reset 在该场景入口执行四个完整 posedge，并于负沿 deassert；失败不是跨场景残留。r8 也已完成 preceding targeted context 的 retire。
- line 765 的 `#0.2` 发生在固定全局 clock 的两个 edge 之间；随后的 event controls 重新对齐到原 clock，不可能累计成 10 ns 相移。

## The only minimal r9 repair

仅在 `release_partial_raw_attack` 中做下面一处 TB replacement：

```systemverilog
// r8
@(negedge clk_core);release_valid=1'b1;raw_valid=1'b1;

// r9
#0.2;release_valid=1'b1;raw_valid=1'b1;
```

理由：调用点此时已经由 `send_config` 保证位于 negedge；0.2 ns 是本 TB 已使用的 deterministic post-edge skew，距下一 posedge 仍有 4.8 ns，既消除多余整周期，又避开 active/NBA 同区竞态。这里增加新的 clocking block 会扩大 diff 和静态审计面；换成 `@(negedge); #0` 仍保留错误的一周期等待；把 expected 29 改成 30 则会把 stimulus 空泡伪装成硬件周期。

r9 必须机械证明：

- 将上述 `#0.2` 逆替换回 `@(negedge clk_core)` 后精确恢复 r8 TB SHA256 `d03fd23a19046d7b96819f2f8b7753a03cb2cf3454564579b03647026a480de2`；
- line 765 的 phase16 `#0.2` 保留；
- RTL SHA256 仍为 `8a7ec11843b1b9c13c22ab679f69d70f73a8f5874f9ccee51c8873f4f7f142d6`；
- SVA SHA256 仍为 `89d4d711e2913e49ed14d3368c786f069cf11b2ec3f89371dd8582358917c1f5`；
- 29/80 cycle oracle、51 assertions、25 covers、V06 legal-fill 以及 V01--V20 campaign 全部不变；
- 重新独立 static admission 后只能执行一次 exact-SHA VCS，失败路径不可覆盖或复用 r8 目录。

## Findings

### P1-1 — deterministic one-cycle TB stimulus bubble

`release_partial_raw_attack` 在 `send_config` 已返回于负沿后重复等待负沿，导致 30 而不是公平 clean 29。按上面的唯一 r9 replacement 修复。

### P2-1 — generic fatal label obscures scenario identity

`V01 cycle mismatch` 被所有 exact-cycle 调用复用，使第一眼容易误判为 campaign 开头 N=1。为保持本轮“不得改 oracle”的边界，r9 不应顺手改文案；后续验证基础设施清理时可把 scenario id 作为只读参数加入诊断，不影响判定。

## Admission boundary

本评审只准入失败诊断和 r9 TB-only 再审方向；不准入 r8 VCS 行为、完整 V01--V20、29-cycle 性能数字、RTL correctness、DC/Formality/STA/power/energy/PPA、system speedup 或 headline。

