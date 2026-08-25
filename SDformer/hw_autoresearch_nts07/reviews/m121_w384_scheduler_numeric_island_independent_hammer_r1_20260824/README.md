# M121 W384 scheduler→tail-bypass→signed19 island：独立打铁复审

日期：2026-08-24

结论：**86/100，P0=1 / P1=7 / P2=5，要求修复 combined-fault quarantine。** 生产 exact-SHA commercial VCS 的两组 dense descriptor happy path、三拍权重重组、尾拍旁路、98,304 次 signed19 update 和 full commit 数值比对都真实有效；但独立 commercial VCS 证明 scheduler fault 后顶层仍接受 window end 并继续输出 commit，合同的 “combined fail-closed” 不成立。

## 生产 sealed evidence

- compile/sim RC 均为 0；input manifest 16/16、output manifest 4/4、runner manifest 1/1 重新验证。
- 8 个 cover 命中：tail event 256、event→update 98,304、key transition 254、update II1 98,048、lane R/W overlap 98,048、descriptor done 2、full commit 1、numeric fault 1。
- 独立整数复算：

| 指标 | 公式 | 数值 |
|---|---:|---:|
| events | `2×128×384` | 98,304 |
| loads | `2×128×3` | 768 |
| service tokens | `2×128×(384+3)` | 99,072 |
| nonfinal key transitions | `2×127` | 254 |
| update II1 pairs | `2×128×383` | 98,048 |
| commits | `8×384` | 3,072 |
| commit lane checks | `3072×96` | 294,912 |
| signed19 payload | `384×8×96×19/8` | 700,416 B |

这些 happy-path 数字全部成立。

## P0：combined fault 后仍可消费 partial commit

独立 VCS 步骤：

1. 正常启动 accumulator window，完成一个 event descriptor 并等待 update 落入 accumulator。
2. 提交 `row=400` 的非法 scheduler event，得到 sticky `scheduler_protocol_error=1`、top `protocol_error=1`，而 `numeric_protocol_error=0`。
3. 在顶层故障持续时观察到 `accumulator_window_end_ready=1`，window end 成功握手。
4. 随后 `commit_valid=1`，仍与 top `protocol_error=1` 同时存在。

根因是 scheduler error 只 OR 到顶层 error，没有进入 M120 abort/quarantine，也没有门控 window-end 或 commit。下游若按 commit valid 消费，会接收 scheduler 已中止剩余工作后的 partial accumulator。

修复要求：增加 sticky composite-fault/abort 状态；故障后停止 window start/end acceptance，并对外 suppress commit，直到 reset 或显式 verified abort 完成。可以允许已接受的内部 update 在 quarantine 后面排空，但不得把结果作为合法 commit 暴露。必须新增 `protocol_error |-> !window_end_accept && !commit_valid` 一类 SVA，并复跑“已接受 update 后再故障”。

## 另外三个反例

### 1. same-valid 与 whole-descriptor replay

同一个 ingress valid 在握手后保持三拍，只接受一次，证明 M117 exact grace 正常。但 valid 采样为低、descriptor bank 回收后，完整重放同一 base/context/key/row descriptor 会再次被接受：2 accepts、2 closes、6 loads、2 events、2 updates、`protocol_error=0`。

所以 M119 P0 只在 directed scheduler service cut 上关闭；没有 sequence/epoch ID 时，应用层 retry/replay exact-once 仍开放。M121 合同已把 heldout duplicate/retry/escape replay 标为 false，因此这是重要缺口，不是性能 overclaim。

### 2. 计数全对但权重数据错

把 behavioral weight response 从固定一拍改成两拍后，仍得到 3 loads、1 event、1 update，且没有 protocol error；但：

- lane0：得到 0，应为 -128；
- lane95：得到 -93，应为 67。

当前 weight port 没有 response-valid、返回 key/beat 或 error。tail bypass 只能在精确固定一拍模型中解释正确数据。接 foundry SRAM/arbiter前，应冻结宏时序并断言，或增加 response identity/valid。

### 3. cross-descriptor identity 未进入 numeric path

M117 产生 `service_destination_row` 和带 context 的 prefetch identity，但 M121 numeric path 只用 `row_offset`；实际 weight read 也只有 7-bit key+beat。生产 TB 虽用了两组不同 base/context，权重 oracle 却只依赖 key，两个 descriptor 也被加到相同 local accumulator rows。

因此需要明确“同一个 output window、同一套 resident weights”的外部不变量，或把 context/window identity 带进 weight response 与 commit，并断言 destination-to-local row 映射。

## Claim boundary

M121 没有把 `2.53546204172554×` 冒充 RTL、physical、system 或 headline speedup；receipt 明确 `module_cycle_projection_admitted=false`。heldout trace、foundry weight/accumulator SRAM、macro PPA 也都是 false。性能边界正确，错误只在 architecture 的 combined fail-closed 声明。

独立 VCS 日志位于 `vcs_counterexamples_r1/`，机器审计见 `m121_independent_audit.json`，详细 findings 见 `m121_w384_scheduler_numeric_island_independent_hammer_review.json`。本评审只写本目录，未修改 production 或 `docs/359`；后者 SHA 仍为 `dedde7ce...`。
