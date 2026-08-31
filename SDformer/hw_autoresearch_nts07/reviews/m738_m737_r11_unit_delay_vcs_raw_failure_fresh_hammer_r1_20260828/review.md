# M738：M737/M533/M528 r11 UNIT_DELAY VCS RAW 失败独立打铁

## 裁决

**PASS（failure classification），98/100。** M737 结果包的双 seal、inventory、编译和 fail-closed 收据均完整。VCS 以 foundry `UNIT_DELAY` 功能模式成功编译；`sim.log` 中没有 timing violation、没有 assertion-failure 签名，但也没有功能 PASS token。唯一 fatal 是 928.5 ns 的 `stalled RAW timeout epoch=2 consumer=7 parent=6`。runner 即使看到 simv 返回 0，仍因内容门失败而返回 1，处理正确。

这个 fatal **不能证明 RTL livelock**，也不是 UNIT_DELAY 宏响应相位的反例。源代码已经足以证明 TB r4 的 RAW watchdog 假设不成立：它把任何瞬时 `stalled_same_address` 都升级成“8 个墙钟周期内必须 forward”的义务；而合法协议允许队列容量在随后被 prior response 占满，使 live write 不能 forward，之后再通过 macro read/response 向同一 consumer 交付 parent。TB token 不识别这条合法替代路径，也不在 consumer 已完成时清理，最终会自触发假阳性。

因此本次不改 top r2。仅允许一个新的 TB-only 候选身份，修复 causal monitor 并增加 observation trace；本评审不授权直接启动 VCS。C1 当前仍是 `RTL verified=false`，r11 仍为 `FAILED_DO_NOT_CITE`。

## 机械复核

- `SHA256SUMS` 和 `SHA256SUMS.seal.sha256` 重新校验均通过。
- inventory 共 140 项：119 个 regular、19 个 directory、2 个内部 symlink；逐项字节/SHA/target 复核为 0 error。
- `RUN_FAILED_OR_INCOMPLETE.json`：`status=FAILED_DO_NOT_CITE`、`phase=functional_and_coverage_gate`、`runner_exit_rc=1`、`child_rc="0"`、`paper_citable=false`。
- `compile.log` 明确编译 checksum-verified foundry `.v`、macro adapter、top r2、SVA r2、TB r4；编译成功。
- `sim.log`：timing violation 0，功能 PASS token 0，assertion failure 签名 0，fatal 1。
- SVA 在 fatal 前报告 `cp_stalled_same_address` 6 次命中；这证明 direct-forward 场景已有覆盖，但 cover 不能被改写成对每次 stall 都必须满足的断言。

## 第一性原理根因

### 1. `stalled_same_address` 不是“forward 已被保证”的事件

top r2 的 `stalled_same_address_w` 条件（492--495 行）要求：当前是 live final、lookahead parent 等于当前 row、且 `base_issue_ready_w` 为 0。它没有 `reserved_count_w < 2` 前提，也没有任何 sink fairness 前提。

与之相反，真正的 `forward_accept_w`（496--498 行）必须满足 live write 已接受并且 `reserved_count_w < 2`。所以从 `stalled_same_address_w` 推出固定时限内 forward，在逻辑上缺少两个必要前提：最终 sink 接受，以及届时队列仍有 reservation 容量。

### 2. 宏读响应是合法而精确的另一条交付路径

当 current 后续完成写入但 queue 已满时，forward 可以合法不发生。消费掉已有 slot 后，top 可用 `macro_read_accept_w` 发起读取，并在下一拍通过 `read_pending_q` 把 exact `(parent, consumer, data)` 响应插入 slot。top 522--576 行明确冻结了 pop → prior macro response → same-cycle forward 的队列顺序；798--814 行分别提交 pending/read/forward 事件。

TB cleanroom 自己也建模了这条路径（944--985 行），但 RAW watchdog（828--847 行）只在 `expected_forward` 时清 token。它没有在 matching `expected_read_response` 时清 token，也没有在对应 consumer 已完成时拒绝 stale token。因此 timeout 反映 monitor bookkeeping，不是设计没有前进。

### 3. 八周期墙钟上界没有协议依据

两个外部 sink ready 由独立 LFSR backpressure 驱动（TB 418--430 行），top 没有声明“8 周期内必 ready”的 assume。即便 queue 永不满，墙钟 8-cycle liveness 也不能从 ready/valid 协议推出。SVA 225 行使用的是 `cover property (stalled_raw ##[1:8] forward_event)`，不是 assertion；TB 把它变成 fatal 是语义升级错误。

### 4. UNIT_DELAY 相位不是本次 fatal 的根因

foundry 模型在 read edge 后 `SRAM_DELAY=0.010 ns` 更新 Q。top 在发起 read 的边沿登记 `read_pending_q`，到下一时钟边沿才把 `scratch_read_data_w` 放入响应队列；10 ps 延迟远早于下一拍。TB 又在 negedge 做 response identity/data observation。本次没有出现 1125/1131 行的 response mismatch，也没有 timing violation。因此没有证据支持“宏相位差导致 RAW timeout”。

## 最小修复候选

新建 TB r5 身份，top r2、macro adapter、SVA datapath 和 foundry 模型都不改：

1. RAW token 按 exact `(epoch, consumer, parent)` 跟踪交付；matching forward 或 matching macro read response 都可完成 token。
2. 删除无 fairness 前提的 8 个墙钟周期 fatal。若要保留有界检查，只能在一个显式 forced-ready、已知 queue-capacity 的 directed subtest 中使用局部 watchdog。
3. `cov_stalled_raw_recovery` 继续只统计 direct forward，并维持 minima；宏读替代路径另增 cover，不能用它冒充 forward cover。
4. 新增纯 observation trace：raw identity/age、current row、两个 slot identity、pending identity、reserved、两路 sink ready、forward、macro-read、read-response。trace 不参与 oracle 状态生成。
5. cleanroom 算术、queue data/identity、conservation、六个 attack、fatal/error/assertion 零容忍和唯一 PASS token 门全部保持。

禁止做法：只把 timeout 从 8 改大、为了迎合假 watchdog 修改 RTL、删除 response identity check、在响应到达前把 macro-read request 当成已交付、把 r11 改写成 PASS。

## P0/P1 与论文状态

- P0：1。C1 没有功能 PASS token，当前不可写 `RTL verified`；须先完成 TB-only 修复身份、静态 hammer、release，再允许一次新 VCS。
- P1：1。UNIT_DELAY 仅为功能验证；slow-corner macro-inclusive setup/hold 仍需独立 DC/PT。
- C1 CPU 同账本性能数字不受本失败改写，但不得借此声称 RTL/物理闭合。
- r11：`FAILED_DO_NOT_CITE`。
- 新候选：`ALLOW_ONE_TB_ONLY_CANDIDATE__NO_LAUNCH_NOW`。

## 冻结检查

`docs/359_DATE终局冻结_20260813.md` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
