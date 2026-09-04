# M812：M799/M533 R17 unit-delay VCS 覆盖失败独立审计

## 裁决

**PASS_FAILURE_AUDIT，100/100；P0/P1/P2 = 1/1/0。** R17 编译、展开和链接成功，正常阶段运行到 5,527,500 ps；在此之前没有观察到数值 mismatch、normal scoreboard error、`protocol_error`、SVA assertion failure 或 timing violation。唯一终止原因是 TB 第 1495 行的必需覆盖 fatal：`pending_plus_forward=0`。

这仍然不是功能 PASS。覆盖 fatal 发生在 P2 strength 打印、held-final stale-parent 测试、六个 protocol attack 和唯一 PASS token 之前；runner 随后在 `functional_and_coverage_gate` 因 PASS token 缺失返回 1。R17 原子结果目录已经建立并双封，attempt 永久消费，raw 只能作为失败证据，禁止引用、重跑或 resume。

## 13 个 TB 计数的精确映射

fatal 向量 `22 9 20 0 166 1 1 12 12 1 58 1 1` 依次为：

1. `dead_plus_read=22`
2. `deadline_read_write=9`
3. `same_address_forward=20`
4. `pending_plus_forward=0`（唯一 TB mandatory gate 失败）
5. `full_no_credit=166`
6. `liveness_sequences=1`
7. `parent_modes=1`
8. `stalled_raw_recovery=12`
9. `stalled_raw_forward_recovery=12`
10. `stalled_raw_response_recovery=1`
11. `pingpong_overlap=58`（只是假名 TB proxy，见下）
12. `endpoint_rows=1`
13. `all_slices=1`

SVA 独立报告中 `cp_pending_plus_forward=0/1842`，与 TB 的 0 一致；`cp_pingpong_overlap=0/1842`，却与 TB 打印的 58 不一致。其余正常路径多个 cover 为正，包括 dead+read 21、same-address forward 18、full/no-credit 157、exact-parent 25、deadline 7、partial-parent multibeat 103、back-to-back completion 43 和 stalled same-address 32。

## 两个 0-cover 的第一性原理判断

### pending + forward

它不是 `forward` 的别名。RTL 为 `read_pending_q && forward_accept_w`，TB oracle 为 `expected_read_response && expected_forward`，SVA 为 `read_pending && forward_event`。三者都要求“上一拍 macro read 的返回”与“本拍另一条 live parent 的 direct forward”同拍进入两槽队列。

该组合在现有 1RW unit-delay 结构中静态可达：n-1 拍队列为空时读取一个已写 parent；n 拍返回该数据，同时另一个 live 当前行完成并向下一个未预取 consumer forward。宏在 n 拍返回 n-1 的读数据并接受 n 拍写入，队列逻辑又明确按 pop→response→forward 顺序入槽。当前 trace 分别命中了 forward 和 response recovery，却没有命中二者交集，所以证据指向 **directed stimulus 对齐缺失**，并未证明 RTL 缺陷。这个 cover 是验证两槽队列最危险转换所必需，不能放宽。

### ping-pong overlap

RTL 的双 bank 允许一个 bank 为 `BANK_EXEC` 时另一个 free bank 接受 preprocessing；SVA 的真实定义是 `prep_active_q && exec_active_q`。但 TB 的 `cov_pingpong_overlap` 只在 `prep_valid && prep_ready` 时递增，谓词里没有 `prep_active` 或 `exec_active`。因此打印的 58 不能替代 SVA 的 0，也不能证明真实 ping-pong。

这首先是 TB coverage proxy 定义错误/刺激保持不足，而不是已证明的 RTL 缺陷。新 TB 必须直接计数实际内部状态或等价公开状态，并让 SVA 同时命中，不能把 SVA minimum 删除。

## attack 与终局边界

normal coverage fatal 先于 `P2_STRENGTH`、`test_held_final_stale_parent_then_legal()` 及全部六个 attack task。日志中没有 coverage/P2/PASS token，三个 SVA attack cover 也都是 0。因此第二阶段没有启动，不能形成 attack functionality 结论。

结果目录 `results/m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_r17_20260828` 是 canonical attempt 本身；`RUN_FAILED_OR_INCOMPLETE.json` 与 `FAILED_DO_NOT_CITE` 存在，`RUN_COMPLETE.*` 不存在，141 项 artifact inventory、成员 manifest 与外层 seal 全部复核通过。runner、release 和 final release hammer 的 exact SHA 与 receipt 绑定一致；资源 final ACK 与三次 collision scan 也正常。`docs/359` SHA 未变。

## 唯一最小下一步

**只制作新的 TB + runner identity，不先改 RTL。** 新 TB 必须加入一个确定性的 response+forward 同拍场景，并把 ping-pong 计数改为真实 `prep_active&&exec_active`；同时保留现有 cleanroom 数值、queue identity/data、foundry response、RAW recovery、P2、六 attack、watchdog 和所有原 coverage minima。随后走新的 exact-SHA source/candidate/release/final-hammer 链；当前不授权 VCS、simv、license 或任何 EDA 运行。

只有当新 directed stimulus 触发数值、协议或 assertion 错误时，才升级为 RTL 修复。R17 本身永久 `FAILED_DO_NOT_CITE`。

