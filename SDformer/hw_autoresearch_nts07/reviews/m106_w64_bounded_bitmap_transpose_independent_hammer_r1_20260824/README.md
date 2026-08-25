# M106 w64 bounded bitmap transpose：独立商业 VCS 打铁评审

日期：2026-08-24

结论：**63/100，P0=2、P1=5、P2=5；M106 当前 NO-GO。** 容量、排序、双 bank、empty window、stall 和既有 fail-closed 路径的正向压力通过，但冻结的 accepted-valid 合同被独立 VCS/SVA 反例击穿。在修复并重跑本评审前，不放行 accumulator miter，也不能把 2.1439× / 2.1422× 升级为 actual replay、scheduled、physical、system 或 headline 指标。

## 独立商业 witness

工具：Synopsys VCS V-2023.12-SP1。评审 TB/SVA、filelist、runner 和全部产物只位于本目录。

正向 adversarial campaign 通过且无 assertion failure：

- 1 个 empty window，确认零 service token。
- 1 个满容量窗口：128 个 key、每 key 全部 64 rows，共 8,192 events；逆 key/row 顺序写入，按升序 key/row 精确读出。
- 后续 sparse window 和 bank reuse window，精确比对 context、source、block、三拍 load、row、destination、direction、last-for-key。
- 8,198 ingress events、133 active keys、399 load tokens、8,198 event tokens、8,597 service tokens全部一致。
- 双 bank fill/drain overlap 6 次；service stall 7 cycles；exact event/close grace 各 2 cycles。
- duplicate `(key,row)`、open-window context mutation、open-window base mutation、stalled-service 上的 event/close collision 四类攻击均进入 same-cycle、reset-only quarantine；6 次 reset recovery 通过。

## P0 反例

### P0-1：精确保持的 close 跨 bank 被重收

当 bank 0 的 close 被接收且 bank 1 原本 EMPTY 时，bank 1 同拍转成 FILL。`accepted_close_grace_match` 虽为真，但 `window_close_ready` 没有用它关断 ready；下一拍保持完全相同的 close 会再次被接收到 bank 1。

独立 assertion `ap_exact_held_close_is_not_reaccepted` 在 19.5 ns 失败，观测到 `accepts=2`。这会制造 phantom window，并使 empty/control-cycle 账本失真。原 sealed TB 的 close grace 发生在另一个 bank 正在 drain、没有 fill bank 可用的情形，因此没有覆盖此分支。

### P0-2：valid-low 前的 request identity mutation 未 fail-closed

RTL 注释和冻结 contract 都规定：刚接收的 valid 可以保持；完全相同的请求不重收，任何 identity mutation 必须同拍 fail-closed。当前组合逻辑只在“新 payload 同时不满足普通语义”时触发 violation：

- event 保持 valid，但把 row 从 1 改成尚未使用的 row 2，会作为第二个合法 event 接收；`ap_held_event_mutation_fails_closed` 失败，`accepts=2`。
- close 保持 valid，在 bank switch 后改变 base/context，会作为下一个 close 接收；`ap_held_close_mutation_fails_closed` 失败，`accepts=2`。

修复要求是把 outstanding grace identity 置于普通 semantic-valid 判定之前：grace 存在时，exact match 只能等待 valid-low 且不得 ready；任何 mismatch 必须组合隔离。修复后应保留连续请求能力的话，需要另行冻结标准 ready/valid 契约，不能同时声称当前的“valid-low 前 mutation 必须 fail-closed”。

## 32,768-bit 存储审计

`row_valid_q` 确为 `2 × 128 × 64 = 16,384` bits，`row_negate_q` 也为 16,384 bits，因此 **presence+direction raw bitmap payload 精确是 32,768 bits / 4,096 bytes**。

它不是模块总状态，也不能直接当成 macro-inclusive SRAM 数字：

- `active_key_q`：256 bits。
- 两 bank base：24 bits。
- 两 bank context：32 bits。
- 两 bank identity-valid：2 bits。
- bitmap 加上述 bank-scoped metadata 的最低值是 **33,082 bits / 4,136 bytes（向上取整）**；尚未计 bank state、drain pointers、accepted-grace shadow、valid/ECC、macro rounding。

key 和 row tag 可由固定地址隐含，不需要逐 event 存；destination 可由 bank base 加 row offset 恢复，context 由 bank-scoped context 恢复。但是这只有在一个 bank 的 `{base, context}` 完整且不可变、并且 16-bit context 格式包含 weight/partition/operator 所需身份时才成立。当前 contract 没有冻结 context bitfield、base alignment/range 或最后一个 partial window 的合法 row 范围。

## 24-bit accumulator 与 1R1W gate

M41 的四层 dense INT8 magnitude envelope 是 877,824，要求 21-bit signed；24-bit accumulator 有 3 个额外 bits，正向幅度余量约 9.556×。纯数值宽度选择合理：

- 64 rows × 768 lanes × 24 bits = 1,179,648 bits = 147,456 bytes。
- 八个 output-block banks 时，每 bank 是 64 × 96 × 24 = 147,456 bits = 18,432 bytes。
- 每个 event cycle 需要对一个 2,304-bit accumulator word 完成一次 vector read 和一次 vector write。

bitmap 保证一个 key 内 row 唯一，因此连续 event 不会访问同一 `(block,row)`；同一 block 在后续 key 再出现前至少有该新 key 的三拍 load。这足以提出“1-cycle RMW、stall 与 writeback 对齐”的 miter 假设，但不足以证明目标 SRAM：真实 macro latency、read-during-write 语义、2,304-bit word 的分段、clear/init、432 partitions 累加、commit 和 backpressure 均未实现。

因此 accumulator miter 的判定是：**当前 NO-GO；accepted-valid P0 修复后，补 context/address 格式与具体 1R1W latency contract，再 GO_TO_IMPLEMENT_AND_PROVE，而不是直接 admission。**

## P1

1. 原 directed VCS/SVA 没有覆盖“close 后另一个 bank 可立即填充”的 exact-hold grace，也没有覆盖 valid-low 前的合法新 event/close payload mutation。
2. `event_storage_bits_total=32768` 只对 raw bitmap payload 成立；最低 bank metadata 另有 314 bits，物理 macro/ECC/rounding 未计。
3. 16-bit context 的 bitfield、partition/operator/weight identity、base alignment/range 和 partial-window row legality未冻结。
4. accumulator 未实现；24-bit 只是 M41 数值上界选择，不是 finite-width、clear、432-partition accumulation 或 commit miter。
5. “8 banks + 1R1W”仍缺具体 macro latency/宽度分段/read-during-write 语义和 stall/writeback schedule。

## P2

1. 本次正向满容量是 synthetic adversarial witness，不是 M105 actual-record replay。
2. RTL 对全部 bitmap bits 使用同步 reset 和任意 bit set/clear，是否可映射目标 SRAM 尚无证据。
3. 12-bit `base + row` 会自然截断；冻结 heldout 范围安全，但接口未 fail-closed 检查 overflow。
4. empty window 的零 service 行为已验证，但全系统 control、descriptor、fill/drain 与 accumulator commit 周期仍未合并。
5. VCS 对本 TB 的 `$fatal` 返回码仍为 0；封存 runner 因而同时强制检查 assertion-report 和反例文本，不能只看进程 RC。

## GO / NO-GO

- exact input identity、原 sealed evidence identity：GO。
- raw 32,768-bit presence+direction payload arithmetic：GO_WITH_METADATA_QUALIFIER。
- synthetic full-capacity ordering、dual-bank reuse、empty/stall、既有四类 quarantine：GO。
- accepted-valid contract：NO-GO，P0。
- next accumulator miter：NO-GO_BEFORE_P0_REPAIR_AND_CONTEXT_PORT_CONTRACT。
- 2.143907× / 2.142234×：仅保留 M105-derived conditional token envelope；actual replay、scheduled cycle、physical/equal-area、system/headline 全部 NO-GO。

本评审未修改 production、contracts、results 或 `docs/359`。
