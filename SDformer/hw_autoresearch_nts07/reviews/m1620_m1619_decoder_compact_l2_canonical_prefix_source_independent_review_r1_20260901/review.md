# M1620｜M1619 decoder compact L2 canonical-prefix 不同作者静态评审

状态：**NO-GO，72/100，P0=0、P1=3。当前 M1619 不授权编写真实 L2 runner，只授权另命名的 source-only 修复。**

本次绑定 exact M1619 source/test/contract/作者收据，以及 M1610、M1615、M1539 和 docs/359 的 SHA/双封。只运行了 CPython 3.6/3.12 合成 mutation，没有打开真实 payload，没有执行 L2/L3、EDA 或 GPU。

已验证的正面边界：prefix 是连续 destination `0..41`，覆盖四种 parity 与 corner/edge/interior；20 个 request exact field 和 16 个 destination exact field 的单边突变全部被拒绝；reset、skip、prefix 不完整、RSS 缺失/超限以及已有单调历史攻击也被拒绝。`actual_prefix_release` 传入 provider/token 仍拒绝，不存在 actual CLI 或论文指标授权。

但 8 个关键 mutation 可穿透，归并为三个 P1：

1. **跨 destination 状态连续性不成立。** 早先 request 的 cycle-5000 return 会被后一 request 的较小 return 覆盖；尚未到期的 outstanding return 可在下一 destination 消失；`last_psum_write_ready` 可从 5000 倒退为 0；cache 可从 9 个 valid entry 清空，只要 tick/计数不倒退。
2. **request 范围与 prefix 账本未绑定。** 等值的 reference/compact request 即使属于错误 module/timestep/destination/output-block 也能通过；实际接收四个 commit request，prefix 却可伪报四个 `external_read`、0 byte 和任意变化的 digest。
3. **跨配置 fresh-session 没有证明。** `validate_bundle` 可接受三条手工构造、request=0、无 dense coverage、未经 `finish()` 的行，因此无法证明三个配置分别使用独立新 session，也无法防止 cache/calendar/outstanding 跨配置复用。

修复要求是让 miter 自己从 accepted request 累加 max-return、kind/count/bytes 和 packed digests，同时保留并审核 outstanding、last-psum 与 cache-content 的合法状态转移；`finish()`/bundle 还需要不可伪造的 session 身份、资源/人口绑定、非零 request 和覆盖证据。修复 source/test/contract 必须再经一次不同作者 P0=0/P1=0 评审，才能考虑另写 actual L2 one-shot runner source；执行仍是更后的门。
