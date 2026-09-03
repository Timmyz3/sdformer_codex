# M1857｜M1850 C2 fresh-mapped Formality/PT 唯一失败独立审阅

结论：**审计 PASS（99/100），M1850 生产准入 FAIL_CLOSED；P0=0、P1=1、P2=0。M1850 已消费且 `automatic_retry=false`，不得重跑；本次既没有证明等价，也没有证明不等价，PT 完全没有启动。**

## 执行与身份

- attempt latch 和 failure quarantine 的 manifest 与外层 seal 均独立校验通过。
- attempt 中的 M1852 release SHA 与当前双封 release 精确一致；quarantine 的 `input_identity.json` 精确绑定 M1850 runner、source contract、Formality/PT Tcl、K8/K1x8 mapped V/SDC/SVF 以及 13 个 live RTL SHA。
- `docs/359` 仍为 `dedde7ce...`，本审阅没有访问或修改 `ucli.key`，没有启动 EDA、license、重试或源修改。
- 唯一 K8 `fm_shell` 进程返回 1；Formality=1，PT=0，K1x8=0，canonical result=0，`FORMALITY_INTERNAL_COMPLETE`=0。
- sealed runner 在消费 attempt 之前固定顺序查询 `Formality` 和 `PrimeTime` 两个 feature；attempt 已在 gate 之后发布，因此可推定 license query=2。但 quarantine 没有独立 `lmstat` 日志，故这是一项由 runner 语义和时序推出的计数，不冒充逐次运行日志。

## 故障发生在 reference link，不是 compare

K8 reference elaboration产生了恰好 8 条 `FMR_ELAB-147`：3 条来自 M214 queue index，5 条来自 M218 protocol/service index。随后 Formality 明确报告：

```text
Error: Unsuppressed RTL interpretation message(s) : FMR_ELAB-147 ... (FM-262)
Error: Failed to set top design ... (FM-156)
Linking requires the following command to be executed:
set_mismatch_message_filter -warn FMR_ELAB-147
```

因此 reference top 没有成功建立。后续 implementation read/set_top、`match` 和 `verify` 虽然在 Tcl 控制流中被走到，但日志分别报告 top 未链接、unknown implementation top 和 `Reference design not set (FM-045)`。报告也明确写着 `Reference: <None>`、`Implementation: <None>`；没有 passing compare point、没有 `Verification SUCCEEDED`。

所以本次失败的准确分类是 **Formality reference-link setup failure**，不是 failing compare points，也不是设计不等价。`verify_return=0` 只反映无有效 reference design 时命令失败，不能被解释为 equivalence compare 的 false。

## 最小合法 successor

M1850 必须永久保持 consumed/failed。若继续闭合，应建立新的 additive namespace，并只做以下最小修复：

1. 在 reference `set_top` 前执行 Formality 自己要求的精确命令 `set_mismatch_message_filter -warn FMR_ELAB-147`；
2. 固定并审计这 8 个 warning 的文件、行号与数量，拒绝新 warning 类别或数量漂移；
3. 在 `match/verify` 前显式断言 reference 与 implementation 两个 design 均已建立；
4. 保留原有严格准入：passing compare points 必须非零，failing/aborted/unmatched/unverified/black-box 必须为零；
5. 经过新的 different-author source review 和 exact one-attempt release 后才可执行。

这个 filter 只能把 Formality 点名的 RTL interpretation diagnostic 从 mismatch severity 降为 warning，**不得**过滤、waive 或隐藏任何 compare point，也不得放宽 verify 失败。只有 successor 真正完成有效 compare，才能建立等价性证据。

## 论文边界

M1850 不产生 C2 Formality、PT setup/hold、功耗、能量或新性能证据，也不推翻此前独立闭合的 C2 DC 面积/周期结果。当前唯一可引用结论是：该唯一 attempt 在 K8 reference link 阶段 fail-closed，未进入有效等价比较。
