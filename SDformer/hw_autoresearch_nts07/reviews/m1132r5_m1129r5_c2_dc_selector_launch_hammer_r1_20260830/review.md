# M1132r5 M1129r5 C2 final launch hammer

结论：**STOP r5 launcher；必须 additive r6 修复后重新打锤。** 不授权 root 或 agent 执行 r5，不授权自动重试。

独立静态与 controlled fake-child 测试共 909 checks、24 attacks，均通过；但在回执形成后，使用 engine 自己的 `static_gate()`/`verify_future_authority()` 做最终反向读取时，发现冻结 engine 无条件读取 `M1130r5 review.identity["m1121_outer_seal_file_sha256"]`，而冻结 M1130r5 review 中不存在该键，因此触发 `KeyError`。这会让真实 r5 launcher 在创建 attempt 前失败。

r4 与 r3 的 consumed attempt 和 failure quarantine 均按 exact outer seal 复核；两者永久 no-retry。测试前后 r5 attempt/result/work/failure/lock 均为空，未运行真实 launcher、engine、pgrep、lmstat、DC 或 VCS。docs/359 SHA 保持 `dedde7ce...`。

r5 命令授权撤回。正确修复必须是 additive r6 engine/launcher namespace：将该检查改为依赖已经双封 launch receipt 中的 M1121 outer，或显式核对 M1130r5 已有的 r3 attempt/failure identities；不得改写冻结 M1130r5 回执，也不得运行 r5 以制造可预见失败。
