# M1130r5 对 M1129r5 C2 engine/source 的不同作者 hammer

结论：**GO 只授权由不同作者撰写 zero-argument launcher；STOP launch / attempt / VCS / DC / mapped-VCS。**

词法和机械 diff 闭合：r5 RTL 相对冻结 M1112 仅改 1 个真实 module identifier token；r5 TB 仅改 top 和 DUT type 两个 token。RTL module、TB top、TB DUT 均只有 1 个真实词法命中，filelist 直接选中 r5 RTL，不含 define/include alias。

Selector 的 `verify_dc_selector`、`process_identity`、`terminate_process` 和 `run_dc_with_selector_capture` 在 r5/r4 命名归一化后 AST 完全相同。受控 fake-Popen 验证了 `dc_shell -f` 选择器 argv、same-PID `common_shell_exec -shell dc_shell ... -f` 捕获及环境原样传递；错 argv/UID 攻击被 fail closed。该 mock 没有调用真实 `dc_shell`。

337-bit reset provenance 用受控合成网表通过：每个 active-low clear 都经一级允许反相器回到 `rst_core`。336-bit、直连 active-high reset 和 buffer 攻击全拒绝。

M1128r5 outer、r4/r3 已消耗 attempt 与 failure quarantine 均精确绑定；r4/r3 禁止重试和 namespace 复用。未来 launcher 链不把未来 launch-hammer outer 写入其前置 receipt，无 hash 环；r5 最多仅 1 次 attempt。

本次仅静态与 controlled mock，651 checks、17 attacks，r5 attempt/result/work/failure/lock 在前后均不存在。
