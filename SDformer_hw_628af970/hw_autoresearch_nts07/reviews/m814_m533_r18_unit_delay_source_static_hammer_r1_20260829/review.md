# M814 / M533 R18 fresh source hammer

结论：**PASS 100/100，P0/P1/P2 = 0/0/0**。本审查只准入源包进入下一轮独立 candidate hammer；不授权 VCS、simv、license query、release 或任何论文指标。

## 独立语义复核

六个 mask `0001/0003/000c/0031/004c/0083` 按 frozen matcher 的“最大 popcount 精确子集、同 popcount 保留低 row ID”规则，父映射独立推导为 `[null,0,null,0,2,1]`。

四行 P0/P1/C0/C1 witness 确实无效：P0 完成时会直接 forward 给最早的 C0，后续不再存在可与 P1 forward 同拍返回的 P0 macro response。六行 witness 引入 A 和 CA 后，P0 先 forward A；A 消费 P0 且因 CA 保持 live，A 的 live write 阻止同拍读取无关的 C0/P0；随后 P1 第一拍发起 P0 read，第二拍 `read_pending_q` 返回 C0，同时 P1 live write 直接 forward C1。冻结 RTL 的队列顺序是 response 后 forward，因此双 enqueue 可达。

真实 ping-pong 计数条件已改为 `dut.prep_active_q && dut.exec_active_q`，旧 `prep_valid && prep_ready` 代理不存在。96-cycle 合法 sink stall 有成对 release。13 项 normal gate 位于 P2 之前；P2 位于 held-final 与六项 attack 之前；最终 PASS 位于全部阶段之后。

## 机械复核

- 源身份与成员/外层双封全部重算通过；RTL r2、SVA r2、macro adapter、binding plan、foundry Verilog 和 docs/359 均保持指定 SHA。
- `require_regular_sha` 的实际 literal call 是 83 个；函数定义未计入。R17 为 76，R18 新增 7。
- pinned Python 3.6.8：TB static PASS；closure positive PASS；delete-definition、rename-definition、inject-stale 三项负例都按预期失败。
- runner 内 M770 executable heredoc 正例通过；missing-key 与 wrong-key 两项负例均失败在 `M770 launch boundary`。
- runner-owned pre-mkdir stub 返回 rc86，事件顺序精确，VCS identity/license/compile/simv/result side effect 全为零。
- 未创建 R18 result，未创建 launch release，未修改 docs/359。

下一步只能由独立 agent 审 candidate，随后再建立真实 release 与 final hammer；当前仍不可运行 VCS。
