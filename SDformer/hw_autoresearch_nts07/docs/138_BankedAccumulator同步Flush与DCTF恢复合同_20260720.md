# Banked Accumulator 同步 Flush 与 DCTF 恢复合同

## 1. 为什么必须修改 Acc

原 `hitflow_banked_accumulator` 只有 reset，没有运行时 flush。中途 abort 后可能残留：

- `group_active_q`；
- bank `busy_q`；
- `value_valid_q` 与 `bias_committed_q`；
- 被 `final_ready` 反压的旧 final；
- 当前 group 的 overflow 状态。

epoch 或 tag 只能防止外部迟到响应匹配错误，不能清除这些内部状态。等待 drain 也不能保证恢复，因为未完成全部 bias 时 `group_finish_ready` 永远不会成立。

## 2. 新同步 Flush 合同

模块新增：

~~~systemverilog
input logic flush
~~~

flush 当拍组合屏蔽：

- `group_start_ready`；
- 所有 `update_ready`；
- 所有 `final_valid`；
- `group_finish_ready`；
- 由 flush 拍输入诱发的 `protocol_error`。

flush 上升沿清除：

- group active/tag 与 group-local bias commit 计数；
- 每 bank busy、token/address、read/addend 暂存；
- valid bitmap 与 bias-committed bitmap；
- accumulator overflow 状态。

`acc_mem` 数据本体不物理清零，依靠 valid bitmap 将旧数据隔离。这样避免大规模清零写端口和复位网络。

性能累计计数器在 flush 上保持：

- updates；
- writes；
- bias commits；
- bank stall cycles；
- final stall cycles。

这些计数仅在 reset 清零。

## 3. 定向验证

专用回归覆盖：

1. 普通 update 已进入读改写流水时 flush；
2. bias 已形成 final、但 `final_ready=0` 时 flush；
3. flush 当拍所有 ready/valid 屏蔽；
4. flush 后使用相同 tag 重新 group start；
5. 新 group 从逻辑零开始，不读取旧有效值；
6. 旧 final 不再出现；
7. counters 跨 flush 保留；
8. overflow sticky 状态被 flush 清除，新 group 可正常产生两个 final 并 finish。

主代理独立复跑输出：

~~~text
PASS: HIT-Flow banked accumulator
RESULT status=PASS quarantined=1 overflow_cleared=1 recovery_finals=2 counters_preserved=1
PASS: accumulator synchronous flush Icarus/Verilator-SVA/Yosys/Erie
~~~

## 4. 工具与回归

| 检查 | 结果 |
|---|---|
| Icarus 主 Acc TB | PASS |
| Icarus overflow/recovery TB | PASS |
| Verilator 动态 SVA 两套 TB | PASS |
| Yosys check | PASS |
| Erie RTL/TB/SVA | 0 error / 0 warning |
| 既有 single-head projection | PASS |
| 既有 multihead projection | PASS |
| 既有 G1 基础路径 | PASS |

现有 Central/G1 顶层显式连接 `.flush(1'b0)`，因此本轮不改变它们的上层生命周期。DCTF 完整顶层将连接真实 context flush。

## 5. 仍需解决的系统问题

1. adapter/executor 的 sticky error 当前保持到 reset；完整 context controller需要定义 error clear 或 context reset边界；
2. bias response也必须携带epoch，不能只保护weight response；
3. 合法数值范围必须证明32-bit Acc不会overflow；否则已经被外部接受的早期 final无法在晚期overflow后撤回；
4. 性能计数器自然回绕时单调性SVA会触发，长时验证需限定窗口或使用wrap-aware property。

本轮只关闭 Acc 本地恢复 P0，不等价于完整 tile 原子提交。
