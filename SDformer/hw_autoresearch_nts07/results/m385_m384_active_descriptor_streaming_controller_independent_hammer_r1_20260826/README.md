# M385：M384 bounded streaming controller 独立打铁

结论：**92/100，P0/P1/P2 = 0/0/5，准许 exploratory logic-only DC**。
Primary evidence 是修正合同文字后 exact-SHA 重跑的 M384 r1b；r1 保留但已
superseded。r1b 的 RTL/SVA/TB 未变，canonical descriptor 始终是 LSB0：
`[11:0] row / [27:12] original / [34:28] center / [39:35] distance /
[40] use / [47:41] reserved=0`，sticky fault 只能 reset 恢复。

fresh VCS replay 再现：

- receipt 与 `assert.report` byte-identical；
- 18 个 named coverpoint 全非零；
- 4 phase、8 replay、10,804 checked bundle、14 PWP run；
- L1/L2/L4/L8、FIFO/outstanding/credit 最大值均为 8；
- 10 次动态协议攻击、100 sticky cycles、0 numeric/order mismatch；
- PASS payload 保持 `system_speedup=false`、`headline=false`。

独立审计另外执行 100,000 个 descriptor roundtrip、1,114,112 个 strict-use
组合，以及 100,009 个 bitmap、825,077 个 maximal run。`base + center*576`、
`bytes=len*576`、32-byte alignment、64-KiB bounds、no-remap/no-unused-transfer
全部通过。tile1 prefetch 与 replay0 原子启动，replay1 必须等待匹配 tag/bank
的 done；wrong tag/bank、early/duplicate/post-done、third replay 与 unexpected
response 均由 fail-closed 谓词或现有动态攻击关门。

五个 P2 是验证/范围边界：部分 tag/prefetch 负例尚非独立 named VCS test、
q32 nearest/lowest-ID matcher 在模块外、日志未保留逐拍最长连续 request streak、
SRAM 仍是行为接口、没有 17,280-phase RTL cycle miter。它们不阻断 controller
logic-only DC，但禁止 physical SRAM、system speedup、energy、paper PPA 或 DATE
headline 宣称。

主要证据：

- `m385_m384_independent_hammer_review_r1.json`
- `m385_independent_audit_r1.json`
- `recompute_m385_independent_audit.py`
- `fresh_exact_sha_vcs_replay/`

`docs/359` 未修改。
