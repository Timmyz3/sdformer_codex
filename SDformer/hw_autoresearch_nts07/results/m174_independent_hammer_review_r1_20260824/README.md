# M174 独立打铁评审

结论：`87/100`，有条件通过为“独立前端 RTL + logic-only DC”里程碑。M174 的 exact-SHA VCS/SVA、fresh seed 复跑和 r1c DC 预承诺全部通过；但它仍只是 bitmap-to-group descriptor 前端，不能把 `2.956840630x` 写成物理、完整 FC2、FFN 或系统加速。

## 独立复核结果

封存 VCS 的 8/8 输入 SHA 和 4/4 输出 SHA 均复核通过。seed=1 的 13 个 required coverpoint 全部正命中，0 assertion failure。独立复用同一个 sealed `simv` 以 seed `174021` 重跑，固定功能账本仍是 50 beats、7 tokens、478 events、174 unique groups、716 replayed group results、1 次 same-cycle token rearm；随机 stall 计数从 165 变为 151，说明 fresh seed 实际生效，13/13 coverpoint 仍全部命中。

r1c 原始报告独立解析如下：

| 指标 | r1c | 预承诺 | 结论 |
|---|---:|---:|---|
| Cell area | 1,530.648002 µm² | < 12,000 | PASS |
| Cells | 2,145 | < 16,000 | PASS |
| Sequential cells | 264 | 无上限 | 记录 |
| Logic levels | 60 | ≤ 80 | PASS |
| Setup slack | +0.3397 ns | ≥ +0.05 ns | PASS |
| Hold slack | +0.0003 ns | ≥ 0 ns | PASS，但物理余量很薄 |
| Macro | 0 | 0 | PASS，亦说明 memory 未计入 |

相对 fail-closed 的 M171，M174 从 103 降到 60 levels，恢复 43 levels；cell area 下降 69.885%，cell count 下降 76.050%，但 sequential cells 增加 26.923%。这说明共享 hierarchical selector 确实消除了 M171 的组合深链，而不是靠放宽 3 ns 约束。

setup 恢复的凭据口径有一个小问题：M174 收据用合同声明的 M171 `+0.0012 ns` 基线计算 `0.3397-0.0012=+0.3385 ns`；但封存 M171 `timing_setup.rpt` 的最差 slack 显示 `+0.0000 ns`，QoR 也显示 `0.00 ns`。因此可直接审计的 report-to-report 恢复是 `+0.3397 ns`；若继续写 `+0.3385 ns`，必须明确它来自合同声明的高精度基线，而非当前 M171 报告。

## fail-closed 审计

- r1：exit 10，SHA preflight 失败。TCL expected digest 只有 62 个十六进制字符，没有进入 DC，不可引用。
- r1b：DC 本体 exit 0 且报告数值与 r1c 一致，但 runner exit 1、没有 `RUN_COMPLETE`，所以仍不可引用。其详细 post-DC failure cause 没有保留，这是 P2 证据工程问题。
- r1c：有 `RUN_COMPLETE`，9/9 输入 SHA 和 21/21 evidence manifest 均通过，是本评审唯一采用的 DC admission。

没有发现把 r1/r1b 失败冒充通过的行为。

## M173 周期口径与 same-cycle rearm

M173 在 128-bit 点给出 isolated-token K1/K4：

- K1：`432,951,702` cycles
- K4：`146,423,753` cycles
- ratio：`2.956840629539x`

M171 无法在 done consume 同拍接收下一 token，所以连续串行口径要各加 `5,579,999` 个 boundary cycles，ratio 降为 `438,531,701 / 152,003,752 = 2.885005766173x`。

M174 的 `scan_ready` 在 `token_done_accept` 为真时开放，identity check 同拍切到新 token 规则，时序块也让新 token 状态覆盖 done clear；directed TB 和 SVA 都实际命中一次 `token_done_accept && scan_accept`。所以在 group/done always-ready、128-bit bitmap 同拍可交付的 M173 假设下，M171 的 token boundary gap 的确被消除，M174 的连续前端 analytic ratio 可回到 `2.956840630x`。

这仍不是 exact payload 的 RTL cycle measurement：120 个 payload 没有通过 M174 RTL timeline 重放，memory 和算术也没有组成。合规说法应是“exact-payload analytic continuous-frontend boundary for the reviewed M174 handshake semantics”。

## 必须补的部分

P0：

1. 128-bit bitmap 目前是免费顶层端口。需要真实 SRAM/寄存器组织或 ATLIF producer tap，并计入 latency、backpressure、area 和 energy。
2. 需要把 M174 与四个 distinct-bank 96-byte weight response、tag、M169 K4、2304-bit accumulator context、BN2/residual 和 FC2 completion 组成有限带宽 shell，在同一记账下比较 K1/K4。

P1：

1. 用 120 个冻结 payload 的代表性与边界片段做 RTL descriptor/timeline replay。
2. 保留 96-bit physical A/B。128-bit 相对 96-bit 的 K4 analytic throughput 仅增加 `1.075677x`，但 bitmap width 增加 33.333%。
3. 扩大 same-cycle rearm 压力测试；当前只有一次 directed 命中。
4. 做 Formality 和 topology-aware/routed STA；ideal-clock/ZeroWireload 下的 `+0.0003 ns` hold 不是物理闭合。
5. PAFT 发布后重新验证 threshold-one 或 folded-weight INT8 数值桥。

P2：保留 r1b 的精确 runner failure cause；修正/限定 M171 `+0.0012 ns` 基线；用 post-run admission overlay 表达通过状态，不修改预运行合同。

下一里程碑应是最小“有限内存可执行 FC2 slice”，不是继续扩裸前端：M174 + bitmap delivery + tagged weight response + M169 + 一个有界 context + completion。只有这一层闭合后，才知道接近 3x 的前端边界还能保留多少。
