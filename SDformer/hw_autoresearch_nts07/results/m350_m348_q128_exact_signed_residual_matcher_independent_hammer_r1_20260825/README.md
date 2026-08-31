# M350：M348 q128 exact signed residual matcher 独立打铁评审

结论：**78/100，P0/P1/P2 = 0/2/3；数值核心 GO，当前合同完整 RTL、DC/Formality/PTPX 与 headline 全部 NO-GO。**

我用 M348 的 exact-SHA runner fresh 重跑 Synopsys VCS，原始 manifest 和二层 seal 全部回放通过；fresh receipt 与原 receipt 字节一致。随后从顶层 ready/valid 和 payload trace 独立重建 3000 笔交易，重新扫描 128 个 center，不依赖原 TB scoreboard：

| 检查 | 独立结果 |
|---|---:|
| accepted / retired | 3000 / 3000 |
| 数值、顺序 mismatch | 0 |
| lowest-ID winner mismatch | 0 |
| use PWP / fallback | 2898 / 102 |
| mixed plus/minus / exact use | 1667 / 277 |
| latency min / max | 128 / 178 cycles |
| 恰好 128-cycle retire | 407 |
| output stall 稳定 | 609 / 609 |
| 最终最近距离有 tie 的交易 | 1211 |
| 最终最近距离额外 tied centers | 2729 |

因此 `center/id/distance/population`、`1+d<pop`、plus/minus/fallback、lowest-ID tie-break、顺序、全局 stall 与 directed II1 都通过。原 receipt 的 `12730 ties` 不是最终最近距离 tie；TB 是在扫描途中每次遇到“等于当前暂时最优”就累加。这个覆盖仍为非零，但标签不准确。

## 真实阻断：sticky error 后可复活 cfg_active

fresh VCS 反例：reset 后先发送非法 group1，得到 `cfg_protocol_error=1,cfg_active=0`；**不 reset**，再发送合法 group0..7，最终得到：

```text
cfg_protocol_error=1, cfg_active=1, in_ready=0
```

这违反合同的“任何错误 fail closed until reset”，也违反现有 SVA `cfg_protocol_error |-> !cfg_active && !in_ready`。根因是 `cfg_ready` 和配置状态机没有被 sticky error 阻断，合法 group7 仍可把 `cfg_active` 置 1。原 TB 的唯一攻击后立即 reset，所以没有覆盖这条路径。

最小修复：在 `cfg_protocol_error` 为 1 时禁止 `cfg_ready` 和全部配置状态更新，并新增 sticky-error 后无 reset 的完整配置攻击。还应补 early/missing commit、duplicate/skip group、reload、mid-config/mid-pipeline reset、内部 128-stage stall freeze 与精确 latency/order assertion。

## M335 P0 是否修复

- `M335-P0-01` 未受影响：FC2 的 `4.764209x` 仍不得转移成 Conv 数字。
- `M335-P0-02` 只修复了孤立数值接口：M348 已能无条件给出 nearest center、lowest ID 和 signed residual；但还没有连接冻结 q128 catalog/PWP 地址、M307 tau1011 conservation、有限队列或 ordered trace。因此 executable PWP Conv 接口仍未闭合。

## 综合与后续门

当前 SHA 不建议直接进 DC：它有已知合同错误。修复并 fresh VCS 后再做 logic-only DC/Fmax/area，并与 SERIAL16 做等面积吞吐归一化。128 个 stage 包含 127 组 XOR/popcount/compare、约 6.4 kbit pipeline state、2 kbit pattern flops、高扇出 global advance 和 128-way nonempty reduction，必须实测，不能从 II1 直接推性能/PPA。

Formality 需等待修复 SHA 和干净 DC；PTPX 还需 Formality、代表性 SAIF，以及 catalog/PWP memory 的宏或能耗账本。完整评分、反例、盲点和 admission gate 见 `m350_independent_hammer_review_r1.json`。`docs/359` 保持 `dedde7ce...` 未修改。
