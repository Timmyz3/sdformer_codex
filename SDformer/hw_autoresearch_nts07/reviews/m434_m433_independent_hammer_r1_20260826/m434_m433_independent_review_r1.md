# M434：M433 exact dual-bank co-read 独立打铁评审

结论：**93/100，P0/P1/P2 = 0/3/2。M433 standalone exact-delta adapter 通过；允许进入 DC、随后 Formality、以及单独 full-population integration 三道下一关，但三者目前都还不是 PASS。论文 headline / system speedup 仍为 NO-GO。**

## 独立复验结果

M434 没有修改 M433 的 RTL、SVA、TB、runner 或已封结果，另写 scoreboard 和攻击 TB，并在 Synopsys VCS V-2023.12-SP1 下 exact-SHA 重编译：

| 项目 | 独立结果 |
|---|---:|
| wide signed12 原码 | 4096/4096 全覆盖 |
| narrow signed8 原码 | 256/256 全覆盖 |
| legal accept / retire | 650 / 648 |
| 显式 fault→reset quarantine | 2，恰等于 accept-retire |
| 正常合法事务丢失 | 0 |
| arithmetic / metadata / order mismatch | 0 / 0 / 0 |
| 攻击类别 | 14 |
| same-cycle / sticky 泄漏 | 0 / 0 |
| 最长 stall | 16 cycles |
| 同拍 pop+push | 643 |
| SVA failure | 0 |

对每一 lane，wide 的位级结果严格是 `{high4,low8}`；narrow 是 signed8→signed12 符号扩展。4096/256 的全原码覆盖不是只测四个边界。

## M427r3 语义红线

端口审计通过：M433 没有 `old_psum`、correction 或 seed-fusion operand，只输出 `contribution_data=update_delta`。因此它没有重犯 M426 的 `old_psum` 覆盖 P0；下游仍必须执行：

```text
new_psum = old_psum + update_delta
```

M426 的 437,640,532 cycles / 1.695794× 继续保持撤销，不能被 M433 复活。

## 309/308 是否偷丢事务

没有发现正常事务偷丢。原 M433 的差 1 来自其明示攻击：先合法 accept 并缓存 5004，再注入 busy reload；组合 fail-closed 在同拍阻止旧输出退休，sticky fault 将其隔离，随后测试用 reset 明确丢弃该隔离项。

M434 独立构造两个这种情况（busy reload、buffered 时 padding fault），得到 `650-648=2`，与显式 reset discard 逐项相等；其他合法事务全部按序退休。这个恢复语义是 fail-stop，不是透明恢复，集成时仍需 poison/rollback 或 window replay，不能把 fault discard 混进正常吞吐率。

## 带宽与声明边界

wide 每次 accept 是 96-byte low + 48-byte logical high = **144 logical B/cycle**；high 由 64-byte sidecar 承载，所以物理输入界面总宽 **160 B/cycle**。冻结 SHARED96 是 96 B/cycle，因此 dual co-read 是新增带宽/面积/互连 Pareto 点，不是免费升级。

本里程碑只证明 standalone exact RTL 的数值和协议，尚未证明 51.84M-row/full-population、周期、系统倍速、DC/STA、Formality、SRAM macro、PPA、功耗或论文 headline。

## 分级问题与下一关

P1：

1. 还没有 M433 DC/STA。padding/metadata legality 到 ready/valid 是组合 fail-closed 路径，功能 II=1 不等于 3 ns timing II=1。
2. 还没有 RTL→mapped-netlist Formality。
3. 还没有接到 full-population codec+accumulator，且没有把 serial96、dual144/160、K2-192 的 SRAM 端口、wire、area、power 做资源归一。

P2：

1. fault 时已有 buffered output 会被隔离并在 reset 丢弃，系统集成需定义重放/回滚。
2. 原 SVA 不直接证明算术/顺序；这次由独立 scoreboard 补齐，full-population 阶段必须继续保留 reference checker。

决策顺序：**GO standalone DC/STA → GO Formality after DC → GO separate full-population integration**。每一道都必须新封证据；在端口公平和宏/互连代价闭合前，paper/headline 仍 NO-GO。

证据备注：第一次 M434 runner `r1` 在 VCS 已跑完后，因 `cd` 后使用相对 `BASH_SOURCE` 写 self-SHA 而终局失败，已保留为 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`。修复仅涉及 runner self-SHA 路径；`r1b` 是唯一 admitted run。

`docs/359` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
