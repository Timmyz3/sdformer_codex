# TCAS-II C1/C2 增量机制取舍（r1，2026-09-04）

## 裁决

不替换 C1 或 C2，也不增加第三个并列贡献。主线保持：

1. **C1：finite-lifetime single-1RW exact product capture**；
2. **C2：typed-K8 + context-safe TSBG pre-read weight delivery**。

二者由同一条电路原则统一：**resource-preceding exact admission**。只有在请求发出前证明该工作对可观察结果冗余，才关闭对应的 product issue、write 或 SRAM read；每个 destination、sign、Acc24 和 completion state 仍独立。

## P0：TSBG 因果消融，不是新机制

采用同一 workload、端口、cache 容量、时钟和 PVT 的三轴：

- ordinary：每个 context 独立取 weight row；
- post-read：共享调度与 identity 判定，但仍完成所有 SRAM read，随后丢弃可复用 payload；
- pre-read：命中在 read request 之前合并，只发一份 weight delivery。

必须同时报告 cycles、accepted bank activations、logic dynamic、SRAM dynamic、leakage、total energy、logic area、setup/hold。该消融用于证明节省来自 **pre-read admission**，不是普通 broadcast、计算后门控或更强 cache。

若 post-read 与 ordinary 的 SRAM activation/energy 相同、pre-read 显著下降，而三轴输出 bit-exact，则 C2 的因果主张成立。当前 M2213 只是 source-only；任何预期 request 数都不是结果。

## P0：reuse-density 到能量的因果曲线

冻结 low/median/high reuse-density 窗口，不按结果重新选择。每个点同时画：

- reuse hit/share ratio；
- accepted bank activations；
- logic dynamic energy；
- SRAM dynamic energy；
- leakage 与 total energy；
- ordinary/TSBG cycles。

若 `R` 为 live rows、`H` 为命中行数、`B` 为每行 bank activation，则理想请求差为

`N_read_post = B*R`, `N_read_pre = B*(R-H)`, `DeltaE_SRAM = B*H*e_read`。

论文必须用实测 counter 和 PTPX/宏模型校验该关系；不能只用解析式代替功耗结果。若低复用点总能量上升，应保留并作为控制开销交叉点，而不是删除。

## P1：B4-union selective bank fill，条件式并入 C2

该机制不成为第三个 contribution。只有同时通过以下门，才把 C2 重述为 **typed-K8 pre-read coalescer with bank-selective fill**：

- 相对同样支持 bank mask 的 ordinary 基线，功能与所有 Acc24/context/tag/terminal bit-exact；
- matched logic area overhead <= 2%；
- 3 ns setup 与 hold 均闭合；
- logic + SRAM 总能量额外下降 >= 15%；
- partial refill、missing-bank、reorder、backpressure 和 eviction 均有覆盖。

否则只作为诊断消融：M2211 的 73.47% 是 directed read-count reduction，不是 latency、energy 或系统加速。

## 不采用的替换/新增项

- 不用 S2、RQTB、C3、decoder 或有损 pruning 替换 C1/C2；它们会稀释五页 Express Brief 的电路因果主线。
- 不把 CPU full-token model、directed VCS 或 component ratio 升级成系统 FPS/energy/frame。
- 不把相同 288 KiB weight capacity 计作 TSBG 的 SRAM area saving。
- 不把 selective bank fill 的部分 RTL PASS 提前写进摘要。

## 对录用概率的预期影响

- 仅有现有 C1/C2/TSBG：约 3.8/5，Weak Accept 边缘；
- 补齐三轴因果消融和 matched power：预计提升 soundness/evaluation，而不是虚增 novelty，目标约 4.1--4.3/5；
- selective fill 再过物理与能量门：可把 C2 从调度技巧提升为带 bank-level admission 的完整 circuit mechanism，目标约 4.3--4.4/5；
- 任一门失败时不降低已有主线，只按负消融报告或删除该扩展。

这些分数是内部审稿尺度，不是录用保证。
