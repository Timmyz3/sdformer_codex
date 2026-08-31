# M529/M528 dead-write-only 1RW RTL 作者准入独立锤审

日期：2026-08-27  
模式：admission-only、只读证据审计；未写 RTL/SVA/TB/runner，未运行 VCS/DC/Formality/PTPX/CPU/GPU  
裁决：**98/100，P0/P1/P2 = 0/0/2；只允许一个全新作者 subagent 创建一个完整 source-only 包。**

## 1. 封存与身份

- 准入 JSON 的严格 SHA 为 `968a56143bf95c825af917d0819a502b91b76a274a40e943d9b57fdbd9e31960`；无重复键。
- 准入 member/outer 两层封存通过。
- M528 r4 result、独立 result hammer、RTL reuse prespec 的 member/outer 封存全部通过；准入内嵌 SHA 与磁盘一致。
- M474、M476r2、M498 core、M504 tie 语义、SRAM mapping、generated macro manifest 和 `docs/359` 身份一致，0 mismatch。
- 授权目标目录在本审阅时均不存在，说明唯一作者额度尚未消耗。

prespec 中旧的 M528 r1 execution contract 只用于冻结设计复用边界，不是本次执行或性能授权来源。当前准入直接绑定 M528 r4 result 与独立 r4 result hammer；它们是唯一有效的执行/数字统治链。旧 prespec 身份不授予 launch，也不覆盖 r4，因此不构成阻断冲突。

## 2. 结构边界裁决

唯一顶层是 `m528_dead_write_only_1rw_product_capture_island`。准入完整覆盖：dynamic 64-row exact-subset matcher、ping-pong directory/live、executing-bank written/epoch、稳定 `(popcount(original), row_id)` 顺序、唯一 earliest-parent lookahead、deadline hold、两 reserved-entry queue、signed12/signed19 算术、9-slice generated-macro adapter 和 architectural commit/completion/conservation。

关键 fail-closed 语义也已锁定：

- dead final 只省 parent-scratch write，psum commit、row completion、算术和计数不能省；
- live final 必须写，包括已同拍 forward 的 single-use row；
- 普通 read 必须先 written；唯一例外是同拍 live final 写新值并 forward，同时抑制 macro read；
- queue+pending 不超过 2，full 时禁止借 same-cycle consume credit；
- 9 颗宏共享 coherent control/address，连续拼接 9 个 128-bit slice，仅允许 row 0--63；
- stale RAW、overflow atomicity、ownership/epoch 和 backpressure 都必须进入 SVA/TB。

combined PVRF、single-use store elision、concurrent-1R1W、第二 lookahead、第二结构/fallback 以及 decoder/full-network scheduler 全部明确禁止。

边界确实比 M474/M476r2/M498 大，但它仍是一个完整、可源码化的 product-capture island；不是授权新的总体调度系统，也不是允许第二套 Conv matcher。

## 3. 候选数字的合法口径

准入只继承 M528 r4 的 exact CPU same-ledger 候选：`435,293,339` cycles，相对 M468 strong-zero 为 `1.746753x`，相对 same-coordinate bit 为 `1.741232x`。物理 parent scratch 是 9 颗 `128x128b 1RW SP`，只用低 64 行；物理容量 18,432 B，候选 macro-rounded 总容量 213,376 B，低于 240 KiB。

这些数字只覆盖 H67 ep35 一条序列、10 samples、4 个 bottleneck Conv3x3，尚不是 RTL/VCS/PPA/energy/full-network/system-speedup 或 paper headline。

## 4. 两个非阻塞 P2

1. 准入通过 manifest、exact-SHA runner、`behavioral model not synthesized` 和未来 9-macro DC 条款间接闭合了宏绑定，但没有把 VCS behavioral binding 与 synthesis/Formality blackbox binding 写成两个命名工件。下一轮 source static hammer 必须强制：VCS filelist 绑定封存的 foundry `.v`；综合/Formality 绑定 generated `.db`/blackbox/cutpoints；不得把 behavior model 编进综合，也不得把 `64x1152` 推成寄存器阵列。
2. 新 matcher、ping-pong preprocess、scheduler、arbiter、macro adapter 与 completion 同时进入一个包，源码量较大。任何 placeholder/TODO、漏 ownership transition、未闭合 completion/conservation 或暗藏 fallback 都必须在 source static hammer 直接判 P1/P0，不能带病进 VCS。

## 5. 授权与下一步

只授权一个全新 author subagent，写入准入指定的 `rtl_m528_dw1rw`、`verif_m528_dw1rw`、`tb_m528_dw1rw`、M529 contract/runner 前缀和 author handoff 目录。作者必须交付完整 RTL、SVA、self-checking TB、未执行的 exact-SHA VCS runner、source-only contract、双封 author handoff 和下一轮独立 source-static-hammer 请求。

本准入及本锤审授权的运行数全部为 0。作者不得执行 runner；不得运行 VCS/DC/Formality/PTPX/CPU/GPU。完成后必须由新的独立 source-only static hammer 得到 P0=0/P1=0，root 才能另建 VCS launch admission。

`docs/359` 未修改，SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
