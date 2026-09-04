# GateStack IPD 首次 Miss 自动回填与后续 Tile 驻留

## 1. 解决的缺口

此前 descriptor cache 只能由测试平台或上游显式预填。若一个 head 首次 lookup miss，它会在每个 output tile 都重复走 IPD，跨 tile residency 只对预填 head 生效。本轮把首次 IPD 解码结果直接写回 cache，使后续 tile 自动走 resident。

## 2. 为什么不增加第二解析器

IPD decoder 已按以下顺序工作：

1. 读取并校验 header0/header1；
2. 将 descriptor pair 装入内部已有 buffer；
3. 按 term 顺序回放 descriptor 与 token events。

因此只需新增两个旁路接口：

- header1 后的 `fill_begin(tag, term_count)`；
- term 回放时的 descriptor rendezvous fork。

这比从 slot word 旁路再造一个 IPD parser 更省控制、更容易保持 bit-exact，也避免新增 `MAX_TERMS` 级 descriptor 副本。

## 3. 超深 Head 的无损 Bypass

若 `term_count > RESIDENT_TERMS`，descriptor cache 在 begin 握手时返回 non-cacheable。`gatestack_ipd_cache_fill_adapter` 随即进入 bypass 状态：term 仍被 projection 消费，但不写 cache，直到最后一个 term 后正常退休。这保证大 head 不会因 cache entry ready 永远为 0 而死锁。

## 4. 验证证据

- IPD decoder 原有正常、空 head、malformed drain 全部通过；
- 真实三 decoder 小尺度仍为 79 cycles，T162 仍为 529 cycles；
- fill adapter 的 cacheable/bypass/empty/反压通过；
- full top 从 2 hit/2 miss 变为 3 hit/1 miss；
- 第二 tile 的原 IPD head 已由 control PLAN 自动选择 resident route；
- 16 个最终 token 数值不变；
- memory bits 不变，Yosys generic control 增加 61。

详细数字见 `results/gatestack_ipd_autofill_20260716/report.md`。

## 5. 论文可用的表述边界

可以写：提出 decode-once、跨 output tile 驻留的 descriptor promotion 数据流，使用 header-triggered fill 和 term-stream rendezvous fork，在不复制解析器和 descriptor buffer 的前提下把首次 miss 提升为后续 hit。

不能写：已经证明 H67 端到端加速或节能。当前只有小型 RTL TB 的 2-cycle 改善；真实收益必须由 H67 ordered trace 和 DC/SAIF 给出。
