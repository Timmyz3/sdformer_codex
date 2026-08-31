# M158 r2 独立打铁评审

结论：**r2 的 signed tuple 和四个权重 `sum(abs)` 数值正确；Acc19 只接纳为条件式数学候选，暂不接纳从集成 RTL 删除 runtime overflow tree。**

评分：75/100，P0=1、P1=3、P2=2。`headline=false`、`physical_speedup=false`、`system_speedup=false`。

## fresh 重算结果

评审器没有导入 M158 主脚本，也没有调用 M150 `select_events`。它直接使用 M40 packed heldout support、M72 centers 和四个 exact-SHA M41 INT8 权重，独立完成 im2col、PWP width catalog、center/eligibility 选择、destination event/sign 和 low/high half 重构。

- 20 records，414,720,000 source keys，23,522,595 active keys。
- 188,148,490 events：170,591,133 positive、**17,557,357 negative**。
- `negative_not_event`、event half、negative half 全部 0 mismatch。
- 逐 destination ordered-half digest：`dac2e8f4848bf9e41ccda37e1ab4bffd782575fa0431053d336197426aa3ef15`。
- r1 correction overlay 的撤销状态与 5 条 r2 manifest SHA 均通过；`docs/359` 仍为 `dedde7ce...`。

四个 `I_KY_KX_O_C_ORDER` payload 的每通道 `sum(abs)` 最大值 fresh 重算为 218,338、204,866、207,239、190,753；Acc19 正上限 262,143，条件式最坏余量 43,805。四层均无 Conv bias，且 payload 中无 `-128`。

完整逐 record 数值、payload channel 和 manifest 身份见 [independent_recompute_and_attack.json](independent_recompute_and_attack.json)。

## P0：为什么还不能删 overflow tree

`sum(abs(weights))` 是充分界的真正前提不是“feature 只出现一次”，而是：对每个输出 accumulator、每个 convolution feature，任意已接受前缀的系数必须保持在 `[-1,1]`。

PWP 的 `center=1,target=0` 会交付同一 weight 两次：一次 `+1` anchor，一次 `-1` correction。只要两次都是 exact-once，不论顺序是 `[+1,-1]` 还是 `[-1,+1]`，前缀系数最大绝对值仍为 1，Acc19 数学界成立。但若 stall/reset/stale replay/cache alias 使某个事务重复，系数可到 2，保守界变成 436,676，超出 Acc19，需要 signed20。

当前 M115r2 仍为：

- `integrated_accepted_transaction_exact_once_miter=false`
- `signed19_accumulator_rtl=false`

所以 r2 的 `source_major_integer_reorder_exactness=true` 和 `runtime_overflow_detector_required_for_frozen_domain=false` 只能降级为设计候选，不能作为硬件 admission。

## 其他边界

- Bias：已闭合，四个 Conv2d 都是 bias-free、zero-init。
- 相同数值的重复 weight：不是问题；不同 feature occurrence 已分别计入 `sum(abs)`。
- 重复 accepted transaction：是问题，会破坏系数界。
- BN/late scale/residual：M41 没有接纳动态 BN 和 runtime scale；除非重新加入 bound，必须明确放在 Acc19 raw-Conv 边界之外。
- exact-SHA：目前只是证据身份，没有 build/boot/RTL 的 payload 身份检查和 guarded/wider fallback。
- half split：软件数据级全量证明成立，但 `rtl_trace_miter=false`。

## 删除树的硬门

只有以下条件全部闭合才能删：集成 cache/PWP/correction/Acc19 的 VCS/SVA exact-once coefficient miter；stall/reset/stale/replay/cache-alias attacks；payload/operator 身份 fail-closed；raw-Conv-only zero-init 类型边界；signed19 RTL trace miter；非冻结域恢复 overflow guard 或至少 signed20 fallback。

结构化结论、P0/P1/P2 和允许/禁止措辞见 [m158_independent_hammer_review_r2.json](m158_independent_hammer_review_r2.json)。本评审未修改 production、contracts、原结果或 `docs/359`。
