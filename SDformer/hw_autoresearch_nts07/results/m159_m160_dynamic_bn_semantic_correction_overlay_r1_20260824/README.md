# M159/M160 动态 BN 语义纠正 overlay r1

## 裁决

冻结 H67 trace 的 BN 不是 checkpoint running-stat 推理。配置为 `test.bn_policy=no_running`，production profile 在 `build_model()` 后调用 `configure_batch_norm_evaluation()`，把 78 个 BN 改成 `track_running_stats=False`、`running_mean/var=None`，batch size 为 1。

M160 r1 只执行了 `build_model()` 和 `model.eval()`，漏掉上述 profile protocol 步骤，因此它审计的是“构造默认 running-stat BN”而非冻结推理。这是 P0；本 overlay 立即 fail-close。

## 保留与撤销

M159 以下算术仍保留：12 blocks、120 dynamic groups、Linear 159,784,111 cycles、FFN-local ATLIF 45,600,000 cycles、合计 205,384,111 cycles，占当前 compute envelope 33.1102933%。BN1/BN2/residual 的 element extents 也保留。

以下撤销：

- M159 的 resolved running-stat BN；
- FFN topology 中的 DropPath；它只在 attention residual；
- 删除 hidden channel 会删除 sn2 temporal parameters 的说法；`[T,T]` weight 和 `[T,1]` bias 跨 channel 共享；
- M160 r1 对冻结推理的静态 BN fold、静态 zero-path、`176,640 -> 17,904` bias storage 以及 437,760,000 BN elements/frame 静态 no-materialization 资格。

M160 r1 仅保留为明确限定的 constructor-default what-if，所有 cycle/system speedup、RTL、VCS 和 PPA 仍为 false。

## 正确的完整 FFN

`sn1 -> dropout1(p=0) -> fc1 -> BN1(current-batch) -> sn2 -> dropout2(p=0) -> fc2 -> BN2(current-batch) -> residual add`

正确的 hidden mask 原子单元为：`fc1 rows + BN1 hidden channels/moments + sn2 activation columns/lane state + fc2 columns`。共享 temporal parameters 保留。

## 新硬件方向

动态 BN 仍可与 rank-3 ATLIF 共存，但方法必须改成：

1. fc1 输出流过时同步累计每个 hidden channel 的 `sum/sumsq`；
2. 同时计算 rank-3 的右投影 `R*x`；
3. moment barrier 后，用动态 `alpha/beta` 修正 rank-3 state，再做左投影和 threshold；
4. 不物化 T=10 的 normalized BN1 张量，候选只保留 R=3 state；
5. BN2 则与 fc2 moment 累计和第二遍 residual commit 联合。

这条路理论上把 BN1 中间态从 T-wide 变成 R-wide，但在 PAFT rank-3 数值、SRAM 地址、barrier、端口 recurrence 和 VCS 完成前，不接纳 storage/cycle speedup。
