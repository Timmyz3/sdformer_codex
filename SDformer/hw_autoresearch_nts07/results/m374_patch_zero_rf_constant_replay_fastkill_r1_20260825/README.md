# M374 Patch N=0 receptive-field / constant-replay fast-kill

M374 只做 CPU、只读证据复算。它对冻结 M51 的 60 个 payload（10 samples × 6 个 exact-binary Patch/early Conv3x3）逐输出位置显式展开 `padding=1` 的 3×3 receptive field，并把逐 timestep 空邻域、完整 T=10 空空间位置和对齐 1×1/2×2/4×4/8×8 空 tile 分开统计。

## 结论

- 40,320,000 个 output token 中，`7,257,197` 个 N=0，比例 `17.9990005%`。
- 但 bit-sparse 强基线对这些 token 本来就不发 source：1,774,268,587 个 source contribution 与 M222/M272 完全守恒。因此新增 MAC、active weight-row 和 PWP/weight DMA 节省均为 `0`，不能重复计数。
- 跨完整 T=10 仍为空的空间位置只有 `156 / 4,032,000 = 0.003869%`。156 个全部位于 `resblocks.0.conv2.0`，其后必须读取 identity、做 residual add 并提交；映射到后续 ATLIF 的三个 Conv 全部为 `0` 个 whole-T 空位置。
- 对齐 whole-T 空 tile 只有 12 个 2×2、1 个 4×4、0 个 8×8，且仍全部落在 residual conv2 路径。

冻结推理是 `no_running/current-batch BN`。N=0 Conv 输出虽然因六层均无 bias 而精确为零，但 BN 后常数为每样本 moment 决定的动态值，不是 checkpoint-static 值。`conv2` 后的 residual add 和 `proj` 的 PED shortcut 也使“零卷积分支”等价于“静态输出/可丢 commit”不成立。M273r2 对 N=0 release 的语义仍是 illegal sticky fault，不是可复用的 exact empty-tile response。

## 上界

- 假设不合法地删除所有 measured zero-token commit，isolated M272 仅 `1.003867×`，620.3M envelope 敏感性 `1.001071×`。
- 假设把六层所有 scan 和所有 commit 都凭空删除，isolated 也只有 `1.061681×`，envelope `1.016404×`。
- 假设把三个映射 ATLIF 模块整个删除，envelope ceiling 也只有 `1.038576×`；真实 measured reusable ATLIF population 为零。

因此 G10 checkpoint-static replay 与 dynamic constant-broadcast RTL 均在 1.15× gate 前 fast-kill，不写 RTL、不跑 N>0 accuracy。若将来重开，最小新 capture 已写入结果 JSON：补两个非二值 Patch Conv 的 `input!=0` mask、动态 BN moments/post-BN miter、whole-T ATLIF 输出、residual/PED commit，以及 halo-aware address-timed metadata/transaction trace。

本轮未调用 GPU、Synopsys 或开源 RTL 工具，未修改 `docs/359`，不得表述为 hardware/system speedup。
