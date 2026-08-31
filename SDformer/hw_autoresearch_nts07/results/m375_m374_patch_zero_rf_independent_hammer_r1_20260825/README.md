# M375 independent hammer of M374

结论：**独立复算 0 mismatch，确认 G10 fast-kill；不写 RTL。** 综合评分 `96/100`，evidence quality `98/100`，hardware admission `12/100`，P0/P1/P2 = `0/4/4`。

M375 不导入 M374 analyzer。它重新 SHA 校验并解码全部 60 个 payload（645,120,000 packed bytes），先对 channel 做 Boolean OR，再显式 zero-pad 一圈并 OR 九个 stride-sampled 邻域，得到独立的 N=0 mask。全部 per-record scalar 与 1/2/4/8 tile 计数均与 M374 精确一致。

- 逐 timestep N=0：`7,257,197 / 40,320,000 = 17.9990005%`。
- whole-T N=0：`156 / 4,032,000 = 0.00386905%`。
- 三个 Conv→ATLIF 映射 whole-T N=0 均为 `0`。
- 仅有的 156 个位置全部在 `resblock0.conv2`，需要 residual read/add/commit。
- whole-T 对齐 tile：2×2=`12`，4×4=`1`，8×8=`0`。

独立复算的 oracle zero-commit、impossible all-scan/all-commit、perfect three-ATLIF deletion envelope 敏感性分别为 `1.001071×`、`1.016404×`、`1.038576×`，全部低于 1.15× gate。bit-sparse baseline 已对 N=0 发出零 source/weight-row，因此不能再计 MAC 或 weight/PWP DMA。

当前轴关闭。只有当 Patch normalization 经训练/valid825 变成 static 或 zero-preserving、新 population 在 ATLIF-mapped 路径产生足够 whole-T 空位置，并补齐 halo-aware metadata、动态 BN、residual/PED 和 address-timed transaction capture，且 executable upper bound 超过 1.15×，才允许重开。

本轮 CPU-only；未调用 GPU、Synopsys 或开源 RTL 工具；未修改 M374 producer 结果和 `docs/359`。
