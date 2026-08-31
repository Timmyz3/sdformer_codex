# M382：M381 q32/O4 burst-streaming 独立打铁

结论：**92/100，P0/P1/P2 = 0/0/5**。M381 三个输出 fresh
exact-SHA replay 均 byte-identical；从冻结 M248 packed trace 与 q32 catalog
独立重建 17,280 phase 后，population、used-center bitmap/run、28 个主 sweep
点与 6 个 blocking 点全部零 mismatch。因此仅放行有界 streaming
active-descriptor controller 进入 VCS。

独立重建确认：

- `51,840,000 = 30,368,111 zero + 21,471,889 active`；
- `21,471,889 = 12,709,384 PWP + 8,762,505 fallback`，其中
  6,762,595 个 popcount-1 fallback 全保留；
- only-original-zero elision、lowest-ID tie 与 signed residual 重建均零错误；
- 平均 used center/run 为 `31.396354 / 1.472396`，最大为 `32 / 10`；
- `cmd32, SRAM L8, II1` 为 `505,195,832 cycles / 1.0763829x`；
- 1.05x 的 cmd32/L8 blocking 边界仅为 `0.2955918 cycle/descriptor`。

物理攻击未发现 P0/P1。存在一个 32-byte 对齐、无重叠、无需 center remap
的 64-KiB direct-address layout；credit-aware L8/D8 模拟在最多 2,400 个
descriptor 时无 post-start underflow。但地址/run 命令、单周期 seal、1RW
credit/epoch 与 tile1 overlap 尚未由 RTL 证明，必须由下一步 VCS cycle miter
与协议攻击关门。该 `1.0763829x` 仍是四个 bottleneck Conv 的模块/冻结 cohort
数字，不是 system speedup、PPA、energy 或 DATE headline。

主要证据：

- `m382_independent_recompute_r1.json`
- `m382_m381_independent_hammer_review_r1.json`
- `fresh_exact_sha_replay/`
- `recompute_m382_independent.py`

M381 被审对象与 `docs/359` 未修改。
