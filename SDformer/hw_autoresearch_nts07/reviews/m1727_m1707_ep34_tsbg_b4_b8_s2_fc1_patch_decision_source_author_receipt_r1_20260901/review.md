# M1727 additive successor 作者源码回执

M1727 不修改或覆盖 M1721，而是精确绑定其失败源码并替换 M1725 的两项 P1 与两项 P2 边界。生产 `--run-analysis` 在访问 capture 前必须依次通过 M1727 合同双封、未来 M1728 不同作者目录双封和未来 M1729 one-shot release 双封；当前三者中 review/release 尚不存在，因此分析与 capture verification 均未获授权。

S2 的 `sum_abs_output_code_debt` 已按每个 beta 所覆盖的实际输出通道数累计。定向反例 `1 dropped unit × 32 outputs × |w|=1` 从错误的 2 修正为 32；17-output tail-block 测试得到 `16×1 + 1×2 = 18`，epsilon=0 仍完全旁路。

TSBG 保留 ordinary persistent same-B LRU 公平基线与 fetch/compute/commit/schedule/roofline 分列。B4/B8 的 Acc24 context 和 int8 source FIFO 字节下界现已显式输出；context tag、broadcast control、物理面积/能量尚未定价，且 4-byte weight 没有硬件量化 authority，因此任何未来比值只能是 diagnostic screening，所有 paper admission 强制为 false。

CPython 3.6 与 3.10 定向测试均为 16/16 PASS；3.12 无 NumPy而跳过向量测试，但三版本 source-self-check 字节一致。未读取 M1707 capture，未运行 analyzer/GPU/RTL/EDA，未创建 release/result，未改 docs/359。
