# 全 native 权重驻留：固定 W8 实测

**同一新 W8 函数的完整后段服务减少 3.4507%，外读减少 198,982,272 B。** 收益来自普通量化与驻留布局，作为后续融合的更强底座；不是新的标题机制，也不是 RTL/PPA。

本项用旧 stage ordinary R24+onepass 的真实 projection gate、连续 PED 和原 native W。只对 native W 做固定逐输出行 dyadic INT8。96 行实际均选中 scale=2^-9，q 范围 [-121,96]，最大行绝对整数和 16,926；所以完整 K 的整数累加低于 FP32 精确整数范围。新函数需要单独评价精度，未继承旧 valid825。

| 完整后段，ready | 展开 FP32 W8＋两块目录复用 | 实际 packed W8 全权驻留 |
|---|---:|---:|
| native＋onepass BN＋融合 PED 服务槽 | 427,507,285 | 412,755,104 |
| native 服务槽 | 227,115,624 | 212,363,443 |
| 外部读取 B | 420,360,128 | 221,377,856 |
| 外部写出 B | 147,456,000 | 147,456,000 |
| 权重冷填/重填 B | 199,065,600 | 83,328 |
| coefficient 实际读取 B | 426,912,512 | 106,728,832 |
| state 实际读取/写入 B | 410,873,976 / 378,417,192 | 相同 |

两轴都沿同一个 Engine，从原全图外部 gate 构造两个完整 4×4 输出块的 K864 目录，实际执行 native、T-major 双 stripe/256 树的单遍 BN，再读取真实 PED 并完成最终 ADD。三个 H32 顺序执行，没有偷偷增加并行 MAC。两块目录和门/header/output 高水位仍为 126,304 B；coefficient 容量仍 128 KiB。展开控制每个源块轮流真实装三份 H32 浮点权重；compact 将 82,944 B code 和 384 B scale 一次冷填到同一 coefficient SRAM。

每个 CR256 字包含 H32 的 32 个 code，按八个物理 lane 各四码排列。RF93/94 各保存每 lane 两码的 uint16；加载后各 H8 只在同 lane 内取低/高 byte 并 sign extend 到 RF80。没有任意跨 lane gather：两个 packet 半字解码和四个 H8 解码分别占单独 issue 槽，另有真实 LOAD、RF 写回和依赖等待。普通固定常量合并将统一 scale 保留在 RF81，但 96 个原 scale 的冷填仍全付。RF0–79 是原 P2×T10×H32 累加；目录阶段原 RF64–72 H4 缓存也完整保留。

搬运减少省下 37,309,169 槽，实际解包与载入比展开计算多 13,340,988 槽，末尾 2,304,000 次 scale MUL 与写回多 9,216,000 槽，净省 **14,752,181 槽**。因此不能只引用权重字节减少来暗示整链同倍率加速。实际端口等待计数的 0 仅指原 counter 覆盖的端口/写回日历冲突，RF RAW 等待已计总时间。

完整 18,432,000 个 native 输出对独立升序 K、新 W8 参考逐位零差。两布局之间还逐位比较了原始输出、480 个统计量、完整 BN 张量和两种最终后缀输出，全部零差。新 W8 与旧 GPU 原 W 的 raw 最大差约 0.02505，BN/join 最大差约 0.17325，包含真实量化改变；不得当作仅浮点归约差。唯一同父ordinary diverse10已完成，AEE1.149276778、九帧1.125297062，过同范围NB0；新825未测。[质量与捕获](../../../algorithm/native_w8_aee/README.md)。

A：dyadic 量化、普通 Gustav/LoopTree 分块与打包格式。B：两块目录复用被反复搬入三份 H32 权重压住。当前可执行增量是把同一 W8 函数放进更便宜的完整驻留时间线；它是共同强对照，并未提出普通底座之外的新 X。解包缓存或其他合法遍历仍有未试接口，不主张全局最优。

结果在 [summary.json](summary.json)，两个原始计费表在 [expanded_results.json](expanded_results.json) 与 [compact_results.json](compact_results.json)。参数和准确模块名为 [native_w8.npz](native_w8.npz)、[deployment.json](deployment.json)。使用既有 Python 3.12 环境依次执行 `run.py expanded --capture 完整捕获路径`、`run.py compact --capture 完整捕获路径`、`summarize.py`；默认是已准备的 `/tmp/native_bn_join_20260912/capture/ordinary.npz`。生成源和 C++ 源保存在本目录，旧树没有修改。

最新实际新W8 GPU首帧核对：两个输出窗口native/BN/PED共92,160值及全域480统计量，和CPU compact结果全部0 bit差，[gpu_alignment.json](gpu_alignment.json)。上文对旧原W捕获的非零差仍属于另一比较，不能替换新W8对齐结果；尚未捕获全部GPU raw。
