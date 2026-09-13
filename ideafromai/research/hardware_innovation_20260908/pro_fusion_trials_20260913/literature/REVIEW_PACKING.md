# 共同活动配对执行的独立审阅

2026-09-13。静态检查 [pair_execution.sv](../packing/pair_execution.sv)、[tb.cpp](../packing/tb.cpp)、[run.py](../packing/run.py)、[results.json](../packing/results.json)，随后检查 [run_r0.py](../packing/run_r0.py)、[r0_results.json](../packing/r0_results.json)。未重跑 Verilator/RTL/EDA/训练；只对现有 JSON 做计数与公式核对，及只读检查 r0 NPZ 的输入类型/数值范围。没有改动 packing 作者文件。

**结论：接受“固定端口抽象下的精确整数线性叶执行对照”，不接受“完整 ELSA/Phi 迁移”“真实 SRAM bank 冲突回放”或“共同活动配对产生新收益”。** mode 2 是必要的强普通控制，代码没有凭空增加第 21 个加法器。当前共同活动配对不如相邻配对，主要原因可以由已经报告的词读取和 issue 数量精确解释。

## 1. 优先结论

| 优先级 | 发现 | 应如何报告/处理 |
|---|---|---|
| 高 | `replay_cycles` 表示同一 psum lane 同时需要 a/b 后的第二次加法；没有 W-bank 冲突队列 | 改报告术语为 lane 冲突第二次 issue；不要据此宣称已兑现 bank replay |
| 高 | `GET_MASK` 每次任意读取两个 `source_mem` 地址，同时访问映射表；原生输入预先在 CPU 展开 | 当前合同是两路源掩码读取；具体 SRAM bank/多端口/复制及 native im2col 尚未支付 |
| 高 | mode 2 的两源部分和在既有加法器空闲阶段形成；它持续优于 mode 1 | 必须把 mode 2 保留为普通强 B；mode 1 相对 mode 0 的少量收益不能算新 X |
| 高 | `coactivity` 实际调用 `pairs_greedy(...,'conflict')`，优先降低 AND 的 popcount | 它是避碰目标，不是提高共同活动复用的目标；须按当前物理成本重新定义优化目标 |
| 中 | r0 扩展已覆盖全 K864/N96/T10，但只有固定 16 个 P2 tile | 可以称完整维度的线性叶；不能称全图/完整 r0、norm/residual/PSN 或任务部署精度通过 |
| 中 | Phi 已有多窗口容量/psum-bank 检测、flush、元数据与可重配归约 | 进一步加入这些模块应明确作为完整借 A，不直接作为新颖性 |

## 2. 三个 mode 是否公平、是否正确

mode 0 在同一对两个源都存在任意活动时，先整向量处理 a，再整向量处理 b。mode 1 每个 lane 先选 a，否则选 b；只有 `mask_a & mask_b` 非空才做第二次 issue，第二次只处理真正相交的 lane。mode 2 使用四种二值组合对应的 `0, wa, wb, wa+wb`，每对只需一次向量提交。三者实现相同的整数点积函数；存储的 W 参数没有按 Column Combining 方式合并删除，mode 2 只形成临时的两系数和。

SV 的 `RESPONSE` 状态将现有 `add_a[0]/add_b[0]` 切到两个返回系数，`pair_sum<=add_y[0]`；这时没有 psum 提交，下一状态才执行向量更新。因此 **mode 2 没有隐藏额外的第 21 个加法器，也没有从 TB 取得 gold 部分和**。它使用系数返回周期的既有算术空档，属于有效的普通两源预归约/二位查表恒等式。数据到达组合路径、额外输入 mux 和 `pair_sum` 寄存器仍存在，不能由“加法器数量一样”推导相同频率/能耗。

所有 mode 是同一运行时配置 RTL，拥有相同 source/map 存储、20 个 ACC 加法器、20 个 psum 寄存器、单个 32-bit W 接口和共同 FSM。这支持**此实现中的同资源逻辑对照**。它不证明各模式独立优化到相同面积/时序后的优劣。尤其当前 stop-and-wait W 流程留下足够的空闲加法阶段；若普通基线进一步流水化系数读取/issue，也必须让三者在新流水线下重新对照。

mode 0 是普通整向量 event AAC 控制，并不是最强的 event 引擎。RTL 本身未实现 2:4 权重 metadata、权重零检测/跳过或 native source packing，因此不能仅因输入路径包含 `ordinary` 就声称它是完整普通 2:4 控制。mode 2 足够揭示“只降低 lane 碰撞”这一新方案目标的不足，但并未替代未来真正的 2:4/DBB 控制。

## 3. 端口与存储边界

- **外部 W 端口确实受限。** 每次只请求一个 32-bit 词，包含两个 signed16 系数，TB 限制一个未完成请求，并注入 ready/response 延迟。三种配对均将原始两个独立系数离线写到同词，不是将系数加在一起。任意重排带来的物理 W 词布局是合同的一部分，不能移回原布局后仍按一个词读取。
- **内部源端口未物理落定。** `source_mem` 是 `864×20 bit`，约 2160 B；`map_mem` 是 `432×20 bit`，约 1080 B。`GET_MASK` 需要任意双地址源读取，代码没有 source-bank 占用、端口仲裁或冲突回放。它可用寄存器/mux 或真正双读口实现，但成本不能假定等同一个普通 SRAM。相邻配对未来可能用相邻宽词读到两源，而任意配对需要双地址；这一差别在当前抽象里被统一成双读口。
- **源词加载全部付了，但 native gather 没付。** 每 tile 加载 864 个 20-bit 已展开词，零词也加载。P-major/T-minor 和 C,dy,dx 顺序由 Python 准备；重复 halo 读取、原生 bit 词布局、对齐、line buffer、读-改-写、排列网络均在 RTL 外。
- **map 装载在计时外。** TB 在 start 前写 432 项。静态配对可随模型加载后跨 tile 摊销，因此计时外并非天然错误，但需注明冷启动排除与额外 map 容量；当前每个测试实例实际都会重新装载。W 重新排布同样是离线预处理，不是运行时零成本证明。
- **psum 没有 SRAM bank 冲突。** 20 项 accum 是各 lane 的寄存器，每 lane 每 issue 一次更新；`ISSUE_SECOND` 是单 lane 仅有一次加法资源的冲突处理。它尚未包含 ELSA 多目的队列或 Phi 的多行/多 bank psum 打包。

当前三种配对共享这些限制，内部趋势可解释；把它外推成源 SRAM 读节省、bank 端口改进或能耗结论则超出证据。

## 4. 原 N16 叶的数字为什么自洽

`results.json` 的 30 次运行各检查 320 个输出，总计 9600 次输出比较；这是同一真实 P2 tile 加 all-zero/all-one 控制，在配对/mode/stall 维度重复检查，**不是 9600 个独立真实样本**。TB 检查输出顺序、逐元素数值、输入接受数和完成条件，包含 source/request/output backpressure；未发现由这些已读路径导致的明显函数错误。

无 stall 时，固定开销为 `C0=21936`，且全部运行符合：

`cycles = C0 + 3 × weight_words + issue_cycles`。

3 来自当前请求/响应状态及 TB 延迟合同；不是所有实际 SRAM 都应采用的普适系数。

| 原 N16 真实 tile、无 stall | cycles | W 词 | issue | 第二次 issue |
|---|---:|---:|---:|---:|
| adjacent / mode 0 | 36608 | 3264 | 4880 | 1616 |
| adjacent / mode 1 | 36464 | 3264 | 4736 | 1472 |
| adjacent / mode 2 | 34992 | 3264 | 3264 | 0 |
| coactivity / mode 1 | 39568 | 4352 | 4576 | 224 |

mode 1 相对 mode 0 仅少 144 周期（0.393%）。mode 2 再少 1472 周期（相对 mode 1 减少 4.04%）。避碰配对将第二次 issue 降低了 1248 次，却增加了 1088 个 W 词，并多出首轮 issue；最终比 adjacent/mode 1 多 3104 周期。不是“没有减少碰撞”，而是减少了错误成本的碰撞。

固定空扫描本身有 20736 个周期，即 `16×432×3`；全零也仍扫全部 pair。将这类扫描压缩或复用 mask/map 可带来工程收益，但这是另一个需要完整普通控制的数据流改动，不能与配对目标混成一个 X。

## 5. r0 全维度扩展：确认的新增范围

只读检查捕获 NPZ：输入 `float32 [10,64,864]` 仅含 `{0,1}`，因此 `astype(int64)` 没有丢弃连续幅值。权重是 `float32 [96,96,3,3]`。新导出将权重乘 `2^16` 后 round-to-even，实际 signed16 范围 `[-13925,16243]`；最保守 raw sum 幅度上界 `864×16243=14033952`，在 signed32 内。因此该批数据没有系数截断或累加溢出的显见问题。

新运行是同一捕获的前 32 个位置生成静态配对，后 32 个位置构成 16 个 P2 tile；中心索引集合无交集。每 tile 完整覆盖 K864、N96、T10，检查 1920 个输出；`288×1920=552960` 次比较与 JSON 一致。独特的真实整数输出只有 `32×10×96=30720` 项，三配对/三 mode/两 stall 重复覆盖。中心留出不等于独立序列验证，且没有证明两个集合的感受野完全不相交。

`local_quant_output_nrmse=4.6158e-5` 比较的是该叶上新 Q16 点积与 FP32 权重同输入点积；它不是全图 norm/residual/flow 输出误差，更不是 NB0/AEE 通过。run_r0.py 已正确标记 diagnostic export、no inherited AEE，最终总结应保持这个限定。

| r0 16 个 P2 tile、无 stall 合计 | mode 0 周期 | mode 1 周期 | mode 2 周期 | W 词（各 mode 相同） |
|---|---:|---:|---:|---:|
| adjacent | 3439296 | 3400896 | 3331584 | 323712 |
| coactivity | 3530304 | 3475200 | 3452928 | 354048 |
| union | 3521376 | 3477504 | 3441024 | 351072 |

全部合计精确符合 `cycles=2036736+3×weight_words+issue`，公式残差均为 0。coactivity 相对 adjacent 的 active pair 总数增加 316，对应多读 `316×96=30336` 个 W 词；相交 pair 从 722 降到 232，对应少做 `490×96=47040` 次第二次 issue。mode 1 的首轮 issue 同时增加 30336，故总 issue 只少 16704；最终周期差为 `3×30336−16704=74304`，正好匹配结果。mode 2 根本不需要碰撞重放，coactivity 只剩 W 词增量，差为 `4×30336=121344`。

扩展支持的结论是：**在本次同帧留出位置、预展开源词和单 W 词返回合同下，完整 K/N/T 的普通两源 product-sum 优于逐 lane 配对重放；这两个离线贪心目标都未优于相邻配对。**它不支持“共同活动配对家族失败”，也不支持“全模型获益”。

## 6. 为什么离线 coactivity 配对更差

`pairs_greedy` 先取活动最多的源，再以 `(popcount(a&b), popcount(a|b), index)` 选搭档，优先避免活动相交。当前模式的主要费用却是**20-bit mask 的整词 OR 是否为零**：一个 pair 任一 bit 存活就读整 32-bit W 词。原相邻 K 含有空间/通道局部性，可能把多个活跃源放在同词，并留下更多全零词；拆开这些源能减少交集，却使更多词不得不请求。

训练尺度也不完全匹配：旧版本 calibration 是 14 个位置上的长 bitmap，新版本是 32 位置×T10；真实硬件每次只处理一个 P2×T10 小包。长 bitmap 的总 bit popcount 与每个 P2 tile 的 `bool(OR)`、尾部 replay 成本并非同一个量。`union` 贪心虽偏向更小 OR popcount，仍没有直接优化真实 P2 词交易数，且本轮未赢。因此不应仅改命名或再微调同一个权重参数便宣称找到新机制。

更合适的候选目标由固定硬件给定：mode 1 在此无 stall 合同下最小化 `4A+C`，其中 A 是各 P2 tile 的活跃 pair 数，C 是相交 pair 数；mode 2 最小化 `A`。加入真实原生 source-bank 后，还要加读词、双地址冲突、对齐/halo 和重排成本。只在固定全部这些硬件后改变学习目标，才能测试潜在 X。

## 7. 更强的 Phi 先验及下一步边界

精读本地原文 `survey_ab_fusion_20260910/p0_txts/2505.10909.txt` 570–650 行，对应 Phi §4.2–4.3：多窗口 packer 同时检查容量与 psum-bank 冲突，无窗口能放入时 flush 最满窗口；L2 pack 带多行 metadata，经可配置 8 输入归约树与 psum crossbar 写回。这比 ELSA 的队列容量处理更直接覆盖“有限窗口＋bank 冲突感知打包”。[Phi 作者原文](https://arxiv.org/pdf/2505.10909)

因此未来实现多窗口与 bank 检查，应完整借入 Phi 的 pack 元数据、仲裁、flush、行分隔归约和写回，而不是只拿一个 conflict predicate；ELSA 的目的分组/队列、TensorDash 的受限前移和 Bishop 的训练/特征对齐也都是强 A。当前二源 product-sum 仅覆盖其中极小的精确归约叶，不是上述整机复现。

建议下一工单先补原生 source-word 到该叶的实际读口与 im2col，并保留 mode 2；把 replay 计数拆成 lane 二次 issue、source-bank stall、W-bank stall、queue drain。使用同一 RTL、相同存储与端口，比较相邻、普通启发式和真实事务目标的分组。若后一目标仍不能胜过相邻＋mode 2，停止这个具体 X，保留已完成的普通强 A 与功能结果。
