独立审阅结论：`r0_transfer/shift14_request` 是已经接到当前大算子、且通过十帧 NB0 的有效接口；当前证据没有支持它成为优于普通 2:4 的主候选。值得保留的增量是“用真实源活动和 H8 系数响应费用选择例外坐标”，而不是非零 fill、等权合并或 popcount 本身。

本审阅未修改 sparse_owned。实际读过 [r0 transfer](../sparse_owned/r0_transfer/transfer.py) 的拟合、选择、序列化、打包执行和全部结果，以及 [response_execution.py](../sparse_owned/response_execution.py)、[fill_chain.py](../sparse_owned/fill_chain.py)。整网质量取 [root 的 32 组 AEE](../root_owned/aee_all.md)，不能拿本地 `AEE=False` 字段否定 parent 已完成的前向，也不能把先前 r1 的完整整数消费者服务自动转记到 r0。

| 维度，0–10 分 | 当前分数 | 依据 |
|---|---:|---|
| 新颖性 | 4 | fill、重复权重求和、运行时稀疏遍历均有强先验。剩余增量有精确挂点，但尚未与完整的现有编译器在同接口下对照。 |
| 实际适配 | 8 | 当前 96×96×3×3 r0 Conv2、真实首帧源、真实 FP32 字节流、最小 metadata、同权限普通控制和独立十帧整网执行均已完成。 |
| 性能证据 | 4 | 已出现真实取消的 CR256 请求，并有组内因果控制；r0 当前仍是请求/issue 计数，未闭合完整服务且被普通 2:4 同时占优。 |

这些是当前证据强度评分，不是投稿接受概率。

实际机制为每个自然平铺 K4 保存一个例外 s 和两个系数 c、v：`c * (sum(g) - g_s) + v * g_s`。其三个共同权重的源计数在请求 c 之前排除了例外位。因此当例外脉冲单独出现时，c 的请求可以不发；这与先发 c*sum(g)、再发抵消残差的执行确有服务区别。代码只用当前源 operands 作请求判断，没有借未来系数值或下游 gold。该表示适用于静态 θg；不能直接套到连续 PSN/PED，旧整数机器也仍使用原 IMAC，并未证明乘法器消失。

先验核查不是只按标题归类：

| 已核 primary / 位置 | 已覆盖的 A，及尚未证明覆盖的部分 |
|---|---|
| [Finch，OOPSLA 2025 / 2024 预印本](https://arxiv.org/abs/2404.16730)，并核 [官方 tensor formats](https://finch-tensor.org/Finch.jl/stable/docs/tensor_formats/) 的 fill 参数及 Sparse/Run/Block 层 | 通用非零填充值、只存非 fill 项、结构与控制流共同编译已经存在。“稀疏不一定围绕零”不能作为新贡献。未在本项目完整迁入 Finch；不声称其现成后端已实现这里的 P2T10/CR256 请求取消。 |
| [SumMerge，ICS 2021 原文 §3.2–3.4、图 4/6](https://cwfletcher.github.io/content/research/2021.ics.summerge.paper.pdf) | 相同权重对应输入先求和，组内/跨 filter 公共子和、离线构图与运行时图执行已经存在。三个相同 fill 权重的组合求和属于这个强 A 的覆盖范围。其论文身份是 ICS 2021；本轮没有迁入完整图优化器。 |
| [Compiler Support for Sparse Tensor Convolutions，OOPSLA 2024，§2–3](https://ajroot.pl/publications/oopsla2024-spconv.pdf) | 稀疏卷积的复合下标、只遍历非空子区间、格式组合及循环次序均有系统方法。不能把 packet 中跳过零组或稀疏卷积本身申领为新概念；本文也不把该文夸成专用 nonzero-fill 求解器。 |

组合后的强 A 应允许：非零 fill + 例外表示、等权输入子和、源驱动跳过、输出分组与系数布局选择。不能只拿“先乘总和再抵消”的差执行作唯一对照。可检验的剩余 X 是：在真实 θg collector 已到达后，将例外删除融入发系数请求前的 operand 编码，并在同一 AEE 约束下用实际 H8 物理响应费用优化例外，从而取消普通组合编译仍发出的响应。当前尚未证明最后这个“仍发出”的差分，因此新颖性不应预支。

当前 r0 的结果更明确地限制了主张。下表服务仅为 `2*(value CR256 + metadata CR256) + SIMD8 issue + support decode`，不是整层周期：

| 接口 | 十帧 AEE | 首帧空间留出 relative squared error | value / metadata 请求 | 含 support decode 服务计数 |
|---|---:|---:|---:|---:|
| ordinary24 | 1.1591833472 | 0.05811859 | 43718 / 4014 | 231001 |
| ordinary34 | 1.1860722596 | 0.01203720 | 57108 / 2688 | 279730 |
| shift14 magnitude | 1.1777446144 | 0.02716068 | 54499 / 2688 | 294552 |
| shift14 Gram | 1.1949790382 | 0.02870204 | 54249 / 2688 | 293155 |
| shift14 request | 1.1762471512 | 0.02908496 | 52882 / 2688 | 284177 |

request 相对同函数族 Gram 减少约 3.06% 服务计数，是应保留的实际信号。它比 ordinary34 少值字节，且此十帧 AEE 更低，但服务计数仍多 1.59%；比 ordinary24 服务计数多 23.02%，AEE 也更高。ordinary24 与 shift14 均存两项 FP32 系数，前者 3-bit 支持、后者 2-bit 例外；最小 metadata 控制已给足。ordinary24 的局部误差较大却有更低 AEE，再次说明不能按局部 L2/PED 误差淘汰接口。上述十帧不是新的 valid825，也不作显著性声明。

公平性做对了三点：普通 2:4/3:4 获得相同 ridge Gram refit、同一次八 lane 坐标优化及同 10% 局部 SSE 余量；metadata 从实际最小位流读取，含响应跨界；打包执行逐值匹配有效重构核，且整网组合由 root 重新运行。magnitude→Gram→request 是有区分作用的控制，不宜合并为一张“稀疏方案更好”的卡。

当前尚未闭合的费用有明确位置：

1. `operand_counts()` 一次生成每个输出 lane 的 `pop-g_s`，`support_decode8lane_issues=1/nonempty group` 尚未证明八 lane 多路选择、计数减法和 token 遍历能在给定寄存器/端口中按该节拍完成。必须接相同的 collector/operand latch/地址生成执行；不能只在 Python 向量操作后数 products。
2. 源 packet 形成、空间 gather、源端口延迟、仲裁及输出写回均排除。当前 H8 外循环读每个源 packet 12 次，报告中明确列出 41472 次 source packet read、829440 次 nibble decode；下一步可优化这些共同服务，但所有普通控制要得到同样权限。
3. r0 的 `2*CR+issue+decode` 是固定加和代理，并非旧 r1 Machine 的 RAW/端口事件调度。没有实际 fp32→整数 θW 量化、RNE/溢出与全网定点复核，不能用 r1 已有定点检查为 r0 兜底。
4. ordinary34 应额外允许同值 dense-zero 存储/静态共同支持和响应复用控制。本组自己的 count 原型已经发现 indexed metadata 的费用可能吞掉 25% 系数节省。sparse 采用自然 flat K4，本组 count 采用同 tap 四通道；两种 pack/gather 不同，不能把服务比分母混用。

建议本阶段把 ordinary24 保留为通过质量门槛的强 A；把 shift14_request 保留为已做过的 X 原型与明确的 3.06% 组内信号。后续优先问题是同值、同源格式、同端口的编译与执行差分；现有结果不足以宣称非零 fill 方向已被否定，也不足以将它置于普通 2:4 之前。
