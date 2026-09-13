# 原生 r0 固定资源数据流

已完成真实八 tile、三调度、三 mask 的 **288 次 RTL 运行/1,105,920 个输出零差**；独立事务账本精确闭合。主要周期及边界见 [REPORT.md](REPORT.md)，机器结果见 [real_summary.json](real_summary.json)。含每新 tile 装入后，源优先 dense/physical25/magnitude25 为 480,384/391,288/379,099 cycles；物理费用 mask 尚未胜过普通幅值控制。只写本目录；不运行模型、训练或 EDA。

`native_sparse.sv` 内部从原生 `(c,y,x)` 的 10bit 时间词生成全部 3×3 地址和合法目的，完整 C96/N96/T10，4×4 输入、2×2 输出。三个运行时模式共享 [资源合同](resource_contract.json)：输出优先、原生源优先、权重优先。八个 32bit 数据加法器分时完成 WA+WB 和部分和更新；权重优先跨四空间消费者保留 WA/WB/sum。三种模式都在读取前关闭完整无需求权重词、只对实际共同活动形成 sum，并在静态 C4 块已删除时跳过对应循环段。

TB 配置的是原生源词、原 W、288bit 结构 mask。origin 指定全图输入起点，SV 在单源口前计算 240×320 边界、置零并取消越界读；毒值控制保证正确性不依赖 TB 先填零。没有预 im2col、模式神谕、部分和或实际跳过结果输入。所有部分和在八个有限 bank 内保持，清零、读改写、控制、背压和最后 480 个输出 beat 均在 RTL。输入 native T-word 布局作为本模块入口；其上游 PSN 与时间词生产不在本模块。

线性整数函数为 `Σ S(t,c,y+ky,x+kx) × Wq(n,c,ky,kx) × M(n//8,c//4)`，输出 signed32，尚未接原 norm/residual。权重 Q16 是诊断整数接口，不据此继承 FP32 AEE。所有 signed16 权重的绝对最坏累加不超过 `864×32768=28,311,552`，signed32 足够；WA+WB 用 signed17。

配置/装入共 12,193 拍，单独报告并可加到执行周期：source 1,536、W 10,368、mask 288、origin 1。每个新 tile 还需 source+origin 的 1,537 拍装入；第二次不 reset 命令仅是功能检查。静态 W/mask 可跨 tile 摊销；报告不给免费首次装入。核心计时从 start 后清零到最后结果接受及 done；不将三种调度的槽数或不同核的周期直接相加。

运行功能控制：

```bash
/opt/anaconda3/bin/python3.12 run.py
```

这调用 Verilator 4.028 的 `--cc --exe`，再独立 make。固定零、全一、原生角点/部分掩码、全掩码、越界非零毒值五个控制各跑三模式、两种端口/输出压力、两个不 reset 的连续命令，共 60 次、230,400 个输出。它们验证边界、符号、清零和握手，不能当真实数据性能。原始结果为 [control_results.json](control_results.json)。

实现采用 ELSA/Gustavson 式按目的累计与常规 loop interchange/部分和复用的适用部分；固定 bank、持久 psum、队列或重排本身不申领 X。本版无 bank 冲突消除主张，也不冒充 ELSA/Phi 全系统复现。其原文与完整借入边界见 [前轮点名文献](../../pro_fusion_trials_20260913/NAMED_LITERATURE.md)。新结构目标只有在超过相同硬件上的最佳普通模式、同质量控制后才有增量证据。
