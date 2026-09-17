# A+X 筛选（2026-09-14 起）

按用户 2026-09-14 指示执行的筛选目录：**先照抄 A 套用（有开源代码直接用，没有就写 RTL），效果好→找针对性创新点；效果不好→定位问题→针对性改进**。两条路径都是文章路径。不追求"首个 xxx"：融得好 + 有创新点 + 性能好即可，新颖性是加分项不是红线。

## 口径锁定（本目录所有卡片共同前提）

- 网络输出**只有二值 {0,θ}**（AT-LIF 合同）；连续 I24/PSN/PED 只是**内部状态**，不作故事主框架，只能作为"内部消费者"出现。
- 质量门：优于同环境原 SDformerFlow NB0（valid825 AEE 1.445353；同环境复现 1.447937）。旧 +0.005 相对门已废弃。
- 准入门（项目自定，不变）：完整链同端口/同状态/同背压净服务 ≥15% + AEE 过门，才开隔离 RTL；RTL 证明净收益才开生产 EDA。
- 禁止把局部加法减少 / AEE 端点 / 仿真计数冒充周期或 PPA 结论。

## 已保留的强 A 底座（截至 2026-09-14，照抄+适配已转正的）

| 底座 | 来源 A | 实测净收益（边界） |
|---|---|---|
| 位平面 pipeline mode10 | BISMO AND/popcount 位平面 | 单ctx完整I24冷服务 −14.08%/−16.47%，背压下 −9.38%/−10.95%（组件Verilator，非整网） |
| K16条带+系数驻留 mode8 | 同上+驻留 | −8.16%/−10.04%，位图存储 4320B→80B |
| R8窄整数核+父和归并 | 自研分解（r0挂点） | vs expanded21 −51.37%；整层RTL −5.13~−5.45%；AEE 1.353326 |
| halo+直接消费+宽链组合 | 自研 | −5.058%（19200块全量） |
| count21 固定配对 | 计数/psum复用 | vs borrowRR −1.76%/−2.42%（条件可用） |
| Gustav 共享bitmap mode2 | GustavSNN NRV∩W | 训练稀疏W末门 −12.49%/−8.05%（2输出tile×4源ID切片） |
| 原生窗口/双P打包 | 自研 | −3.58%/−3.56%（R8完整消费者） |

## 候选卡（一页一张，B/A/X/杀门/两路径处置）

| 卡 | A（照抄对象） | X（差分方向） | 杀实验 | 状态 |
|---|---|---|---|---|
| [C1 消费者分级条件完成](C1_consumer_graded_completion.md) | BISMO位平面 + 已有K=4条件完成 | 按消费者精度分级提前接受（lane级证书） | [kill_b.md](kill_experiments/RESULT_B.md) | 卡+实验并行 |
| [C2 R-Sparse互斥两路](C2_rsparse_two_path.md) | R-Sparse (ICLR25) 幅值两路 | 内部连续消费者的互斥精细/近似 + 源字退役 | [kill_c.md](kill_experiments/RESULT_C.md) | 卡+实验并行 |
| [C3 舍入裕度分级证书](C3_rounding_margin_cert.md) | lifting R8 + da4ml常量编译 | 精确RNE不变式跳过证书 | [kill_a.md](kill_experiments/RESULT_A.md) | 卡+实验并行 |
| [C4 运动对齐delta](C4_motion_aligned_delta.md) | MotionDeltaCNN | 粗层流作参考的跨窗脉冲delta编码 | 待motion对齐统计 | 卡先行 |
| [C5 扫描窗口调度](C5_scannow_window.md) | ScanNow (ICCAD25) | 完成反馈感知的双context窗口调度 | 待RTL | 卡先行 |
| [C6 运行时激活分解](C6_dcom_runtime_decomp.md) | D-com 预印本 | r0贵算子的显式计费运行时分解 | 待全文精读 | 卡先行 |

## 杀实验

[kill_experiments/](kill_experiments/) 目录，全部用已有捕获数据（不新开训练），已完成（2026-09-14）：
- **A**（[RESULT_A.md](kill_experiments/RESULT_A.md)）：lifting 系数 |q|<Q/2 = 45.0%/42.5% → C3 过静态门（但 d≥2 档仅 22.5%）。
- **B**（[RESULT_B.md](kill_experiments/RESULT_B.md)）：逐通道接受 97.65%、块级门保持率 78.5%、L=8 lane 联合接受 85.64%、位深锁定 b=4 即 95.0% → C1 过静态门且空间大。
- **C**（[RESULT_C.md](kill_experiments/RESULT_C.md)）：块粒度门占用 81.2%（anchor=25% 并集节省仅 14.1% → 块粒度杀）；lane8 粒度占用 25.1%（节省 56.2% → lane 粒度有条件保留）。
- **综合判读**（[SYNTHESIS.md](kill_experiments/SYNTHESIS.md)）：粒度是决定性变量——384 通道聚合到块级后 78.5% 块必有失败通道；任何块级提前退休机制都被吃掉，lane 粒度是 C1/C2 唯一物理空间。C1 晋级 A' RTL，C2 保留但 X 必须 lane 粒度，C3 保留降优先级。

## 未覆盖文献检索（任务5）

[literature_search/](literature_search/) 目录：2025-2026 顶会顶刊未入 775 实体 catalog 的新工作检索记录。

## 边界声明

本目录全部为筛选与静态测量，不是 RTL 周期、整网 FPS 或 PPA。生产树 hw_autoresearch_nts07 与主稿对本目录只读。
