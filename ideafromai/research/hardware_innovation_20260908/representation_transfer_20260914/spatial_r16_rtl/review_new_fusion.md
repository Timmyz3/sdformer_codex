# 连续 Q2 Winograd 有界独立审阅

2026-09-14；本轮只核 `../spatial_winograd_inputs/{PLAN.md,export.py,stats.json}` 及 `../spatial_winograd/{PLAN_RESOURCE_CONTRACT.md,spatial_core.sv,implement.py,prepare.py,tb.cpp}`；另核 `../spatial_r16_direct/os_core.sv`。源码静态闭环见末尾；RTL 回归由负责人执行，本审阅没有把源码成立当成测试通过。未跑实验、未改实现。

结论：应完成这一次 Q11 同函数普通/Winograd RTL 对照。移到已经连续的 Q2 避开了“前级脉冲 AAC 被变换成多位数”的原失效原因，合法且有实际瓶颈依据；它目前仍是强普通组合/适配候选，暂评 **算法组合 2/10，具体接口实现 3/10**。该分数不意味着停止实验或否定空间快卷积家族。

公式逐项成立。对横向四个整数 Z，令

```
D = [d0−d2, d1+d2, d2−d1, d1−d3]
U = [2g0, g0+g1+g2, g0−g1+g2, 2g2]
M_i = sum_rank D_i U_i
2p0 = M0+M1+M2
2p1 = M1−M2−M3
```

展开第一式得到 `2Σr(g0 d0+g1 d1+g2 d2)`，第二式得到 `2Σr(g0 d1+g1 d2+g2 d3)`；所以每条带恢复和也必为偶数，算术右移 1 是精确除法，不是中间 RNE。跨条带先恢复再加或最终一起恢复在线性整数域等价。原 Q1 θ/尺度已经吸进固定系数，T10 每个时刻独立，未改变卷积 padding/stride 或输出顺序。

`export.py` 从固定空间因子的有效 Q2 重新做唯一一次 per-output Q11 量化，同时重算 a_q40、展开 W32、p/wide/I24 gold；原 q1/Z/identity/J 不变。Q11 是新的模型函数，不能继承 Q13 的 AEE，不能把 Q13 普通因子当同函数周期分母。135 真实 tile 的 518,400 个整数输出与 Winograd 重建零差、奇数重建为零是 exporter 已有结果；本审阅核公式与代码，没有重跑该实验。该证明不覆盖任意 INT8 q1 或任意新的缩放矩阵。

| 项目 | 本模型已给出的保守界 | 对接口的含义 |
|---|---:|---|
| 原 Z | [−8562,7427] | Q1 双 P15 安全 |
| 变换 D 绝对值 | 17124 | 需 signed16，再显式扩到 19；不能沿用 Z15 解包 |
| 变换 U | [−2981,2981] | 保持 signed13 乘法操作数；Q11 原 g 与 U 必须用同一冻结导出 |
| 每个 M 任意 rank 前缀 | 239174076 | signed32 归约安全 |
| 恢复加减的绝对值上界 | 507080380 | 除 2 前的每个部分加减也在 signed32 内 |

controller 合同的资源/生命周期安排合理：每个 (y,T) 先把两个原 Z 字都实际读完，再通过同 8ALU 得到四 D，两次整字写原地覆盖为 16×2；下一对使用不同的两个地址。保持 1280B Z，不额外造全变换数组。M0 用原 acc，M1–3 加 96B；恢复完成前不能把尚需的 M 覆盖。stripe1 的旧 p 读到此时闲置 z_hold，随后另付 8ALU 累加与 p 写，不能把 p 加回藏在恢复表达式里。实际 SV 需核这些控制成立后才可闭环。

Q2 一份 8×768×13 表支持两种模型布局，普通臂配置原 g 的 576 向量、Winograd 配置 U 的 768 向量；换模式必须付实际重载，不能只 cfg mode 一拍就免费改系数。总系数配置 1152/1344 拍、Q2 静态额外 2496B、cache 416B（+104B）、尾字 holding +32B、M holding +96B 已在 PLAN 计入。另已向负责人指出：若每向量 live 位沿用，q2_live 576→768 还要 **+24B**；block_live/remaining 24→32 各 +1B，控制/地址/计数寄存器另计。原 position_live 80×8 可存四 D 的完整支持，但必须在任何 Q2 使用前全部刷新。

最强反对并非数值无效，而是普通分解、连续级的 F(2,3)、较窄 Q2 去适配已有乘法器和寄存器复用都容易由已有知识组合得到；且多出的输入变换/四域保持/恢复/Q2 表可能吃掉乘积数优势。原空间 factor raw 中 Q2 BASE_MAC 占 56.5%/55.2% 是瓶颈证据，并非新颖性证明。另一直接展开 output-stationary/同预算 bitmap 强臂已完成同 Q13 函数测试；空间 Q13 factor 对它的冷增益为 10.94%/17.38%，旧 Gustav RMW 的 29.55%/38.44% 不能再当主增益。Q11 Winograd 的同函数主对照仍须使用 Q11 ordinary，不能挪用这个 Q13 OS 周期比。

有界最近邻核查：

- F(2,3) 的小 tile 最少乘法、输入/核变换与输出恢复来自经典快卷积；本轮两输出六乘积变四乘积属于借入 A。[Lavin/Gray, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)
- 低秩与 Winograd 已有直接组合先验，但该文面向 3D CNN、Winograd 域低秩训练及列稀疏；摘要不足以断言其实现了本轮二值前级/连续后级接口。[Qin et al., 2023，核到摘要](https://arxiv.org/abs/2301.11180)
- 分离 CNN、核分解、共享 1D/2D Winograd 引擎及交替装载复用已有硬件论文；其摘要中的 DSC/SKC 不等于这里跨通道空间 R16。它否定“分离+Winograd+共享引擎”作为宽泛首创，不能据摘要断言本轮全部控制已出现。[Li et al., TODAES 2025，核到作者机构摘要](https://research.tue.nl/en/publications/algorithm-hardware-co-design-for-accelerating-depthwise-separable/)
- 量化与 Winograd 的动态范围/数值适配已有明确先验。WACV 2024 的方法使用训练、各阶段 8-bit 量化、clipping 与复数变换，与本轮无中间舍入的 Q11→D16/U13 有差别；该差别并不能让“缩权重位宽以适配 Winograd”自动成为新方法。[Mori et al., WACV 2024，核到论文摘要/引言](https://openaccess.thecvf.com/content/WACV2024/papers/Mori_Wino_Vidi_Vici_Conquering_Numerical_Instability_of_8-Bit_Winograd_Convolution_WACV_2024_paper.pdf)

真正值得另立、尚未实试的 X 只能是明确的接口机制：利用横向 B^T 与竖向 Q1 的线性交换，把 `B^T(Q1*g)` 推到二值门字的空间 pair，直接生成可精确累计的差/和事件，再用同 ALU/同 Z 容量完成变换域 Z，消除“先建原 Z 再读/改写为 D”的实际税。该候选必须收费 pair 元数据、正负/两倍事件、支持变化及 signed16，并与当前保留原 Z 的普通 Winograd 同函数对照；交换恒等式本身也不是首创。本轮不实施它，不以候选替代当前 RTL 的完成义务。

后续只需补静态 SV 闭环：唯一 ALU/mult、D 的16位解包与符号、两原字读后才覆盖、四 M 的恢复生命周期、stripe1 p 读加写、实际换布局配置、ordinary 与 Winograd 的同 Q11 raw/I24；不再广扫历史或扩参数。

追加：同函数直接展开 OS 强臂已核 `../spatial_r16_direct/os_core.sv` 当前实际源码，未重跑。其 1080B 逻辑位图确实落在同 **8bank×40×32=1280B** 物理存储：135 个 64bit 逻辑行通过 `(a%4)*2 + half` 选择相邻 bank，`a/4` 为公共地址；BWRITE 仅向同 row 两 bank 写 64bit，WORD_READ 只读一 bank 的 32bit。两种状态互斥，未偷偷增加第二个 bitmap 地址口。

OS 每次仅构造一个 P 的 864×T10 位图，付四次 source 窗口遍历与全部 BITS/BWRITE；10×27bit word_live 在每个 P 全覆盖。`phase_hold[10]` 的 40B 在构图与八 lane 输出累加间复用，第一次位图提交前每个 bit 均实际写完，后续 K32 也全覆盖，不能称免费预制元数据，但不必额外清零。每个 (og,P,T) 清八个累加寄存器，然后依据真实位图取 K，单个 256bit W32 向量读与八个 ALU 加法同拍；完整归约后只写一次 p_mem，全部 480 行写好才 DRAIN。这是实质更强原生臂，消除了原 Gustav 每事件 psum 读/写税，未增加 40×8 个输出累加器。metadata 270bit、remaining27/pending32 及地址/控制仍应计入。

OS 静态未发现隐藏端口或生命周期错误；其 W 读+ALU 同拍、factor 的 Z/cache 读+乘加同拍都只验证周期模型，没有等频/等面积证据。OS 的同函数最终结果由负责人独立实测确认后，应优先作为分母；若它吃掉空间 factor/Winograd 收益，本报告不会继续以旧 RMW 降幅宣传最强基线加速。

实际 SV 闭环：首次读取发现 q2_mem 已扩 768 而 q2_live 仍只有 576，mode1 尾部 192 向量支持会越界；已立即通知根和 RR。负责人现已同步 `implement.py` 与生成的 `spatial_core.sv` 为 q2_live[0:767]，本审阅只读确认该修复，+24B 支持成本也已纳入 PLAN。没有改负责人的代码。

其余静态核对未发现第二处明确功能/端口错误：mode_q 只在 start 锁存；ordinary 地址保留 `stripe*288+og*24+term`，Winograd 为 `stripe*384+og*32+term`；两臂同一 Q11 factors/gold，TB 实际各配置 1152/1344 拍。四个 XD 状态用同八条32bit carry链，减法由 rhs 反相/最低 cin=1，Q1 cut15 仅 ZADD 开启。XDREAD0/1 把 tx 对应的原字完整读取后，XD1/3 两次提交；tx 的 y/T 及 component 对应 0/10、20/30 四段地址，位置支持全 80 项刷新完才 VLOAD。新 D 的高低各 16bit 解包并显式 signextend19，未沿用15bit符号。

BASE_MAC 每拍唯一八个 19×13 乘法，按 component 路由 M0=acc/M1–3=m_aux；Q2 cache 单 selected_term 的8×13读，Z只 selected_rank bank32读。恢复 p0 后 M0 可以覆盖，p1只再用尚保持的M1–3；W_INV1A 显式由 M1−M2重启 acc，不继承已写 p0。W_PREAD/W_PADD/W_STORE 分拍收费，旧 p 暂存在不再用于本 pair 变换的 z_hold；两个输出地址相差10，最后仍 DRAIN 原480行。ordinary路径从同source到同q11 raw成立，联合额外寄存器不被当作免费等面积。

验证范围需保持准确：现 TB 逐值检查原 Z、变换 D 和最终 raw，M 更新依赖静态路由、前缀/偶数断言与最终 raw oracle，**没有 M 逐值 monitor**。每个运行进程固定一种布局，连续换源/重复 go 可测；“不 reset 跨mode重载”尚未由当前 harness 实测。负责人仍在完成回归时，本审阅不签发任何 Winograd 通过或加速数。后续读取其最终报告更新测试状态即可，不需要为这些范围措辞中断已授权试验。

OS 最终独立结果已到 `../spatial_r16_direct/OS_SUMMARY.json`：572 命令/2,196,480 raw与独立逐状态全部通过；ready held/disjoint cold64=2,446,720/2,850,988。对同Q13空间factor的主冷增益现为10.9380%/17.3764%，本目录 README/SUMMARY 已转用该强分母。该结果不会被当成尚未运行的 Q11 OS 或 OS 完整I24端点。
