# TCAS-II 加速器文章：怎么讲得亮眼，以及我们下一步做什么

2026-09-05。结论：保留 C1 + C2/TSBG 两个贡献，把主题从“整网稀疏加速器”进一步聚焦到“零跳过之后仍然存在的重复数据搬运”。优先完成能量与物理实现，再把消费者生命周期作为 C2 内部改进；不新增第四个算子故事。

本次阅读区分了全文与摘要。Juracy 和 Cheshire 已检查原文评价部分；DWS 和点云 FPS 采用作者/机构发布的摘要，只分析其问题、机制和指标组织，不假称读过其所有表格。最新 PredLM、DeltaTrack 的作者页面确认了题目与发表信息，但未取得可读全文，本轮不据此背书倍率或具体电路。

## 1. 最值得照着学的四篇 TCAS-II

### A. Juracy 等，A Comprehensive Evaluation of Convolutional Hardware Accelerators，2023

[作者机构全文，§III–V、Fig. 3、Table II](https://repositorio.pucrs.br/dspace/bitstream/10923/24902/2/A_Comprehensive_Evaluation_of_Convolutional_Hardware_Accelerators.pdf)。DOI：10.1109/TCSII.2022.3223925。

它将问题限定为：在不同存储延迟和设计约束下，选哪种卷积数据流。五种核使用相同外部接口，28 nm 物理综合，真实 CIFAR-10 首层输入的 post-P&R VCD 驱动功耗；存储能量另用 CACTI-IO 建模。曲线给核心/含存储能量、面积、访问和周期，Table II 总结取舍。它不是靠完整分类网络 FPS 才成立。

我们直接借鉴：ordinary / post-read / pre-read 三轴共用接口；横轴放 SRAM 延迟或复用密度；纵轴放“完成相同 Acc24 输出的能量”。核心、缓存、SRAM 分开，再给合计。尤其不要把内部缓冲副本免费：这篇的缓冲取舍正好对应 M2241。

### B. An 等，8.81 TFLOPS/W DRL Accelerator，2024

[KAIST 原始摘要](https://pure.kaist.ac.kr/en/publications/a-881-tflopsw-deep-reinforcement-learning-accelerator-with-delta-/)。DOI：10.1109/TCSII.2024.3374725。

摘要用三个机制分别回答三个成本：delta weight sharing 减少外存访问；block-mantissa PE 支持精度/吞吐配置；位宽适配取数器提高有效带宽。最高 64.3% 访问削减、最高 4× PE 吞吐与 8.81 TFLOPS/W 是不同指标，不是一个统一的全网倍率。

可学的组织方法是“一项结构对应一种昂贵事件”。我们的 C1 对应残差工作/parent 访问，TSBG 对应 SRAM 读使能，K8 对应重复 Acc24/控制逻辑。别把这些结构拆成三个相互相乘的速度故事。

### C. Zhou 等，Adjustable Multi-Stream Block-Wise Farthest Point Sampling，2024

[作者页面的 TCAS-II 2024 条目](https://changchun-zhou.github.io/)，[出版社条目](https://ieeexplore.ieee.org/document/10430381)。

它把范围收在 FPS 算子：四个可配置参数、统一硬件和配置搜索。摘要报告 1k 输入点、200 MHz、0.9 V 下的 0.005 ms 与 0.09 µJ/frame，并分别给最高延迟/能量改善和网络精度。这里的 frame 有明确点数和算子范围，不应理解为整个点云网络能量。

对我们最有价值的是“范围主动收窄”：完成一个 B4 FC 区域或一组 parent-product row，完全可以是有意义的电路评价单位。必须同时写输出形状、有效工作量和从装载到最终提交的时间边界；若只测 post-load，标题/表注就写 post-load。

### D. Ottaviano 等，Cheshire，2023

[作者全文](https://arxiv.org/pdf/2305.04760)，DOI：10.1109/TCSII.2023.3289186。重点看 §III、Fig. 8–11。

这是一篇流片的加速器主机/接口文章，不是我们应复制的验证规模。值得学的是表达方式：先用 burst-size→bus-utilization 说明协议效率，再拆接口面积，最后拆 CORE/IO/RAM 功耗。文中同时暴露宽缓冲占据接口面积的大部分；周期级 RTL 功能评价与硅测结果分别陈述。

迁移到 C2：先解释读请求为什么消失，再解释宽响应缓冲为什么仍花钱。新机制应该去掉一份可证明无须保留的 payload，而不是再给普通多播改名。

## 2. 可以直接采用的呈现技巧

1. **选择贡献真正改善的指标，而不是强求全部变成倍速。** K8 的主指标是等服务面积效率；TSBG 是相同端口下的读取削减及完成时间；C1 是有限端口下捕获的 product 复用。功耗出来后，TSBG 用 energy/workunit 把请求与电路开销统一起来。
2. **时间百分比可以与倍速并列。** Juracy 原文将 225,078→135,450 cycle 描述为约 66.2% faster；换成时间削减则约 39.8%。已有 TSBG 1.8345× 等价于 post-load 时间减少 45.49%；C1 1.6945× 等价于模型时间减少 40.99%。这是同一数值的直观表达，不是额外收益，也不是整个光流推理时间。
3. **用一张因果图替代很多防御文字。** 相同产品与提交数下，ordinary→post-read→pre-read 分别画 SRAM read、logic energy、SRAM energy。已有定向请求数为 2304/2304/576；尚无对应三轴能量，不能用当前两轴功耗任务冒充。
4. **画有解释力的曲线，而非挑最好窗口。** 主栏放冻结样本聚合；旁栏放预先按复用密度选的 low/median/high；再扫描存储延迟。最优点可标注，但不得替代总体值。
5. **把物理代价放在同一张图上。** 现有 K8 是 4.541× throughput/logic-area；加相同 SRAM 容量模型后为 1.687×。两者并列比只突出 4.541× 更能解释集成价值。
6. **把脚手架移出正文。** 文中保留模型/VCS/prelayout 一套标签及必要边界即可；编号、失败史、审批链和反复声明放仓库。引用 prior 后直接讲对象差和实测差，不需要逐段自我辩护。

不能照搬的做法是：把 peak 配置当平均、把局部时间当整网、用逻辑面积分母包装成含存储面积、或者把软件周期模型标成 RTL。公开论文里的归一化数值同样要核对工艺、位宽、时间步和基线；不能凭大数推断别人有不当行为。

## 3. 新近工作怎样借到现有电路里

- [ELSA，ISCA 2026 原文](https://arxiv.org/html/2605.20802v1)及[官方实现](https://github.com/Intelligent-Computing-Research-Group/ELSA)：mini-batch Gustavson 与 bundled AER 是 TSBG 的直接邻居。借鉴“先组织消费者、再发送共享数据”；本轮不搬它的 NoC 和整网 elastic inference。
- [APEX，2026-08 arXiv 预印本](https://arxiv.org/html/2608.19046v1)：在既有 LoAS 数据流内改一个神经元电路，是“已有架构 + 一个局部改动”的组织样板。其神经元模型不能无条件替换 H67 ATLIF；只借计算/修正延迟重叠的思路，不把预印本当已录用顶会。
- [MVQ，ASPLOS 2025 原文 §5.1](https://arxiv.org/html/2412.10261v2)：作者明确承接自己的 TCAS-II EWS 数据流，继续在取数器和 PE 上改进，体现合法的机制继承。其剪枝/码本需要模型改变，不进入本轮；借鉴的是对基线结构逐项改动和面积/能量拆解。

## 4. M2241 已跑完：新候选应卖什么

源码：`system_simulator/scripts/m2241_c2_weight_lifetime_dse.py`；结果：`results/m2241_c2_weight_lifetime_dse/result.json`。只新增一个小周期/端口模型，没有新合同或 SHA 链。

2880 个固定区域 workload 拆成 4320 个 cold G48 chunk，各计一次，不乘 output-tile 或 wrapper 因子；其中 3840 个旧 FSM 基线轴与已有 VCS 周期/读数完全相符。所有轴保留 1,662,312 个 update；数值验证仍由既有结构检查支撑，候选 RTL 尚未验证。

| TSBG 实现 | 模型周期 | SRAM bank reads | 额外 row-cache payload 写入 |
|---|---:|---:|---:|
| 原整行填充 FSM，LRU1/4 | 11,928,718 | 7,519,968 | 此轴未另计 |
| 同两槽 streaming + LRU1/4 | 10,104,872 | 7,519,968 | 120,319,488 B |
| 两槽 streaming，借 response slot、无 row-cache 副本 | 10,104,872 | 7,519,968 | 0 |

独立技术评阅提出的公平对照已经补上。结果说明：原来的 1.1805× 是逐 beat 重叠和 group 选择合并带来的通用改进；**零拷贝相对同样 streaming 的缓存版本，周期收益为 1.000×**。仍值得研究的是取消 1536 B 单行（或 6144 B 四行）的额外 payload 保留及复制，而非新的稀疏率。现有 M803 的 1024 B response slots、Acc24 的 1152 B 不能一并消掉。

120.32 MB 是这批 chunk 的逻辑内部写入累计量，不是外存流量、物理总线切换量或能量。新增一个响应寄存级的候选为 10,184,330 cycle；这仍是模型，不是时序结果。

热缓存反例保留：两个重复四-group bundle 上，stream-LRU4 为 841 cycle/384 reads，无 row-cache 为 1094 cycle/768 reads。冷块上的无损算术不等于跨窗口没有性能损失；不能默认部署总是 cold。

### 当前推荐顺序

**第一：现有 TSBG matched energy 与 hold。** 六份 measurement SAIF 已有。M2235 后续 DC 因 repo-relative filelist 被错误拼到 HW root 而失败，没有可引用新功耗；该失败网表不复用。M2242 已启动：复用六份正常活动，修正路径，在独立 WORK 中重新做 DC/PTPX；新 DC 生成的 SDC 另派生 TT 功耗版本，仅调整 operating-condition/library 名，不改时间/I/O约束。不重跑 VCS，也不新增合同或 seal 链。当前在 ordinary DC，还没有新功耗。真实 checkpoint 权重的 switching 敏感性是下一项，不需要训练。

**第二：消费者结束即释放 response slot。** 先做小型 RTL，保留已有 M803 所有权与 C2 Acc24 运算。最后一个有效消费者真正接受后再释放；与同 streaming 的 ordinary/cache 实现比较。周期相同但总面积/能量降低，仍是合适的 TCAS-II 结果，无须硬凑 1.15×。

**第三：若第二项物理上有效，再试“有界消费者集合的一次性 selective fill”。** B4 已知每个 beat 的全部消费者，可在发请求前 OR 出所需 bank mask；beat 生命周期结束即丢弃，不让 partial valid 延伸到后续 row-cache hit。这可能比在持久缓存里加 per-bank valid/partial refill 更简单。ordinary 同样支持 bank mask；epoch、慢消费者与乱序响应仍需验证。当前仅是结构假设，没有新增节能数字，不替换已通过的 TSBG。

不推荐此时加入近似 attention、重训 N:M、另一个 Conv matcher 或 FPGA 新平台。它们无法直接补目前的电路能量和物理时序缺口。

## 5. 五页写法

建议约：问题与图1 0.6页；工作负载/算术0.4页；C1电路1页；C2+TSBG电路1页；评价1.2页；讨论/结论0.3页；参考文献0.5页。不是新的内容填充硬门。

正文一句话主线：**在零源跳过已经完成的事件流中，分别消除重复 product 的计算/状态访问，以及跨私有上下文的重复权重交付。**

图1画昂贵访问与决策位置；图2画1RW parent冲突/转发；图3画pre/post-read因果；表1统一组件资源与结果；图4画复用密度/存储延迟下的能量。尚未得到的功耗先留待实验完成，不用公式填成实测。

已据此改写主稿的贡献段、工作负载小节标题和结论，减少合同式表达；没有加入 M2241 候选倍率，也没有修改已有摘要指标。具体电路收益完成后再决定是否把生命周期扩展写入 C2。
