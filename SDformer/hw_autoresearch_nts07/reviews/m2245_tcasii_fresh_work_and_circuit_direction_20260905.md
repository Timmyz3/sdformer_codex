# TCAS-II：本轮文献、实际电路进展与取舍

2026-09-05。推荐继续 C1 + C2/TSBG 两条贡献；新扩展只嵌入 C2，不替换 checkpoint、算术或 C1。最有希望的加强是：**既然一拍权重的消费者已经全部可知，就只取它们需要的 bank，并在最后一个消费者接受后释放原响应槽，避免再复制到持久行缓存。** 两步是一条“有界消费者集合”的电路故事，但各自收益必须分开测。

这不是宣称发明广播、bank gating 或 eager fork。可争取的贡献是这些机制在既有 typed-signed K8、私有 Acc24 context 和实际 M803 接口下的低成本组合与验证。当前没有足够新物理结果把稿件称为 Strong Accept。

## 1. 新鲜论文和 GitHub：读到了什么，拿什么，不拿什么

### ELSA，ISCA 2026：直接前辈，也是评价组织参考

[官方代码与论文入口](https://github.com/Intelligent-Computing-Research-Group/ELSA)明确使用 mini-batch spiking Gustavson-product、bundled AER 和细粒度 token 流水。因此“B4 广播复用权重”不是我们独占的新颖点。

本轮继续检查了其实际绘图/汇总代码，而不只读摘要：

- [run_figure16.py](https://raw.githubusercontent.com/Intelligent-Computing-Research-Group/ELSA/main/ELSA_Simluator/run_figure16.py) 中，其他工作的能量、延迟和部分 AEDP 是已有表格常量；ELSA 列来自自己的模拟结果。该图本身不等于把所有 RTL 放在同资源平台上重新运行。
- [run_figure17.py](https://raw.githubusercontent.com/Intelligent-Computing-Research-Group/ELSA/main/ELSA_Simluator/run_figure17.py) 明确列出六个 workload，单独计算 geomean；没有对应公开表行的 ViT 不混进该 geomean。
- [resnet50_metrics.py](https://raw.githubusercontent.com/Intelligent-Computing-Research-Group/ELSA/main/ELSA_Simluator/convolution/elsa_support/resnet50_metrics.py) 分开累计 tile/NoC 能量，并从最后层完成周期计算流水延迟。不能把“最后层时间”规则搬到未证明流水的串行核上。

可借鉴的是：公开比较和本设计同资源消融分两张表；明确聚合集合；模型的时钟、能量拆分、流水时间定义可追溯。不能直接拿图表常量或它的模拟器，给不匹配数据流的本 RTL 冠名性能。

### Procyon，DAC 2026：局部迁移可以成立，但别偷换吞吐对象

[作者论文全文](https://ubaidhunts.github.io/ubaidb/assets/projects/Procyon_DAC.pdf)，[作者组发表记录](https://casl.cs.umd.edu/publications/index.html)。其主张是在已有 Serpens/Chasoň 稀疏流引擎的空拍中安排其他 workload，而非另造完整执行器。§6 用公开 HLS 基线、实现频率和校准周期模拟器；报告模拟延迟与 FPGA 执行时间差在 1% 内。§7 的比较对象是分组的多 workload，不应读成单网络一次推理提速。

可借鉴：以已有电路为底座，解释一处可测的浪费，并证明很小的控制改动能消除它。暂不移植其全局多任务配对：我们的独立 context 已有，当前瓶颈首先是取数与宽缓冲，而不是缺另一个全局调度器。

### APEX，2026-08 arXiv 预印本：局部电路改变，不等于免费改变神经元

[预印本](https://arxiv.org/html/2608.19046v1)在已有 LoAS 执行框架中改神经元相关计算。它适合用来学习“已有架构 + 局部关键电路 + 精度/能量对照”的表达；本轮未核实正式会议录用，不能写成已发表顶会。

不移植其神经元到冻结 H67：神经元动力学和精度配置不同，会产生新的 checkpoint/精度验证工作。我们采取无需训练的存储生命周期方向。

### Sparse Stream Semantic Registers，TPDS 2023：小控制扩展的明确 prior

[作者机构版本](https://www.research-collection.ethz.ch/handle/20.500.11850/636014)，[公开 RTL 仓库](https://github.com/pulp-platform/snitch)。该工作通过流式间接寻址、交并集等操作扩展已有寄存器流机制，而非为每个稀疏算子重建完整核。它支持“复用现有执行单元、把地址/流控制做轻”的组织方式；其 RISC-V ISA 和通用 SpMV 不进入当前设计。

## 2. TCAS-II 的写法，具体学到哪里

详细原文笔记仍见 `tcasii_accelerator_story_and_next_ideas_20260905.md`。其中 Juracy 和 Cheshire 已检查全文；DRL 和 FPS 的结论限于作者/机构摘要，不能假称已经审核它们全部表格。

| 样板 | 应借鉴的组织 | 本文怎么落地 |
|---|---|---|
| [Juracy，TCAS-II 2023 卷积硬件评价](https://repositorio.pucrs.br/dspace/bitstream/10923/24902/2/A_Comprehensive_Evaluation_of_Convolutional_Hardware_Accelerators.pdf) | 相同接口下比较数据流，扫存储延迟，展示缓冲代价；不靠全网 FPS 才成立 | ordinary/post-read/pre-read，另加同 streaming/cache 公平轴；面积和能量含缓冲 |
| [Cheshire，TCAS-II 2023](https://arxiv.org/pdf/2305.04760) | 从协议利用率讲到缓冲面积，再拆核心与存储功耗 | 先证明请求何时被抑制，再看读减少是否被控制/复制能耗抵消 |
| [An，TCAS-II 2024 DRL](https://pure.kaist.ac.kr/en/publications/a-881-tflopsw-deep-reinforcement-learning-accelerator-with-delta-/) | 不同局部机制分别对应访存、有效带宽、精度配置 | C1、K8、TSBG 不乘成一条虚构的系统倍率 |
| [Zhou，TCAS-II 2024 FPS](https://changchun-zhou.github.io/) | 把算子范围、输入规模和能量单位讲清 | 用一个定义完整的 B4 工作区域作评价单位，不把它写成整帧光流 |

最值得采用的“亮眼呈现”不是再选一个弱分母，而是三个动作：

1. **标题和摘要直指被消掉的资源。** 例如“Pre-Read and Consumer-Lifetime-Aware Weight Delivery”，但后半句须等物理证据。已有稿继续用 C1 + C2/TSBG，不把新候选提前写进摘要。
2. **主数用读者直觉能理解的单位。** 已有 TSBG 的 1.8345× 可以并列写成固定区域 post-load 时间减少 45.49%；C1 的 1.6945× 是模型时间减少约 40.99%。每个百分比必须带相同的时间边界，不代表光流推理时间。
3. **一张图讲因果，而不是堆四条 novelty。** ordinary → post-read → pre-read，产品/提交相同，观察 bank reads、cycles、logic energy 和 SRAM energy。读数、面积效率和时间各回答一类问题，最终再用同工作量能量合计。

建议五页结构仍为：问题/图1 0.6页；工作负载0.4页；C1 1页；C2/TSBG 1页；评价1.2页；讨论结论0.3页；参考文献0.5页。新增消费者生命周期若有效，替换 C2 段内较弱描述，而不新增第三个并列主贡献。

## 3. 本轮实际完成：M2243 RTL 与 M2244 traffic 实验

### M2243：新的消费者生命周期电路

新增 `rtl_m2243/m2243_c2_borrowed_weight_consumers.sv`，直接消费现有 M803 保持的权重响应；只保存 metadata/四个 active-sign mask/pending bitmap，不保存第二份 128 B payload。每个 context 单独执行符号修正，INT8 先扩成 9 bit 再取负，最后接受的消费者才释放响应槽。

测试台接入真实 M803，准备 full-bank/union-bank 两次测试，覆盖两个逻辑请求槽、bank skew、消费停顿、空响应释放、INT8 -128、每 ID 恰好一次退休及 SVA stall 稳定性。独立技术检查指出的 posedge slot-busy race 已改为 NBA，并补了遗漏的退休/总更新检查。

当前只是小模块，不是已经替换整个 M2018 的 production scheduler，也不是跨 G48 的完整 Acc24 重放。VCS 已由普通后台进程排在现有 M2242 功耗任务之后；尚未得到 PASS 前不报 RTL 验证完成。没有新 hash/合同链。

### M2244：比“全 bank 都读”更强的公平对照

脚本 `system_simulator/scripts/m2244_consumer_union_bank_reads.py`，结果 `results/m2244_consumer_union_bank_reads/result.json`。全部 4320 个 cold G48 chunk 各计一次，不乘 wrapper/output-tile 系数；4 个单元测试通过，独立重算一致。

| 同一批 chunk 的取数方式 | SRAM bank reads |
|---|---:|
| ordinary，按需 bank，LRU4，支持 partial refill | 2,623,644 |
| ordinary，按需 bank，无 row cache | 2,629,596 |
| 旧 TSBG，全 bank 填充 | 7,519,968 |
| TSBG，消费者 bank union，LRU4 | 1,604,430 |
| TSBG，消费者 bank union，无 row-cache 副本 | 1,604,430 |

**推荐继续测的读数差是 38.85%（union 对 mask-aware ordinary LRU4）。** 相对旧全-bank TSBG 的 78.66% 主要含过量取数修正，不是独立 novelty，也不是周期/能量结果。普通基线拿到 bank mask 后，确实显著强于旧全-bank取数器，这是必须正面比较的 baseline。

同调度 union-cache 与 union-no-cache 完全同读数，故去副本本身没有额外读取收益；它是否值得加入，仍看映射面积和能量。热窗口反例也保留：重复四-group bundle 中 masked-LRU4 仅 96 reads，无缓存 192 reads；不能默认所有真实运行都是 cold。

把两项合起来的可检验命题是：在已知 B4 消费者和有限片上容量下，consumer-scoped fill 是否能以更少驻留状态兑现相同的 bank 选择效果？这比只说“我们也做零跳过”更具体，也比在持久缓存中堆 per-bank valid/refill 状态更值得测。

## 4. 现有收口优先级

1. M2242 完成两轴三窗口 matched DC/PTPX。当前 ordinary 轴仍在综合优化；没有新的可报告功耗。它复用正常 SAIF，不复用之前失败 DC 网表；真实 checkpoint FC 权重的 switching 敏感性仍需后续补，现活动使用确定性验证 INT8 权重。
2. 根据这次真实 hold path 做小范围修复；不能只改约束消掉已有 -16.4 ps 诊断。C2 现有等带宽面积结果保留，别与新不同身份网表混合计算。
3. M2243 两模式 VCS，然后决定是否接进完整 M2018。只有实际映射表明行缓存/复制代价值得省，才做 matched area/power；旧实现保持可选，不全局替换暖缓存路径。
4. 三轴因果能量、复用密度曲线、真实权重活动是比再开有损/新训练更适合当前 TCAS-II 的补强。暂无理由复活 attention、S2 或另一套卷积 matcher。

本轮改变的是新扩展的可行性证据与源代码，不是论文已测性能的等级。Strong Accept 仍取决于能量/时序、直接 prior 对照和清楚的电路因果，而不是增加编号或预写评分。
