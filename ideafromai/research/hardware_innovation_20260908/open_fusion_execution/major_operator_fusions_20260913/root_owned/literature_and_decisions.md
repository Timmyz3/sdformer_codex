# 本轮借入边界与未试接口

本轮不是全目录清空，也不是给新名字评分。先对完整 A，再试不同表示/执行接口；失败只限于测过的布局。独立代理负责三个代码目录，主代理负责当前前向与 AEE；彼此看到过项目先验，因此不是无锚定、完全盲法的独立发散。

| 完整工作 / 年份与出口 | 本轮实做 | 还缺，或为何没有作为新贡献 |
|---|---|---|
| [Prosperity，HPCA 2025](https://github.com/dubcyfor3/Prosperity) | 原样公开 FC；完整本网 M/N/K 外层、父林、真实 W 代数输出；共支持分组/时间编码/空间批 | 公开源码不含完整 ASIC RTL/DC；编码未胜 A。源码已有 m→n→k output stationary，g_psum 访问不能假装 DRAM；继续有限状态/端口试验。 |
| [LoAS，MICRO 2024](https://arxiv.org/abs/2407.14073) | 时间字/静默位图接到本次 Prosperity 源生产、真实编码/解码收费 | 不是完整 LoAS TPPE/校正/inner-join；ProSparsity→LoAS 已在官方例子，不能把组合名当 X。 |
| [TASD，MLSys 2025](https://arxiv.org/abs/2403.07953) | N:M 残差与完整输入/归约费用思想，真实稀疏包 | 未完整复制 TASDER/TTC；普通结构稀疏、分解成多项是 A。 |
| [CALDERA，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a20e8451ffb07ad25282c21945ad4f19-Abstract-Conference.html) | 激活二阶矩、低秩/残差交替投影同权限控制 | 未做完整 Hadamard、LDLQ、格点流程；本轮不能称完整 CALDERA 移植。 |
| [LQER，ICML 2024](https://arxiv.org/abs/2402.02446) | 所有分解同样有 W8 系数控制 | 低秩量化误差补偿已有，未把本轮 W8 当原论文全流程。 |
| [LegoNet，ICML 2019](https://proceedings.mlr.press/v97/yang19c.html) | 有限共享输入基＋输出选择＋直接例外 | 普通共享基、二值选择已有；没有把该文误记为 CVPR 或声称完整训练复现。 |
| [SumMerge，ICS 2021](https://cwfletcher.github.io/content/research/2021.ics.summerge.paper.pdf) | 重复权重合并/输入和是必须保留的强编译控制 | pair 加法、公共幅值本身不是 X，必须比较同样允许这些优化的控制。 |
| [Finch，OOPSLA 2025](https://doi.org/10.1145/3720473) | 实际试了非零 fill 的例外覆盖：c(pop−g_s)+vg_s，完整后段仅多省14槽 | 非零默认值、run/repeated value 不是新代数。候选 X 只能放在物理请求/消费者扰动共同决定的表示与执行上。 |
| [TT-SNN，arXiv:2401.08001](https://arxiv.org/abs/2401.08001) | 查证 tensor-train/训练加速方向 | 本轮没有执行 TT，不能写“已杀”；Kronecker、TT、Winograd 是未试分解接口。稠密连续变换的费用问题仍须测。 |

Gustav CPTB/NRV、先前 lifting/常量编译是现有供数与控制底座。本轮没有复现一套新完整 Gustav，不重复把已有空源跳过、取数换序写成新融合。

## 其他 AI 记录中需要纠正的淘汰理由

本轮重新读了 `grok_review_20260911/GROKBOT_NATIVE_VERDICT.md`。其中“θg不能吸入W”、据此停二值岛，以及旧 AEE≤1.259 都已经被用户后续指令覆盖。**不据该文件的错误身份/旧精度门杀家族。** 其“比较器定向 TB 不是完整因果机制”的实现审阅可以保留，但同一份文档不能因此被整体当成最新合同。

当前合同：AT-LIF `{0,θ}`、静态推理 θ 可折入 W；PSN/PED 连续消费者仍在。AEE 使用同协议 NB0 门。非零 fill 和有限整数分解是这份正确身份下的原型，不是恢复逐事件连续 θ 的双 MAC 故事。

方法说明：使用 scientific-brainstorming 的独立分工、反对意见和负结果范围记录；没有执行其整套十步 CLI 工作流，没有用评分自动选赢家。方法来源记为 Kassis, Agarwal, He, Patel & Brueckner (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。分数与试验结果不代表编辑录用判断。
