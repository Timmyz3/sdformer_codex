# 直接先验补读：动态神经元剪枝与 CFMP

当前最明确的结论是：**CFMP 已覆盖“训练出稀疏中间表示→在生产前译码掩码→生产与消费用同一掩码→恢复稠密输出”的完整行为。** 当前联合训练若只做到这一链，不能算剩余 X。尚可检验的差别是：已经算出的连续响应是否改变非因果 T10 的未决依赖，并在真实 P4/H8 消费者并集下取消尚未发出的卷积请求。该差别尚不能对 CICC 2025 动态神经元剪枝作排他性判断，因为本轮没有取得其正文。

## 1. CICC 2025：身份已核，方法缺口仍在

**A 40nm 0.05–1.4uJ/inference Sample-Wise-Adaptive Spiking Neural Network Processor with Dynamic Neuron-Pruning and Unstructured-Model-Aware Architecture**，Jinqiao Yang 等，CICC 2025，Session 11-5；DOI `10.1109/CICC63670.2025.10983259`。题名与作者见[官方最终节目](https://www.ieee-cicc.org/wp-content/uploads/2025/04/CICC-2025-Program-4-8-25.pdf)及[作者学校目录](https://www.it.fudan.edu.cn/En/Data/View/2025)。

本轮先查本地，再查题名、DOI、作者及代码；本地没有该文，出版社页面返回 JavaScript/robot 验证，作者页面未取得可读正文，定向检索只找到节目、作者目录及二手索引。没有绕过访问限制，也没有把“没有找到公开工件”写成“没有工件”。按任务要求停止重试，转精读已有 CFMP 作者全文。

| 能力 | 已知程度 | 当前判断/尚未迁入 |
|---|---|---|
| 按样本自适应、动态神经元剪枝 | 仅题名明确 | 可以列为直接强先验，不能把本地逐门预测叫第一种动态神经元剪枝 |
| pruning 输入、阈值、训练/校准 | 未知 | 不知道用源活动、膜、历史、分层置信还是其他统计；不能自行补一个算法再称原法 |
| 剪的是时间步、神经元状态、输入权重请求，还是后继突触 | 未知 | 这正影响它是否覆盖我们“取消尚未产生的 Conv 时间列”的 X |
| 有损/精确、恢复及最坏情况 | 未知 | 不能因为题名含 pruning 就把它作为精确证明，也不能预设它只有有损功能 |
| unstructured-model-aware 的索引、负载均衡、存储 | 仅题名明确方向 | 不能声称已承接完整稀疏底座；需要正文确认地址生产与反馈边界 |
| 神经元模型、T、θ 幅值/τ、BN | 未知 | 不允许从 SNN 一词推定只支持因果 LIF，或反过来推定支持满秩 T10 |
| 0.05–1.4 µJ/inference | 题名数值 | 不知道对应模型、样本和准确率；不用于本地优势比较 |

因此，当前缺的不是再给这篇打低分，而是它的方法正文；新颖性文字必须暂时保留这一不确定性。相同作者的 TBioCAS 2025 temporal-spatial post-neuron-processing 题名不同，本轮没有证据把它当作本篇完整扩展替代。

## 2. CFMP：已读完整三页作者稿与图 23.2.5

**A 28nm 0.22μJ/Token Memory-Compute-Intensity-Aware CNN-Transformer Accelerator with Hybrid-Attention-Based Layer-Fusion and Cascaded Pruning for Semantic-Segmentation**，Dong、Tan 等，ISSCC 2025 23.2，DOI `10.1109/ISSCC49661.2025.10904499`。[作者公开稿](https://arxiv.org/pdf/2512.17555)，[共同一作主页](https://yonghao-tan.github.io/)。本地稿为 `literature/ISSCC2025_23_2_ConvFormer_author.pdf`；全文及图 23.2.5 本轮实际读取。未定位公开训练代码/RTL；三页 digest 的完整阅读不等于训练细节已经可复现。

为避免误读，下面数学式是依图 23.2.5 写出的功能表达，不是原文给出的训练公式：

`Z = M ⊙ (X W0)`，`Y_hat = Z W1`。

X/Y 是稠密端点，W0/W1 是**单个卷积**的两个线性因子，中间维 D 扩大后通过 tiled mask 稀疏化。图中不同 X 行块可选不同中间列块；不是只能全空间使用一份通道裁剪。若 M 恰好全空间相同，必须再给静态低秩/因子压紧乃至折回有效 W 的普通控制。

| 原 A 的能力及定位 | 必须完整承接 | 当前尚未迁入/不能偷换 |
|---|---|---|
| 训练端分解、扩 Z、tile-wise sorter；Fig.23.2.5 左上 | 联合适配两个因子及掩码，保持原输入/输出任务监督 | 原文没给齐扩展比、损失、mask 排序准则、优化器/恢复日程；我们补出的方案应标“功能迁移的具体化” |
| 同一 mask 决定 W0 的 TC 与 W1 的 TR；Fig.23.2.5 上部 | 在第一级乘加之前选中间列，第二级只遍历相应行 | 后验把已算 Z 置零不是生产跳过；只做一侧 mask 不是完整 CFMP |
| FMS：16 KB mask buffer、split、one counter、valid-tile 计数、flatten/offset | 译码有用 TC，叠加源/权重/输出 base ID；给控制和端口收费 | early stop 指“当前 mask 的有效 tile 已输出完”，不是根据膜值提前终止神经元 |
| sparse Z 用紧凑的 dense-format tile 存储；FMS→DRU 流水 | 保留 Z 到匹配的第二因子消费，索引伴随数据 | 不能只报逻辑非零数，忽略实际 tile 容量和同时 live 中间量 |
| DRU TC→TR、跨 bank 切片、多 base ID、归约；Fig.23.2.5 下部 | 不同 Z tile 与 TR slice 完整遍历并累加到稠密 Y | 不允许免费 CPU 排序/全交叉栏；一个 TC 地址不等于一拍完整 TR |
| HAPU/LMB/RMB 共同执行，FMS/DRU 交错 | 当前边界相关的 PE、读写、输出恢复和阶段互斥都要给双方 | 原 SoC 的 2 MB LMB＋1 MB RMB 不直接充本地 2 KiB slice，原资源和我们的预算分别说明 |
| hybrid attention 与 LFS；Fig.23.2.3–4 | 若借 attention-conv 融合，承接 KV/权重换驻留与 overflow tiling | 单借 CFMP 不必复制无关 attention 但也不能拿其整芯片倍率；H67 K=V/Motion-XOR 不是原 QKV 路径 |
| non-overlapped LF 缺边界零填，随后 attention 补空间依赖；Fig.23.2.4 | 作为新网络与完整 halo 控制对照 | 原文明确存在准确率损失，不能拿作当前 residual 的无损 halo 消除 |

原文对 SegFormer-B0 的“15%→90% sparsity”脚注是**跳过操作百分比**；图中约 2.03× 能量差同时涉及 VA、fused conv、head conv 与 DDR3，不能贴成 CFMP 单个 patch 的同负载电路收益。论文已经做训练改变网络，不能把我们的原权重没有零块当作它不适用的结论。

还需避免一个数学误用：本网络已有 `Conv1→BN→PSN/θg→Conv2`，中间有阈值非线性，不能直接把这两 Conv 叫作 CFMP 的两因子。例如 x=(1,−1)、两个线性因子为 I 和 [1,1]^T，直接合并得到 0；在中间先做 τ=0.5、θ=1 的发放后却得到 1。保留连续 θ 不会消除这个差别。若迁 CFMP，应分解某**一个**卷积并保留其外侧原 BN/PSN/Conv2；若在两因子之间另加脉冲，那是新模型，须单独评价。

偏置也不能悄悄消失：`M⊙(XW0+b0)` 中 b0 同时受 mask，通常不能直接折成全局常量 b1。若采用无中间 bias 的分解，必须明确这一结构选择，并保持固定 BN、源 θ 与输出 τ 的准确身份。

## 3. 当前 X 的逐句覆盖和判别

| 拟议文字 | 覆盖判断 | 可以留下的实测问题 |
|---|---|---|
| “训练中间稀疏性来减少生产和消费” | CFMP 已覆盖；SpikeX/DynConv 也有费用训练近邻 | 不能作为独立贡献句 |
| “相同 mask 连接两层，并恢复 dense output” | CFMP 直接覆盖 | 完整搬入后看索引/状态费用，不是新意 |
| “稀疏中间值的列转行并跨 bank 消费” | CFMP DRU 直接覆盖 | 只能是完整底座，除非有同资源新行为和额外收益 |
| “根据部分实际连续响应，确认最终 θg 并取消剩余 Conv 列” | CFMP 公开稿没有这类运行时判决；CICC 11-5 未知 | 需与完整神经元 pruning、逐门/整词退出及 one-shot32 比较；目前不能宣称排他新颖 |
| “同时学习 E 的共享/私有时间依赖，让最后真实消费者消失” | 在 CFMP 图/文中未见该具体行为 | row34 与 common3 都给相同普通/物理并集目标，才可测结构与目标的交互收益 |
| “物理请求 OR 费用训练本身创新” | 不能成立：普通 mask 依赖回推、块活动费用已有近邻 | 要测 union 目标是否在相同门错误/AEE和资源下多减少真正请求，不能仅匹配单门率 |
| “保满秩 T10 或 θg 即不同于先验” | 不足以构成性能机制 | 满秩/连续幅值是身份和约束，新增行为必须另证 |

**B 仍成立的条件。** 多个 t/h/p 共享一份物理取权/中间值时，单门提前完成往往不释放资源。这里的 B 必须是实测剩余请求/状态开销，而不是总发放零率。根节点已有 common3 持续检查被普通 one-shot32 超过的负结果，已说明提高检查频率不是可用 X。

**当前最有判别力的对照维持原 2×2，而非新增路线。** row34/common3 × 普通终止目标/真实物理并集目标，双方保留同源目录、同 source×W-zero、同检查点和 one-shot 权限。原始 Y 未来值只给训练标签，推理只能用已算 prefix。结果分别报 Conv1/索引/PSN/Conv2/状态和净服务，不用门率代替；尤其显式计 prefix 与 tail 是否重复扫源、一次 W 是否能跨 T 共用。

当前不能因为 CFMP 有先验就停止训练，也不能因为 CFMP 没写 PSN 就自动把组合升格。若普通 row34＋物理目标吃掉全部收益，common3 不是新增架构；若 common3 只胜普通逐门损失，却不胜同权限 physical-union row34 和一次许可，仍属于弱分母。完整 CFMP 适配后的两因子学生另进 AEE–整链费用表，不能称与现学生同函数加速。

## 4. 建议与交付边界

本轮**不另提第三个结构 X**。更具体、正交的候选还没有比根节点已在做的联合训练更强的证据；新增过完备 latent、mask 或稠密恢复，基本都会直接落入 CFMP 能力范围。先把它的完整功能迁入作为强学生/底座，并补 CICC 正文，才知道需要新增什么。

本轮已完成：CFMP 三页方法和关键图的独立补读、功能/状态/索引对应，以及 CICC 公开可用性查证。未完成：CICC 正文、两篇官方实现工件、CFMP 训练 recipe 的作者级复现。没有训练、GPU、EDA 或生产代码改动。
