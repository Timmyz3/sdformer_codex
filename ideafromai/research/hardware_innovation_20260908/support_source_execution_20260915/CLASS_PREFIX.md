**完整 T10 判决已由 CPU 穷举核过；自然顺序下 response class 比普通 code 少算 0 次源任务。一次仅由 D 决定的排序改善了普通图，但 class 的独立机会仍只有 6/3359＝0.1786%。** 这是源请求数，尚不包含图查询、配置、反压和源 MAC 的实际服务周期。[生成器](build_prefix_tables.py)、[自然序统计](prefix_tables.json)、[唯一排序适配统计](prefix_tables_entropy.json)。没有训练、AEE 或 EDA；这里不把普通决策图当新算法。

**精确函数与最强普通权限。** 对每组 `D[16,16]`，原判决为 `k(x)=argmin_k popcount(x xor D[k])`，距离相同时必须取最小原码 index。用上轮已固定的 [W′](../support_lut_execution_20260915/response_class_W.npy) 计算完整 `H384` 响应，`c(k)` 取响应等价类内最小原码。目标是 `c(k(x))`，不是重造或近似原始 gate；W′ 本身是已改变的函数，不能继承原 W 的网络质量。码间 Hamming 公共项可静态删除：每组信息位为 **9/13/15/8/11/8，共 64**，其余 32 位对所有 D 恒零。普通 code/class 两臂均给这一权限；动态收益的分母为 64，不是 96。

已知位集合与其值为 `s` 时，只有 `S(s)={c(k(x)): x 为 s 的任意合法 16-bit 补全}` 是单元素，才允许 class 退休；普通 code 同样检查 `{k(x)}`。这里枚举每组全部 **65536** 输入，独立核 XOR-popcount 与 `|x|+|D|−2xDᵀ`、最小 index tie、公共项删除、全部前缀证书、ROBDD 全叶值和 64-bit 节点解码。源一次完成同一通道的 **T10 整字**；前缀成本取十个停止深度的最大值，ROBDD 则按固定变量顺序逐次取十条未完路径中最早的变量，只推进需要该变量的路径，全部十条到终点才结束。不能把逐 gate 的平均深度当源任务减少；也不能凭部分 H12 相等而忽略其它响应/消费者。

| 同一输入和函数权限 | 普通 code 前缀请求 | class 前缀请求 | 普通 code ROBDD 请求 | class ROBDD 请求 |
|---|---:|---:|---:|---:|
| 真实两帧×32位置，自然序；静态 4096 | 3995 | 3995 | **3786** | **3786** |
| 同一真实输入，固定 D 熵序；静态 4096 | 3728 | 3723 | **3359** | **3353** |
| 合成均匀 4096 包，自然序；静态 262144 | 261714 | 261714 | 261338 | 261338 |
| 同一合成输入，固定 D 熵序 | 259858 | 259390 | 258734 | 258310 |

真实输入来自根生成的 [source_cases.npz](source_cases.npz) 的 `raw_g_int[2,10,32,96]`，是真 `A16Q12 × X24Q16 / thresholdQ28` 源 PSN 的投影前整数门，已与 `code_index_int` 逐值相等；仅首两个**训练帧**，不代表留出分布。自然序比静态少 7.5684%，两臂完全相同；排序后普通 code 少 17.9932%，class 少 18.1396%。六次增量出现在 g0 四个包、g1 两个包，每包只省一个完整 T10 源通道。例：frame0 / position-slot30 / g0 为 code 9 次、class 8 次。自然前缀的全域早停增量为 0/393216 输入；熵序 g0/g1 分别有 3072/5296 个标量输入早锁 class，但打成真实 T10 后不能直接保留这些逐标量比例。

**唯一失败适配及容量。** 每组先删共同位，再按 D 列中 16 个码的二元熵降序、物理通道号破平局；等价整数排序键为 `(|one_count−8|, channel)`。这是均匀原型先验下的固定信息量代理，不是实际最近码后验信息增益，也不是最优顺序；完全不读取训练/验证活动来择序，没有扫其它排列。节点中仍存原物理通道，调度必须按 `variables_g*` 的序号挑最早 live 变量，不能对物理编号取 min。

| 表容量 | 自然序 code / class | 固定熵序 code / class |
|---|---:|---:|
| 节点数，含共用 ID0…15 终点 | 2179 / 2054 | 966 / 955 |
| 每种顺序两臂共同许可的 64-bit 表容量 | 2180×8＝**17440 B/臂** | 966×8＝**7728 B/臂** |
| 可选紧凑 32-bit 非终点载荷，未用于冒充 RTL 成本 | 8652 / 8152 B | 3800 / 3756 B |

共同自然序硬件预算也可保持 17440 B，让熵序只少占用而不宣称面积下降。若两套表同时驻留，须累加存储，不能把单臂容量当双臂总量。完整无压缩前缀证书各有 88058 节点，逻辑 5-bit 为 55037 B/臂；NPZ 的压缩文件大小不是硬件 SRAM 容量。固定 64-bit schema 为 `lo[15:0], hi[31:16], physical_var[35:32], terminal[36], label[40:37]`；根数组为 `roots[2,6]`，mode 顺序 code/class，各臂 12 B。所有 ID 均容纳 16 bit，T10 节点 ID 本身需要 20 B holding，另须子节点/变量/ready/cache 等状态。若用可配置的 physical→order-rank 表实现熵序，则 `96×4bit=48 B` 加配置服务也须支付。节点字每 128 bit 两个、真实 bank/cache 服务由根 RTL 计，CPU 不能将十路查询当零成本；本页也不把 10-MAC 源组件等同原 96-lane 完整网络链。

**最近邻的实际覆盖范围。** Bryant 的 *Graph-Based Algorithms for Boolean Function Manipulation*（IEEE TC 1986）§2 定义和证明固定变量序下的同子图共享、相同子节点消除与规范表示；本文作者网站版本是重排版及小修订本。当前多终点标签图就是该常规方法的直接扩展，固定序下的图规范性不证明最佳变量序、T10 调度或实际周期最优。[作者全文，§1–2](https://www.cs.cmu.edu/~bryant/pubdir/ieeetc86.pdf)

Lin/Jiang/Lee 的 *To SAT or Not to SAT: Ashenhurst Decomposition in a Large Scale*（ICCAD 2008）§2.1 用所有剩余输出的列模式计数给出可分解条件，§3.1 用 SAT/插值构造等价类编码。它已经覆盖“输出不再区分的源状态可以合并”这个一般原则；本次并未实现它的 SAT 求解，而是对 16 位函数直接枚举并做普通图归约。原始 gate 若还有其它观察者，必须把那些输出一并放入等价定义；仅 H384 一致不足以绕过另一消费者。[会议全文，§2–3](https://cecs.uci.edu/~papers/iccad08/PDFs/Papers/01B.2.pdf)

Xu/Kusner/Weinberger/Chen 的 *Cost-Sensitive Tree of Classifiers*（ICML 2013）§3 显式将弱学习器求值与按需特征提取分别计费，特征首次提取后可缓存；§4 联合学习路由和预测来权衡成本与误差。它已覆盖“只生产判决需要的昂贵特征”的动机。本次区别是保持原最近码/完整响应函数的全补全精确性，并按 T10 共享生产粒度与有限图 RAM 计费；这不是新的一般特征选择原则。[作者会议全文，§3–4](https://proceedings.mlr.press/v28/xu13.pdf)

McNames 的 *Rotated Partial Distance Search for Faster Vector Quantization Encoding*（IEEE SPL 2000）§I 在部分非负距离超过已知完整 incumbent 时排除候选，§II 由码本 PCA 将高方差方向前移，§III–IV 同时报告变换开销、低维退化和可能的双码本存储。它明确说明 PDS 可省距离累加，不能据此推定原输入特征没有生成；旋转后坐标也不能免费拿到。这里的 D 熵排序同属普通提前判决适配，不是 PCA 或欧氏旋转的 Hamming 等价实现。原 Bei/Gray 1985 仅核到作者书目及此文参考文献，未得原全文；不能声称精读它，且 McNames 将 Cheng 等 ICASSP 1984 列为更早提出者。[作者全文，§I–IV 与参考文献](https://web.cecs.pdx.edu/~mcnames/Publications/RPDS.pdf)

**B/A/X 收口。** B 是生成全部原 gate 后才近邻归类的源成本；强 A 是删 32 个公共位、同序/同容量 ROBDD、同 T10 请求与物理查询缓存。待评估 X 仅剩：把已经证明的完整消费者等价反馈给这个昂贵源，是否在图查询后还有净服务收益。当前自然序增量为零，固定排序增量只有六次；这支持继续核实际组件账，不支持新判决算法标题。主观新颖性 **2/10**，不是接收概率，也不据此否定其它可学习的函数/源生产粒度。最关键的后续接口是 `group/channel 整 T10 source request → paid node query → typed code/class`，而非 TB 提供类别或补全证明。

复现：在本目录运行 `/opt/anaconda3/bin/python3.12 build_prefix_tables.py --source-cases source_cases.npz`；唯一适配再加 `--order entropy`。自然序 [prefix_tables.npz](prefix_tables.npz) 与 [熵序 NPZ](prefix_tables_entropy.npz) 分开保留，含全部证书、叶表、变量序、roots 和 64-bit 节点；仅依赖 Python/NumPy 和既存 D/W′/真实源输入。
