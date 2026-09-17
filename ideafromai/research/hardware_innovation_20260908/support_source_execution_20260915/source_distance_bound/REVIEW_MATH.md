# partial Hamming 距离淘汰：数学与先验独立审阅

本次实读 PLAN、`probe.py` 与 `probe.json`，只审数学、CPU 证明范围和新颖性；没有运行 probe 或 RTL。本文件不是 RTL/握手/周期通过证明。新增 probe 已报告四种标签定义各自对 6×65536 输入穷举全同；我核读了穷举与独立 argmin 的代码，没有将其源工作量模型当硬件周期。

## 删除、提前停与查询的正确性

令 K 是 known 位集合，`dk=pop((Dk xor word)&K)`。未知位中，Dk=Dj 的位对两距离差贡献为 0；Dk≠Dj 的每位贡献为 ±1。因此所有未知完成上的距离差最小值恰为
`Lkj = dk − dj − pop((Dk xor Dj)&~K)`。
若 Lkj>0，则 j 对每种完成严格优于 k；若 Lkj=0 且 j<k，j 对每种完成在“距离、原索引”字典序上优于 k。这两条可以删除 k，真 winner 不可能被删。不能把等号条件改成任意 j，或把 d 最小的当前候选当已确定 winner。

允许已删 j 作 witness：j 仍是原合法码字，证明 k 被它压过已足够，不要求 j 自己能最终胜出。查询后 Lkj 只增加 0 或 2（分歧位），或保持不变（相同位），故已证淘汰不会随更多真实位到达而失效。逐 j 或同时更新 survivor 都安全；前提是每次 dk、K、word 属于同一个 t 的一致快照，不能混入下一批或别的组。

真 winner 始终在非空 survivor 集内，故**所有** survivor 的完整 H384 response canonical 相同即可返回该标签，不必确定 winner 的原码号。`probe.py` 已逐组检查每个原 D 响应与 canonical 响应的 384 项整数值全等；因此只对当前 FC1 消费函数安全，不承诺恢复原 projected bits，也不适用于另外消费 raw gate/code 的接口。code 控制使用 identity canonical，不得偷用 class 提前停。

未知位允许全部 2^u 完成，是实际 PSN 可行集合的超集；即使 PSN 位相关，证明仍保守成立，不依赖独立性。known 只能在相应目标 t 的真实 10-term A0×X 完整结果已判 gate 后置位，不能以未完成 MAC、预测或默认零充数。十个 t 可有各自 known/survivor，算法不要求一次生产整 T10。

固定 D-only rank 中取第一个未知的 survivor 分歧位，不看隐藏答案、不重新拟合。若 survivors 对所有未知位一致，则它们的未来距离差不会改变，按当前距离和 index 已可留唯一 winner；所以 exact-code 未结束时应能找到分歧位。相同未知位对所有候选增加共同距离，可跳过。实现应保留“非空 survivor / 找不到分歧即已能退休”的断言，不能在失败时默选某位。固定 rank 是受限查询策略，不是最优特征采集证明。

数值上 dk/dj/u 均在 [0,16]：5bit 存距离可以，**5bit 无符号相减不可以**。可使用 signed6 差，或把 dk 与 6bit 的 dj+u 比较，再处理等号 index；~known 必须截为 16bit。Python 使用 int16 差和显式 65535 mask，未见此错误。

## 当前证据与硬件代价边界

`probe.py` 直接以原 D 对每个 16bit word 做独立全距离 argmin，再比较所构造树的 code/canonical 输出；没有 lo==hi 消查询或额外 ROBDD 合并，树仅作 CPU 验证。四种变体各报告 393216 个输入全同，累计 1,572,864 次标签核对。它覆盖固定 D/标签下所有输入，而非任意 D/W 或未来 RTL 状态。

| CPU replay，32 个训练帧固定 P32 | target (P,c,t) 数 | 标量 MAC 数 | 四槽模型 X words |
|---|---:|---:|---:|
| exact code | 435787 | 4357870 | 115878 |
| W′ class | 435091 | 4350910 | 115570 |
| W″ class | 432713 | 4327130 | 114102 |
| Wz class | 424986 | 4249860 | 112614 |

这些是冻结实际 integer gate 上的机会统计，未制造真实查询 MAC/背压。graph_words_required=0 只说明候选不读图；D、canonical、A/τ、rank 和源 X 仍须付费。PLAN 的 16 popcount16 复用方案，每个未完 t 每次重判为 DIST1+DOM16+REDUCE1=18 拍，再加真实源生产/调度，不能把 CPU 树一次跳转当一拍。四槽 X、候选/known 状态、比较及 REDUCE 组合逻辑都应独立列资源，不宣称与 ROBDD 等面积。

## 已有方法与候选增量

部分距离淘汰与精确 VQ 搜索是成熟 A。Katsavounidis 等的原论文 §I、§III-B 明写部分距离提前排除，并结合树与距离下界完成 exact nearest-neighbor 搜索；它不是只有近似 VQ。当前式子把 Hamming 两候选的共同未知项抵消，得到 pairwise 差的界，区别于仅把部分绝对距离同一个已完整算出的 incumbent 比较，但这是直接的精确 branch-and-bound 实例，不能据此宣称新的最近邻数学。[1996 原论文](https://mcl.usc.edu/wp-content/uploads/2014/01/1996-02-Fast-tree-structured-nearest-neighbor-encoding-for-vector-quantization.pdf)

昂贵特征按实例选择/停止也有成熟先验。Gao–Koller 的工作按既有观察和预期收益/成本选择下一项；本方案没有该概率价值模型，而是固定 D rank 与对所有未知完成成立的证书。因此可以声称不同的正确性/执行合同，不能把“按需制造特征”本身当首创。[NIPS 2011 原始摘要](https://papers.nips.cc/paper_files/paper/2011/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)

当前候选 X 是：用消费者精确 response 等价来定义退休目标，把传统距离界的下一位查询接到非因果 PSN 的实际目标 t MAC，并在有限四槽/有限 popcount 调度下与图供数竞争。它仍接近通用 feature acquisition、决策图 cofactor 与任务相关等价类的组合；无新 RTL 成本闭环时，独立新颖暂评 **3/10**，不据此停止迁移成熟 A。

强对照应保留同权利 static64+next-X PF、当前最强 retained/active-PF ROBDD exact-code/class，以及**同一个距离引擎的 exact-code**。每种 W 函数分别比较，最后一个差分才隔离 class 退休增量；不能以距离 class 对旧弱图/static 的全部差额算 X。最有价值的判别量就是“去掉的实际图请求/源 MAC，是否抵过 18 拍逐 t 判界与槽失配成本”。在这项实测前，保留候选，既不宣布加速，也不把整个家族判负。

