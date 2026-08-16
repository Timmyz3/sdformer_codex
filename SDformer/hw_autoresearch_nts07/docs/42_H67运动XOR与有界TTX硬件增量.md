# H67 运动 XOR 与有界 TTX 硬件增量

## 不变的数据流

H67 保留当前 all12 H60 结构：`Q/K binary event -> per-token TX -> Shiftmax -> gate*K`。
不增加 native `K*sn2(sumQ)` carrier，不增加 SC，不增加 token-token score SRAM，也不改变
105 个 one-sided binary ATLIF wrapper。

## Motion-XOR TTX

对时间窗 `T=2` 的同一空间位置缓存另一时间片 K：

```text
motion = popcount(K_t XOR K_pair)
raw_score = tx_raw_score + motion/4
score_q7  = tx_score_q7 与 motion 合并后做整体 round-to-nearest-even
```

冻结部署参数为`head_dim=32`、score step=`1/128`，因此 Motion-XOR 项在 Q7 域的缩放结果恰好
是`motion`本身，不是`motion >> 2`。由于软件使用`torch.round`，tie-to-even 必须在 TX 整数项、
motion 和 silence 分数相加后统一执行，不能先单独舍入 silence。完整修正和 RTL 见
`docs/44_H67主线切换与增量RTL设计验证.md`。

每 head、每 token 的增量资源为：

- 一个 `D=32` K temporal buffer；可与已有 window K buffer 做 bank/地址复用；
- 32-bit XOR；
- 一个 32-bit popcount tree；优先与 TX popcount tree 分时复用；
- 原始分数域的固定`1/4`在 Q7 归一化中被抵消；硬件为整数加法，无通用乘法器；
- score 位宽需在现有上界基础上增加一位并做饱和审计。

H67 仍按 12 块同公式部署。若软件实验失败，`binary_motion_xor_alpha=0` 时旁路该增量，
原 TTX RTL 不变。

## Castling-TTX 的硬件边界

Castling-TTX 只允许训练期存在 H66a full-matrix auxiliary。导出 checkpoint、推理图和 RTL
只能包含 H60。软件验收必须证明：

- 推理模块仍为 12 个 H60 attention；
- checkpoint 不出现 auxiliary-only 参数或其键被显式剥离；
- 推理 FLOP/cycle 模型中没有 `N x N` alpha-XNOR 或 `weights@K`；
- 相同输入下关闭训练辅助后可独立运行标准 valid825。

因此 Castling-TTX 不增加芯片面积，但可作为“训练时富注意力、部署时统一 dyadic TTX”的
算法硬件协同训练方法。

## 有界 gate bundling

对 binary K，跳过 token `i` 的 gate/value 输出在投影前的精确 L1 上界为：

```text
error_i = abs(gate_i) * popcount(K_i)
bundle_error = sum(error_i)
```

只有当整个 4-token 或 8-token bundle 的上界不超过 epsilon 时才允许跳过。硬件需要
popcount 累加、阈值比较、bundle valid mask 和投影输入 clock-gate。它主要节省 K late-scale、
投影输入切换和 SRAM 读；由于 gate 已计算，不能把它错误计为省掉全部 TTX/Shiftmax。

后续若做 progressive TX，则按 8-channel group 产生 score lower/upper bound；在 centered
Shiftmax 下推导完成前，不允许把简单早停宣称为 error-bounded。

## Exact Delta-TTX

对冻结硬件部署图 `alpha0=1/64` 的 binary alpha-XNOR，先整数化
`S64=64*count(q=1,k=1)+count(q=0,k=0)`。每个 channel contribution 只会在 `Q_t`
或 `K_t` 相对上一时间片翻转时改变。t1 只对 `Q_toggle OR K_toggle` 为 1 的 lane 执行
`S64 += contribution_new-contribution_old`；该方法与完整重算逐位等价，不引入近似误差，
也不改变软件 checkpoint。穷举参考模型为 `scripts/ttx_delta_reference.py`。

训练和 float valid825 配置仍是 `alpha0=0.02`；因此上述“逐位等价”只针对同一 checkpoint
切换到 dyadic INT8 部署配置后的推理图，不声称与 float score bit-exact。最终
`run_h60_family_deploy_eval.py` 会把 TTX 与 H67--H71 的 float rank-1 checkpoint 全部用统一
dyadic INT8 配置重评，再决定硬件候选。

注意历史 `INT8` 名称不等于 8-bit 总位宽：score `[-2,2]`、step `1/128` 有 513 levels，精确
端点至少需要 10-bit code；gate `[0,2]` 有 257 levels，至少 9 bit。当前统一部署评估保留
该网格，RTL/PPA 必须按 10-bit score / 9-bit gate（或明确饱和舍弃一个端点）核算。

注意 1-bit match state **不够**：active-active contribution 为 64，silent-silent 为 1。
实现必须二选一：缓存上一时刻 Q/K 共 `2D=64 bit`，或缓存每 lane 的 2-bit contribution
class；同时保留 S64 score accumulator。

是否采用必须由 toggle profile 决定。需报告每 block 的 Q/K/union toggle density，并计入：

- 每 token/head 64-bit previous Q/K，或 64-bit contribution-class state；
- S64 score accumulator，以及 temporal buffer 的现有 bank 复用；
- XOR、valid mask、稀疏 lane scheduler 和 popcount update；
- SRAM 读写及控制发散。

H68 只借用 Castling-ViT 的“训练期重分支、部署期移除”思想，不复现其 masked softmax 或部署
DWConv。由于 H68 auxiliary 没有参数且 eval config 显式设为 0，RTL 不实现 matrix branch。
论文不得把 H68 的结果写成 Castling-ViT mask pruning 的硬件收益。

若 union toggle density 接近 1，Delta-TTX 不值得实现；若显著较低，Motion-XOR 的 XOR
检测器可与 Delta-TTX 复用。不能只报“跳过的 compare 数”而忽略状态 SRAM，这一点按
CVPR 2025 MEET 的 memory-aware temporal execution 范式执行。

这里引用 MEET 只支持一条设计纪律：temporal suppression 的 compute savings 必须扣除 state
memory，否则可能因片外/高频 state traffic 造成能耗反转。MEET 处理线性卷积和 dense
activation state，不直接覆盖 TTX 的 Shiftmax 非线性或稀疏 scheduler。TTX previous Q/K 已是
packed 64-bit binary state，不照搬 MEET 的网络/权重重构；只在 PPA 中采用其 memory-aware
核算原则。

### DSEC 实测（100 samples）

数据源：TTX best epoch2、all12 forward hook，结果目录
`results/date11_ttx_ep2_delta_profile100_exact_20260711`。以下为所有 block、head、window、
sample 的 raw lane count 求和，不是 stage 均值：

| metric | measured density |
|---|---:|
| temporal t1 lanes | 1,741,824,000 |
| Q temporal toggle | 0.7983% |
| K temporal toggle | 1.9946% |
| Q-or-K update union | 2.7832% |
| t1 ideal lane skip | 97.2168% |
| full T=2 ideal TX compare reduction | 48.6084% |

`97.2168%` 只适用于 t1 增量 lane，不能写成整块 attention 节省；整窗 TX compare 的上限
是 `48.6084%`。下一步 PPA 模型必须从该上限扣除 64-bit previous Q/K state、S64 accumulator、
XOR mask、稀疏 scheduler 和 SRAM 访问。如果扣除后净能耗仍为正收益，Delta-TTX 应成为
当前 DATE 硬件创新第一优先级。

## 计量要求

DATE 表必须把以下项目与 neuron SOP 分开列出：TX compare/popcount、Motion XOR/popcount、
Shiftmax max/exp2/denominator、K gate/late-scale、projection activity、SRAM traffic、bundle
control cycles。当前 profiler 未覆盖 overlay 内所有 attention 算子，不能直接用其 SOP 数字
声称 H66/H67 的 attention 总能耗。
## H69 / Delta-Locality 对齐补充（2026-07-11）

软件线新增 H69 Dyadic-Temperature TTX。它不改变 TTX 数据通路，只在 score normalization 后
增加固定 2/3/4-bit 左移；最终只保留一个 scale，因此 RTL 不需要运行时可编程浮点温度。

硬件线新增 Delta-Locality TTX 候选：exact Delta-TTX 先产生 changed-lane index，再以
token/head bundle 压缩索引并驱动现有 popcount lane。此方案受 SPRINT 的动态 locality、
Energon/SpAtten 的 progressive filtering 启发，但不采用近似 score prediction，也不引入
第二种 attention 公式。RTL/PPA 前必须补齐以下 profile：zero-update token/head、更新 lane
直方图、4/8-token bundle 全零比例、index run-length，并从节省中扣除 previous-Q/K state、
index FIFO 和 scheduler 成本。

## H70 Event-Selective TTX 数据通路

H70 在 centered TTX score 与 Shiftmax 之间增加一条统一控制：对同 token 的 binary Q/K 做
OR-popcount，取 `a+1` 的 leading-one/ceil-log2，饱和到 3，驱动 score 左移 0--3 位。所需
模块为一个可与 TX 复用的 OR-popcount、leading-one detector、3-bit shift control 和 score
位宽/饱和审计。无新增参数 SRAM、无第二套 Q/K、无 token-token matrix、无 carrier。

硬件消融至少报告：固定温度 H69 与动态温度 H70 的 score 位宽、Shiftmax 输入动态范围、
额外 OR-popcount 是否可复用、关键路径和能耗。若 H70 软件精度不超过 TTX，该分支不得进入
最终 RTL 主线。

## H71 Window-Context TTX 数据通路

H71 在 `gate*K` 后对每个 window/head/channel 累加 162 个 token，乘固定 `1/162` 得到 context，
再与各 token 各乘 `1/2` 相加。无参数 SRAM 和 QK score matrix，但需要 window reduction、
context register、broadcast network 与固定倒数实现。软件先用精确均值保证精度判断；若晋级
硬件主线，再单独比较 `1/162` constant-multiply 与 power-of-two denominator 近似。

该方案的首要硬件风险不是算术量，而是广播导致 downstream switching/spikes 增加。PPA 表
必须同时报告 reduction/broadcast 开销和后续层活动变化，不能只计算新增加法器。

## Delta-Locality v2 审计协议

profiler 已增加 raw-count 统计：zero-update token/head、更新 lane 数直方图、changed-token
连续 run 和 4/8-token bundle 全零率。`run_delta_locality_after_h71.py` 将在软件 full30 队列
结束后对 TTX ep2 跑 100 samples，并自动回填本文件。微架构选择规则为：高 zero-token 与
高 empty-bundle 支持 sparse index queue；若 changed run 明显大于 1，则优先 run-length/burst；
否则采用 bitmap 或 fixed 8-lane grouped scheduler，避免 RLE 控制开销。

## Attention operation audit 与 spike-energy proxy 边界

standard valid825 的历史 `energy_uj` 仅按 spike activity 计 AC/logic proxy，未覆盖 H67/H69/
H70/H71 新增 attention 算子和 memory。未来表格将其标为 `spike_energy_proxy_uj`。独立脚本
`audit_attention_candidate_ops.py` 按实际 window/head/token/channel 计增量 logic/add/fixed-MAC，
最终由 `run_delta_locality_after_h71.py` 生成统一候选表。该表仍不含 SRAM/NoC，不能替代 RTL
综合与功耗分析。

## 软硬件候选分流与后续注意力块（2026-07-12）

当前 H67--H71 都是软件主线竞争者，必须先完成 full30 + valid825；短跑正常只说明链路可用，
不能直接宣称硬件方向成功，也不能因短跑指标落后就否决。胜出者再进入 RTL，避免同时维护多套
未证实数据通路。

后续全文审计的 P0 软件方向是统一 all12 Dilated-Match TTX：借鉴 EEMFlow 的“近邻稠密、远邻
稀疏”correlation，但只能采用全网相同固定 offset 集，不允许与 H60/SC 分 stage 混用。若进入
实验，它属于 attention block 重做：新增 K halo SRAM/line buffer、offset address generator、
多候选 popcount 和聚合比较器，必须把带宽、边界 padding、吞吐和面积列入主表。它不是现有
Delta-TTX 的免费增量。

硬件线继续分为两类：Exact Delta、zero-activity folding、token-time bundling 属于数值等价
优化；error-bounded bundling、progressive/dilated matching 会改变数值，必须返回软件线完成
full30 与统一部署 valid825。详细状态以
`neuron_autoresearch/TTX_MAINLINE_COMPETITION_LEDGER_20260712.md` 为准。

H66a-e 也已追加 full30，不再用短测淘汰。操作审计按候选公式分别计数：H66a 为
`N^2D` binary comparison、row Shiftmax 和 weighted-K accumulation；H66b 为 `K^T K` 与
`Q(K^T K)` 两次 `D x D` add/sub accumulation；H66c/H66e 为 2-neighbor，H66d 为
5-neighbor。该计数已加入 `audit_attention_candidate_ops.py`，但仍只代表 datapath operation
proxy；矩阵/halo SRAM、NoC、buffer lifetime 和控制功耗必须在 RTL/PPA 表单列。

## 体系结构深读后的数据布局结论（2026-07-12）

LoAS（MICRO'24）的可迁移部分是 timestep-inner fully temporal-parallel layout，而不是其
dual-sparse weight speedup。本模型所有 H60-family block 均为 `T=2`、`head_dim=32`，所以同一
spatial token/head 的 Q0/Q1 可恰好装入一个 64-bit word，K0/K1 另装一个 word。该布局同时服务
普通 TTX、H67 temporal XOR 和 Exact Delta-TTX，避免跨 timestep 重取 operand/state，是当前
第一优先的数值等价数据流优化。

Bishop（ISCA'25）的 dense/sparse 双核在本项目中尚无面积依据，必须等待 locality v2 的
bundle4/8 空闲率和 changed-lane histogram；若密度不呈双峰，只做 clock-gated 单路径。PADE
的 INT8 K bit-plane pruning 不适用于已经是一位的 H60 Q/K，只在 H66a 全矩阵 full30 胜出后
才可能作为近似候选重新讨论。ICCAD'24 3D memory-on-logic 只作为物理设计相关工作，不在没有
3D PDK/一致 flow 时引用其节能数值。完整审计见
`neuron_autoresearch/literature/idea_mining_20260711/notes/HARDWARE_TRANSFER_AUDIT_20260712.md`。

事务模型已落盘：`hw_autoresearch_nts07/results/ttx_temporal_pair_layout_model.md`。每
window/head 的 Q/K 逻辑数据恒为 10368 bit；若 baseline 在 64-bit 接口上分别发出两个 32-bit
时间片请求，则由 324 次请求降到 162 次、接口传输由 20736 bit 降到 10368 bit。若 baseline
本来已跨时间合并，则 packed layout 的请求和流量收益均为 0。论文只能在地址 trace 证明当前
baseline 未合并后声称 50% transaction/traffic reduction；无论哪种情况都不能声称 storage
capacity 下降。
