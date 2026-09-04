# TTX 主线竞争与软硬件双线研究账本（2026-07-12）

## 1. 目标与冻结边界

目标不是证明某个短测可用，而是在 DSEC 标准 full30 + valid825 下找到可替代当前 H60 TTX 的统一 all12 主线。当前冻结参照为 one-sided all-binary ATLIF + all12 H60 TTX：float AEE 约 1.5003；dyadic deployment（alpha0=1/64）AEE 1.5016、AAE 9.8431、total_spikes 23.2439G。

所有竞争者必须满足：

- 12 个 encoder attention block 使用同一公式，不采用 S2-only 或 TX/SC 分 stage 混合。
- 神经元保持 one-sided `{0,+theta}` ATLIF；若以后引入正负发放，则正负阈值必须对称。
- 禁止恢复旧 native carrier 路径；允许 H60 既有的 `gate * K` 输出语义。
- 旧模块、旧配置和旧 checkpoint 只读保留；新机制只能通过默认关闭的可选分支加入。
- 短跑只检查实现、数值稳定和候选超参，不能据此宣布算法失败。

## 2. 软件线：主线 checkpoint 竞争

共同训练协议：从同一 TTX epoch2 checkpoint 独立续训，DSEC crop 288x384，batch8，workers8，AMP/cupy，30 epochs，warmup720，里程碑 20/25；标准 valid825 评估 epoch 0/4/9/14/19/24/28/29。主线晋级同时比较 float 和统一 dyadic deployment，不用 spike-only energy proxy 冒充 attention 全能耗。

| 顺序 | ID | 机制 | full30 状态 | valid825 状态 | 主线判据 |
|---:|---|---|---|---|---|
| 0 | H60 TTX | frozen dyadic Shiftmax + gate*K | 已完成 | 已完成 | 当前参照 |
| 1 | H67 Motion-XOR TTX | score 加同位置相邻时刻 K 的 XOR-popcount，权重 1/4 | 已完成 | 已完成；dyadic ep19 AEE1.4626/AAE9.3949/26.3948G | 当前精度第一候选；需用完整PPA判断额外XOR/popcount是否值得 |
| 2 | H68 Castling-inspired TTX | 训练期 alpha-XNOR matrix 辅助线性退火至 0，部署回 H60 | 已完成 | 已完成；dyadic ep19 AEE1.4715/AAE9.4517/26.4311G | 部署图最简单的精度候选；推理期与H60相同 |
| 3 | H69 Dyadic-Temperature TTX | 固定 4/8/16 倍二次幂 score 温度，短筛只选温度 | x8 已晋级，full30 运行中 | 待运行 | 三个 short360 均未过 AAE 门；按预注册综合 score 选 x8，仍以 full30 作结论 |
| 4 | H70 Event-Selective TTX | OR-popcount 活性决定每 token 的二次幂逆温度 | 已排队 | 待运行 | 精度提升覆盖额外 LOD/shift 代价 |
| 5 | H71 Window-Context TTX | `(gate*K + mean_token(gate*K))/2` | 已排队 | 待运行 | 精度提升覆盖窗口 reduce/broadcast 代价 |
| 6 | H66a Alpha-XNOR Matrix | `N x N` binary alpha-XNOR + row Shiftmax + weights@K | H71 后已排队 | 待运行 | 精度上界；必须计入矩阵 SRAM/`N^2D` 成本 |
| 7 | H66b Hamming Linear | `Q(K^T K)`，binary 0/1 映射为 -1/+1 | H66a 后已排队 | 待运行 | 必须计入两次 `D x D` 累加及 silence 语义 |
| 8 | H66c Temporal-Pair TTX | self + 相邻时间同位置两个 K 候选 | H66b 后已排队 | 待运行 | 2-neighbor 低成本 pairwise 对照 |
| 9 | H66d Local-5 TTX | self + 四个轴向空间邻居 | H66c 后已排队 | 待运行 | 5-neighbor 精度/halo 代价权衡 |
| 10 | H66e Temporal-Pair Self-Bias | H66c 加固定 self lane bias=1 | H66d 后已排队 | 待运行 | 固定偏置是否恢复方向指标 |
| 11 | H73 DE9 Match-Code | 9 个跨时 offset，分别保留 event-event 与 silence-silence 共 18 维描述子，再做静态 per-head 投影 | H66e 后已排队 | 待运行 | 无动态 K carrier；精度收益必须覆盖 18xD 静态投影 |
| 12 | H74 MC49 Match-Code | EEMFlow 风格固定 49-offset 跨时匹配描述子，再做静态 per-head 投影 | H73 后已排队 | 待运行 | 精度上界候选；必须核算 halo、49 路 popcount 和 49xD 投影 |
| 13 | H75 AX17 Match-Code | Flow1D 启发的横/纵半径4轴向跨时匹配，中心共享共17路 | H74 后已排队 | 待运行 | 以17路成本覆盖大位移，但不具备完整二维联合 offset |
| 14 | H76 PC9 Match-Code | 固定 3x3 跨时 patch-consistency，9 个 offset 使用静态 per-head codebook | H75 后已排队 | 待运行 | 检验局部二维一致性是否比大 offset 集更有效；保持无动态 carrier |
| 15 | H77 LC4 Match-Code | 学习 `n11/n10/n01/n00` 四类二值列联证据的 dyadic 系数 | H76 后已排队 | 待运行 | 最小统计量候选；收益必须覆盖 12 组 LC4 系数和控制 |
| 16 | H78 G4 Match-Code | 32 lanes 固定分四个 8-bit group，分别产生 Match-Code 后静态投影 | H77 后已排队 | 待运行 | 验证分组保留局部通道信息是否优于整头 popcount |
| 17 | H79 CF10 Match-Code | Omega9 row Shiftmax 加 fixed-zero null；null evidence由top2 margin与query activity生成 | H78 后已排队 | 待运行 | 显式建模unmatched/occlusion；必须排除null塌缩并核算top2与beta乘法 |
| 18 | H80 DN9 Match-Code | Omega9 row Shiftmax与destination incoming Shiftmax的Q1.7乘积 | H79 后已排队 | 待运行 | 显式目标端竞争；必须证明双归一化精度收益覆盖第二套Shiftmax与edge product |

主线替代的最低证据：valid825 AEE 不高于 NB0 的 1.05 倍，total_spikes 至少比 NB0 下降 20%；在满足门槛者中，先按 AEE、再按 AAE、再按包含 attention/control/memory 的硬件成本排序。随机种子复验只对最终前二名运行，不对所有候选盲目三种子扩张。

## 3. 软件孵化区：后续全文审计方向

以下方向尚未进入当前训练队列，不得写成已验证贡献：

| 优先级 | 暂定方向 | 论文迁移点 | 最小统一实现 | 进入 full30 前必须解决 |
|---:|---|---|---|---|
| P0 | Fixed Sparse-Offset Match TTX | EEMFlow 的稠密近邻、稀疏远邻 correlation（官方实际保留 49/53 个 9x9 offset，不能直接照搬） | 每个 all12 window 使用预注册的小型固定 offset 集做二值匹配，再用同一 Shiftmax/gate 规则 | offset 数、聚合规则、Swin 边界语义和 SRAM halo 代价 |
| P0 | Spike-Difference Guided TTX v2 | EDCFlow 的 temporally dense difference maps | 不另建 cost volume，把多 bin 差分压成 dyadic score bias | 与 H67 单一 K-XOR 的非重复性及算子计数 |
| P1 | Latent-Key TTX | FlowFormer latent cost token 压缩 | window 内固定 M 个 pooled binary keys，所有 block 同一 M | pooling 后二值语义、回写规则和精度风险 |
| P2 | Attention-Diversity TTX | pixel-level token reduction 对完整 token 拓扑的警告 | 需要另找真正部署零开销的 score regularizer；DAR-TR 官方实现包含 router/adapter/compensator，不直接采用 | loss 定义、是否真正改善光流边界 |

孵化方向完成“全文公式 + 官方代码路径 + 迁移公式 + 硬件操作表 + 单元测试”后，才分配新 H 编号并串到 H71 之后。仍然必须跑 full30；健康检查通过不等于成功，健康检查偏差也不等于失败。

## 4. 硬件线：与软件候选解耦

硬件线分两类：A 类保持 checkpoint 数值严格等价，可直接叠加到胜出主线；B 类改变数值，必须回到软件线做 full30 + valid825。

| 类别 | 方向 | 数值关系 | 当前动作 |
|---|---|---|---|
| A | Exact Delta-TTX | alpha0=1/64 时 `S64=64*n11+n00` bit-exact | 统计 lane update、changed-token run、bundle 空闲率 |
| A | Zero-Activity Folding | 对已证明为零的 token/lane 做 clock/power gating | 用 profile 原始计数估计，不改模型输出 |
| A | Token-Time Bundle | 只改排布与复用 | 统计 bundle4/8 空闲和带宽 |
| A | 64-bit temporal-pair co-residency | T=2、head_dim=32 时 Q0/Q1 或 K0/K1 恰好打包一字 | 建立地址 trace、traffic 和 cycle model |
| B | Error-Bounded Gate Bundling | 近似合并 gate/score | 先建立误差界，再做 full30 或部署校准 |
| B | Progressive/Dilated Matching | 提前终止或固定稀疏 offset | 与 Dilated-Match TTX 共设计并完整训练 |

硬件表必须分别报告 ATLIF、Q/K popcount、Shiftmax/reduction、gate*K、候选额外逻辑、SRAM/DRAM traffic；`total_spikes * pJ` 只称 spike-activity proxy。

2026-07-12 深读裁决见 `literature/idea_mining_20260711/notes/HARDWARE_TRANSFER_AUDIT_20260712.md`：LoAS 的 temporal-inner/packed-spike 数据布局可作为 exact TTX 优化；PADE 的 INT8 bit-plane pruning 与一位 Q/K 不匹配；Bishop 双核必须等待 bundle density profile；ICCAD'24 3D 集成只作物理实现讨论，不作为已实现节能。

## 5. 运行与存盘纪律

H67/H68 full30+valid825 已完成；H69 x8 full30 正在运行，H70/H71、H66a-e、H73-H80 串行等待。H68-H80 full30 新增 `runtime.save_only_force_epochs: true`，仅保存预注册评估轮次和末轮；`runtime.state_save_epochs: [19,24,29]` 只在三个可续训节点保存 optimizer/scaler state。两个开关默认不启用，不改变旧实验。H66a-e 的旧 120/360-step 只作实现证据，五个结构合规候选已通过 `run_h66_full30_after_h71.py` 串到 H71 后。H69/H70/H71 runner 已增加完整 short-screen、epoch29 与 valid825 ranking 的完成态复用，watcher 重启不会重复已完成阶段。H79/H80由`run_round4_assignment_after_h78.py`严格等待H78完成标记后逐项运行；TTB-v2与最终deploy watcher继续后移到H80完成标记。每候选保留 8 个模型 checkpoint 和 3 个训练 state，预计约 7.2GB；已有额外中间 checkpoint 暂不删除，任何清理先取得用户同意。

每个 full30 完成后必须自动写回：配置、起点 checkpoint、ATLIF/Shiftmax 数量、overlay load audit、训练轨迹、valid825 排名、spikes、energy scope、部署量化结果和硬件增量。只有这些项齐全后才能标记“已完成”。

## 6. H73/H74 Match-Code 预注册与加载审计

DE9/MC49 已从孵化区晋级正式 full30，不再用 short 指标淘汰。两者都保持 one-sided binary
ATLIF105、all12 同一 attention 公式，并删除动态 `weights@K`/native carrier；输出由跨时间
匹配描述子乘静态 per-head codebook 生成。共同起点仍是冻结 TTX epoch2，训练协议和八个
valid825 epoch 与 H66--H71 完全一致。

- generator: `entrypoints/make_h73_h74_match_code_configs.py`
- queue: `entrypoints/run_match_code_after_h66.py`
- manifest: `configs/generated/h73_h74_match_code_full30_manifest.json`
- load audit: `neuron_autoresearch/experiments/h73_h74_match_code/load_chain_audit.json`

CPU 全链路实测：三项均安装 ATLIF105、attention12、Match-Code12；TTX checkpoint 含 overlay
keys210，warm-start 唯一 missing 为新 codebook 12，unexpected0、non-Match missing0。DE9 新增
79,488 参数，MC49 新增216,384参数，AX17 新增75,072参数。训练生成的 Match-Code checkpoint 在 valid825 阶段恢复
严格加载，不允许缺任一 codebook。最终 Delta/deploy watcher 已后移至三项 full30 完成之后。

## 7. H76-H78 Round3 Match-Code 预注册与加载审计

H76 PC9、H77 LC4、H78 G4 已生成 full30 配置、manifest、公式单元测试和 frozen-TTX 加载审计。
三者都保持 ATLIF105、all12 attention12、one-sided binary 输出和无 native carrier；共同从同一
TTX epoch2 独立起跑，不继承 H73-H75 的训练权重。warm-start 的唯一 missing 分别为 H76/H78
新 codebook 12，以及 H77 codebook12 加 LC4 参数12；unexpected 均为0。训练后还会逐项执行
strict missing0/unexpected0 审计，再运行八个预注册 epoch 的标准 valid825。

- generator: `entrypoints/make_h76_h78_round3_match_configs.py`
- queue: `entrypoints/run_round3_match_after_h75.py`
- manifest: `configs/generated/h76_h78_round3_match_full30_manifest.json`
- load audit: `neuron_autoresearch/experiments/h76_h78_round3_match/load_chain_audit.json`

## 8. H79-H80 Round4 Assignment 预注册与加载审计

H79 CF10与H80 DN9均已作为默认关闭的可选分支实现，保持ATLIF105、attention12、all12统一
公式、静态per-head codebook输出和无native K/V carrier。H79只存9行codebook，第10个null
codeword由forward硬接零；另有每head两个`1/64`网格beta。H80复用相同Omega9 score，同时按每个
destination的4/6/9条合法incoming edge执行第二次Shiftmax，并将row/destination gate乘积量化到
unsigned Q1.7。

- generator: `entrypoints/make_h79_h80_round4_assignment_configs.py`
- queue: `entrypoints/run_round4_assignment_after_h78.py`
- manifest: `configs/generated/h79_h80_round4_assignment_full30_manifest.json`
- load audit: `neuron_autoresearch/experiments/h79_h80_round4_assignment/load_chain_audit.json`

实测52项attention公式测试及5项加载/优化器测试均通过。冻结TTX warm-start审计：两项均为
ATLIF105、attention12、candidate12、checkpoint overlay210、unexpected0；H79唯一missing为12个
codebook加12个CF10 beta，H80唯一missing为12个codebook，同模式注册state严格重载均为0/0。
训练入口、标准推理加载、optimizer new-module分组和SOP profiler均已纳入`_h9_cf10_beta`，避免
独立verifier通过而正式训练入口拒绝或beta误用backbone LR。
