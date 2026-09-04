# M1529｜ep34 当前创新性与稀疏跳过第一性原理审阅

日期：2026-08-31（Asia/Shanghai）  
性质：独立、只读、fail-closed 证据审阅  
对象：Motion H67，已封存 ep34 checkpoint 与 M1458 40-sample capture  
裁决：**SHARE_WITH_CAVEATS；保留 C1/C2/C3，只授权两个 CPU 快杀，不授权新 RTL。**

本审阅没有运行 GPU、EDA、VCS、SSH 或训练，没有修改任何生产结果、旧 review、论文、`ucli.key` 或 `docs/359_DATE终局冻结_20260813.md`。`docs/359` SHA256 复核为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 1. 结论先行

当前三个硬件点不是“每层各一个小技巧”，而是一套可辩护但仍未达到 Strong Accept 完整度的组合：

1. **C1 有明显对象差和性能意义。**真正可辩护的是：在 240-KiB-class、单口 parent/PWP、dead-write 与 completion 约束下，把 Prosperity/Phi 式重复产品机会转成 H67 signed-source Conv 的有限容量执行。旧冻结十样本 CPU 同账本机会为 `1.7591725402x`，但尚未由 ep34 重绑后的 RTL-cycle/system row 接管。
2. **C2 有明确协议差和很强面积效率。**typed signed K8、共享 Acc24/端点和 fault-closed completion 不等于重新命名 FireFly-T/ELSA。等带宽 K8 对 K1x8 周期仅 `1.01672765x`，但吞吐/mm2 为 `4.541078x`、logic area 低 `77.6104%`；论文必须把三者同句写出。
3. **C3 物理验证强、创新性和性能意义弱。**Fixed-T10 已闭合 prelayout logic-only DC/PT/映射等价，但没有 throughput、speedup、power 或 energy。它适合做 exact neuron-service 与系统完整性，不适合单独当性能 headline。

因此当前整体硬件 novelty 约 **3.5/5**：足以组成 DATE Accept 叙事，但不是仅靠创新点名称就能到 Strong Accept。真正的缺口仍是 ep34-bound、decoder-complete、memory-inclusive 的统一周期/能量行。

新探索只保留两条：

- **A｜typed signed bundle-Gustavson 跨 token 权重行复用**：无损，作为 C2 的 weight-traffic/energy 子机制；核心祖先是 ELSA/SCNN 类 Gustavson 与 bundled AER，不能宣称发明。
- **B｜support-conditioned certified block skip**：有损，把 Bishop/Phi 式误差预算迁到占比更高的 patch/FC/Conv，并且在 weight fetch 前作整块跳过；原始 `weight-block L1 x activation Linf` 版本过粗，必须改成对 binary-sparse 激活有效的 `block max-weight x activation L1` 上界。

两条都先 CPU 快杀。没有同资源周期/字节收益和端到端 Delta-AEE，不值得写 RTL。

## 2. 当前 C1/C2/C3 独立评分

评分是当前可引用证据的质量，不是最终论文潜力。

| 线 | Novelty /5 | Significance /5 | Validation /5 | 独立判断 |
|---|---:|---:|---:|---|
| **C1 finite-capacity product capture** | **3.8** | **3.9** | **3.2** | 对象差最清楚：signed ATLIF/source ownership、单 1RW、有限容量、dead-write、completion。旧 CPU 同账本 `1.759x` 有意义，九宏 28-nm setup/area 也存在；但 ep34 重绑、RTL-cycle、hold/full storage/power 未闭，新 M1506 promotion 因 witness/termination seam 与 exact-II2 cover 缺失保持 FAIL，不能升级。 |
| **C2 typed signed K8 shared service** | **3.5** | **3.7** | **3.8** | 不是新稀疏数学，而是 binary/multi-amplitude/signed source 的统一协议与共享状态。等带宽周期仅 `1.017x`，但 `4.541x throughput/mm2` 与 `-77.61%` logic area 很强；生产 SAIF/PTPX 仍无可引用结果。 |
| **C3 Fixed-T10 exact service** | **2.8** | **2.5** | **4.3** | Fixed-T10/phase service 的算法新意有限，且无加速比；但 M1473/M1479 已闭合 28-nm prelayout logic-only DC setup/hold、独立 PT 与 11,180 mapped compare points，验证是三条中最完整的。 |

### 可写成论文贡献的强度

- C1 可作为第一性能贡献，但必须标清旧 `1.759x` 是 pre-rebind CPU same-ledger component opportunity。
- C2 可作为第二架构贡献，headline 是**等服务带宽下的面积效率**，不是 K8 对单 K1 的并行倍率。
- C3 更适合作为第三条“exact neuron/system closure”；若没有 throughput/energy，不要把它写成与 C1 同等级的性能创新。

## 3. 与公开工作的合法边界

| 原工作 | 原机制 | 本项目能主张的对象/约束差 | 不能主张 |
|---|---|---|---|
| Prosperity | product sparsity/reuse 与调度 | C1 在 H67 signed source、source-owned parent、单 1RW、有限 scratch、dead-write/completion 下的 capture gap | 不能把官方 `2.46x` 或原论文 `7.4x` 写成 ours |
| Phi | L1 exact pattern，L2 residual drop + PAFT | 有限 240-KiB-class 条件与 H67 typed signed residual；只允许把 PAFT 作为有损对标 | 不能把 M70--M76 nominal pattern-op 当系统倍速；当前 clean PAFT/Pareto 未准入 |
| FireFly-T | multi-spike decoder、负载均衡并行 | C2 的 typed signed value、Acc24 atomic update、equal-bandwidth shared endpoint | 不能把 K8 对单 K1 的 4--6x 写成稀疏收益 |
| ELSA / SCNN-family | spiking Gustavson、bundled event flow、稀疏累加 | 候选 A 的跨 token signed value bundle、有限 destination contexts 与相同 K8/port 约束 | 只换名字或把 weight-row multicast 称为新算法 |
| SNE / neuron-synapse fusion | neuron output 直接驱动 synapse | C2/C3 的 ATLIF typed source 与不物化接口 | 普通 fusion/FIFO 本身不是 novelty |
| Bishop | error-constrained pruning | 候选 B 把 certified budget 放到非 attention 的 Conv/FC weight-fetch-before-compute gate | 误差上界数学、阈值剪枝本身不能称首创 |
| DeltaCNN | temporal delta/update mask | 只能作为负结果/相关工作比较 | decoder exact temporal XOR 已更密，不能改名复活 |

## 4. 已死亡或只配支撑的机制：不得复活

| 机制 | 现有证据/第一性原理原因 | 裁决 |
|---|---|---|
| G7 bottleneck 幅度门 | 冻结输入基本是 `{0, layer-constant}`，没有可用中间幅值 Pareto | KILL；有损预算改打权重贡献，不再按 activation magnitude 造 RTL |
| G8 整 token FFN skip | 冻结 tau 网格的 site-level exact 机会为 0；post-hoc 大 tau 只有 oracle，缺 executable skip 与 Delta-AEE | KILL 原 G8；不得把 oracle Amdahl 当 speedup |
| G10 N=0 空 tile | 空 output-site 仅约 `0.1117%`，task 空率也不足以撑主加速 | KILL as performance；最多保留 fetch-energy 诊断 |
| G11 cumulative source budget | 对 binary 值域与既有 zero/source skip 高度重叠，收费端口后没有可靠 >=1.15x | KILL 原实现 |
| G12 ATLIF remaining-budget early stop | S10 term skip `6.58%`，但 32-lane issue cycle 仅降 `0.0676%`，fixed projection `1.00008x` | KILL RTL；只作负消融 |
| G15 cost-aware direct parent | 同资源最好约 `1.07x`，容量/端口门不过 | KILL |
| M501 adjacent overlap/APEC | event-work `1.3796x`，理想全 envelope 仅 `1.0366x`；trace 退化为 positive support，且 ExSpike 是直接 prior | 不做 standalone novelty；最多作为候选 A 的负/上界对照 |
| M523 跨 parent 拼包 | 真实机会约 `1.007--1.034x`，只闭合 descriptor 功能 | SUPPORT decoder completeness only |
| Decoder temporal XOR / N2 | 相邻 XOR 相对 full active source 加权约 `1.352x` 更密 | KILL |
| M70--M76 Phi/PAFT headline | M72 nominal `1.503x`，乐观 32-B port+matcher 约 `1.259x`；catalog/valid split/PAFT accuracy 身份曾不满足正式门 | 不复活为主线；只作 related-work/负 Pareto |
| rank-3 ATLIF | 当前 exact T10 矩阵不是 rank-3 子集，且无 admitted accuracy | KILL for current checkpoint |
| RQTB/epsilon-RQTB 系统主线 | attention 工作份额太小；即便无限加速也无法支撑系统 headline | SUPPORT local energy/completeness only |

## 5. M1458 ep34 capture 当前能测什么

M1458 已封存 40 个 sample、9,880 个 ordered records，其中 `operator_runtime.json` 为 79 个 Conv2d/Linear module aggregate，`execution_trace.json` 为 7,360 个逐 call aggregate row，`unified_ordered_records.jsonl` 含 160 个 C1 与 160 个 decoder retained raw/support payload，其余类别主要是 ordered statistics。

### 可以直接测

1. 每 call/module 的 shape、input active/elements、全局顺序、sequence/sample identity。
2. 79 个已跟踪 Conv2d/Linear 的 dense MAC 与 activity-weighted-MAC proxy；只能用于优先级，不是 cycle 或 full-network denominator。
3. C1 四层和 decoder D0--D3 的精确 retained FP32/support-sign payload，可做 exact local replay、group occupancy、source/tap 计数与 local error-bound 检查。
4. decoder 120 个正式三序列 record 加 C1 cohort 的 40 个 decoder record；M1511 又证明 D0/D1/D2/D3 各层只有一个稳定正码字，负值/非有限值均为 0。
5. ep34 ordered-statistics 密度诊断：C1 `10.079%`、decoder `20.923%`、FC1 `12.310%`、FC2 `3.154%`、patch category `17.359%`。这些是 input-nonzero density，不是 skip rate 或 speedup。

在 79-module activity-weighted-MAC proxy 内，patch-embed family、FC1、FC2、C1 bottleneck 分别约 `40.58%/23.39%/6.91%/15.64%`。该四项用于说明为什么新有损机制应优先打非 attention；它们不是 decoder-complete system shares。

### 当前不能测

1. FC1/FC2/patch 的逐 token/逐 channel 精确 support/value bitmap：这些 record 的 payload 标为 `ordered statistics only`、`retained=false`。
2. 跨 token 同一 input-channel 的真实 overlap、weight-row reuse distance、bundle occupancy 与 signed/non-unit bundle 比例。
3. 任意候选的 address-timed weight/activation/psum transaction、bank conflict、queue/backpressure、destination-context stall。
4. 候选 B 的端到端 Delta-AEE：M1458 不含候选 forward 输出、ground-truth 对齐或逐 epsilon 的预测结果。
5. full network 的 exact per-layer propagated error；局部 norm bound 不是 optical-flow AEE bound。

因此 M1458 足够做 retained C1/decoder 的**局部快杀**，不足以把 A/B 升为全网性能或精度结果。

## 6. 候选 A｜typed signed bundle-Gustavson 跨 token 权重行复用

### 6.1 机制与合法 novelty

对一组相邻/同 window token，把相同 input-channel 的活动 source 合成一个 bundle；weight row/tile 只取一次，随后把同一权重广播到多个 destination Acc24 context。source value、token/destination id、terminal 与 sign/magnitude 继续走 C2 typed descriptor。

它减少的是 **weight row/tile fetch**，不是 MAC 数下界。若同一权重已在 baseline row buffer 中，或者 B 个 destination update 仍需串行 B 次，则周期不会凭空减少，只可能节能。

可辩护对象差仅在：

- H67 的 typed signed/non-unit source，而不是只支持 binary spike；
- 同一 K8 weight banks、240-KiB 总预算和有限 destination contexts；
- bundle 构造、row-buffer、B 路 Acc24 更新与 completion 全收费。

若 ep34 真负载中 signed/non-unit bundle 覆盖很低，该点就是 ELSA/SCNN 的 binary workload migration，不能升为独立 novelty。

### 6.2 现有数据给出的上限提示

仅按 M1458 aggregate density、假设 token 独立且 bundle B=8，理想 unique-row/event 比约为：FC1 `1.51x`、FC2 `1.12x`；这只是统计代理，不包含空间相关性、row cache、destination update 或端口。M501 在另一组相邻 Conv 上测到 `1.3796x event-work` 却只有 `1.0366x` 理想 envelope sensitivity，说明 overlap 很容易被 Amdahl 与执行税吃掉。

### 6.3 缺失 capture

只需新增轻量、只读 payload，不重训 checkpoint：

- 12 个 FC1/FC2 模块、40 sample 的 per-token x per-input-channel support bitset；
- 非零 value 的 INT/Q-code、sign 与 exact non-unit 标记；
- token/window/spatial order、consumer output-tile identity；
- 每层量化 weight identity 与 row/tile byte layout；
- baseline row-buffer hit/miss 或可由相同 simulator 重建的 address key。

### 6.4 48 小时 CPU 快杀门

同一 ep34 capture、K8/K1x8、bank/port/BW、240 KiB 和 output commit；baseline 必须拥有相同容量的普通 weight-row buffer，禁止把 baseline cache miss 当 novelty。

1. **Exactness：** contributor multiset、INT Acc24 与 output 必须 0 mismatch。
2. **收费项：** bundle search/sort、row-buffer、B destination contexts、update cycles、bank conflict、tail、metadata 与 weight refill 全计入。
3. **性能门：** FC1+FC2 ratio-of-sums same-resource cycle `>=1.15x`，且每条 sequence `>=1.05x`；或者周期不回退超过 5%，weight bytes `>=30%` 减少、包含 metadata 后的 memory energy `>=20%` 减少。
4. **差异化门：** signed 或 non-unit source 至少占 admitted bundles 的 `5%`；否则只准写 binary ELSA-style support，不准称 H67-native novelty。
5. **RTL 门：**只有 1--4 全过，才做一个 B2/B4 的最小 C2 bundle frontend RTL；否则 `NO_RTL`，留 energy/negative ablation。

当前裁决：**CPU_FASTKILL_ONLY，novelty ceiling 约 3.1/5；未授权 RTL。**

## 7. 候选 B｜support-conditioned certified block skip

### 7.1 否决原始粗界

原提案对每个 weight block 存 `||W_b||_1`，运行时用 `||x_b||_inf`，若乘积小于预算则跳过。这个界数学上可用，但对当前大量 `{0,1}`/`{0,theta}` 激活过粗：任何非空 block 的 `||x_b||_inf` 几乎恒为 1/theta，无法区分 1 个 active 与 16 个 active；最终只剩空块跳过，与现有 exact zero-source skip 重叠。

因此原形式 **NO-GO**。

### 7.2 修正版与可证界

把输入分成 source group `G`、输出分成 tile `O`。离线每个 `(G,O)` 只存一个定点 metadata：

`M_GO = max_{i in O, j in G} |W_ij|`。

运行时从已存在的 source descriptor 累积：

`A_G = sum_{j in G} |x_j|`。

若丢弃整个 `(G,O)` weight block，则有：

`||Delta y_O||_inf <= M_GO * A_G`。

对多个被跳 block，硬件累加 bound debt，只有 `sum(M_GO*A_G) <= epsilon_layer,tile` 才跳过 weight fetch、compute 与相应 psum update。`epsilon=0` 时只允许 bound 为 0 的 block，严格退化为 exact C2/C1 子集。

该界对 binary sparse activation 能看到 active count，不会像 Linf 一样非空即饱和。16x16 INT8 weight block 若用 16-bit metadata，静态 metadata 约为 weight bytes 的 `0.78%`；真实 bank/port/读取能量仍必须收费。

这不是新的误差界。合法对象差是：把 Bishop/Phi 类 error budget 放到 H67 非-attention Conv/FC 的**fetch-before-compute block gate**，并复用 C2 source accumulator；论文必须同时给 local certified bound 与 measured Delta-AEE，不能把前者说成 AEE 上界。

### 7.3 当前能做与缺失数据

- M1458 retained C1/decoder payload + checkpoint weights可做首轮 local bound/skip/weight-byte fast-kill。
- FC1/FC2/patch 要求与候选 A 相同的 per-group support/value payload。
- 端到端 accuracy 需要在算法服务器用冻结 ep34 做 epsilon grid inference，保存 sample/sequence AEE、输出 flow SHA、实际 skipped block/weight bytes；不需要训练新 checkpoint。

### 7.4 48 小时 CPU/推理快杀门

1. block size 枚举 `{8x16,16x16,32x16}`，epsilon 必须包含 0，并以同一 ep34 checkpoint、同一量化/rounding 执行；bound violation 必须 0。
2. 每点同时报告 skipped weight block、weight bytes、compute issue、psum update、metadata bytes/read、bound debt distribution；只减 MAC 不减 fetch 的点不准晋级。
3. 高份额 patch+FC1+FC2+C1 的 ratio-of-sums same-resource local cycle `>=1.15x`，或 cycle 不回退超过 5% 且 weight bytes `>=30%`、包含 metadata/controller 后的 memory-energy proxy `>=20%`。
4. accuracy 门：全体 `Delta-AEE <= 0.02`，且任一 sequence `Delta-AEE <= 0.03`；报告 AEE 悬崖，不以均值掩盖序列退化。
5. `epsilon=0` 必须逐位对回 exact baseline；有损行与 C1/C2/C3 exact 行分表，禁止相乘。
6. 只有 1--5 全过，才做一个 `M_GO` ROM + bound accumulator + prefetch-kill 的最小 RTL；否则不做 RTL。

当前裁决：**最高优先级 lossy CPU/forward fast-kill，潜在 significance 高于 A，但 novelty 只在“非-attention + fetch-before-compute + typed source”组合，未授权 RTL。**

## 8. 最小 capture 增量

不要重做 M1458，也不要重新抓全部张量。最小新增包为：

| 字段 | A 需要 | B 需要 | 原因 |
|---|---:|---:|---|
| FC1/FC2/patch per-token/channel support bitset | 是 | 是 | overlap、group occupancy、exact source work |
| nonzero INT value/sign/non-unit code | 是 | 是 | typed bundle 差异化、`A_G` bound |
| token/window/spatial/global order | 是 | 是 | bounded reorder、same-resource schedule |
| quantized weight block identity/layout | 是 | 是 | weight-row bytes、`M_GO`、bank key |
| baseline address/bank/row-buffer key | 是 | 是 | 避免弱 cache baseline |
| candidate output flow + GT alignment + per-sequence AEE | 否 | 是 | Delta-AEE/Pareto |
| skipped block/bytes/issue counters | 否 | 是 | 证明 skip 同时打 fetch+compute |

新增 capture 必须继续绑定 checkpoint SHA `4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48`、原 40 sample SHA/order和 M1458 outer provenance；它是增量 payload，不是第二份“新 checkpoint capture”。

## 9. 48 小时执行顺序

### 0--8 小时

1. 用 retained C1/decoder 跑 B 的 local bound fast-kill，先判断粗粒度 block 是否能在低 epsilon 下减 `>=30%` weight bytes。
2. 同时定义 A/B 共用的 FC/patch bitset/value 增量 capture schema；只准备，不抢占正在进行的 MVSEC/full-sequence 流。

### 8--24 小时

1. 在 GPU 空闲后执行一次增量 capture；若不可用，不阻塞 C1/decoder 的 Table-A replay。
2. A 跑 B2/B4/B8 same-resource CPU schedule，强 baseline 带同容量 row buffer。
3. B 跑 block/epsilon local grid，输出 certified bound 与 bytes/cycle proxy。

### 24--48 小时

1. 只有 B 的 local gate 先通过，才跑 ep34 end-to-end epsilon inference与 Delta-AEE；失败立即封 NO-GO。
2. A/B 独立 hammer；最多一条升级为 supporting RTL candidate。
3. 任何新点不得阻塞 ep34 decoder-complete replay、Table-A 和能量收口。

## 10. 论文位置与最终裁决

即使 A/B 过门，也不新增第四条并列贡献：

- A 写入 **C2**：weight-row multicast/bundle 是 typed signed source fabric 的 memory specialization。
- B 写成 **C1+C2 可选 lossy mode**：epsilon=0 是 exact 子集；有损 Pareto 单独一表。
- C3 保持 exact neuron closure，不与 A/B 叠乘。

当前 DATE 模拟判断：

- **Novelty：3.5/5**，C1/C2 足够辨识，C3 偏支撑。
- **Significance：3.4/5**，局部数字可写，但 ep34/system/energy 尚未闭合。
- **Validation：3.7/5**，C2/C3 物理证据较强，C1 新 promotion 与 power/System 仍缺。

现阶段可争取 DATE Accept，但不是 Strong Accept ready。A/B 的意义不是堆 idea 数，而是尝试给高份额非-attention 算子补一条可测的 weight-fetch/energy 优势。若 48 小时门不过，应立即杀掉并专注现有 C1/C2/C3 + Table-A；这比保留一个无数字的“稀疏剪枝创新”更有利于接收。

## 11. 证据身份

| 对象 | SHA256 |
|---|---|
| paper skeleton `main.tex` | `8240b35830ab50e5f132843725585416053fbf8997c2b15374831740d6e18900` |
| M1266 Strong-Accept evidence audit | `63f24a348e182a212585886e794ba9fc5e365332265a331ca2088cd0d923271d` |
| M709 first-principles audit | `e61c01d256e6bf07329407603d17f2d4f36c88e00149908de2c49e109352da71` |
| M1458 manifest | `3ab8431e3d7d17d6933c0b87da4a3405e87c97ccc302a27c78491b0a02491d6d` |
| M1458 operator runtime | `eb0cd40e701361f8acc08d6003680de0ca35626e8e75dcf56827c978899e8a8e` |
| M1458 execution trace | `55759fb2e723b4d1a5902a84b95682245b8fde70b21187f1fe1ad9fa08c4ffaa` |
| M1458 ordered records | `5956085b196979848c3d283744396ea3b0a38a268fb21af0eaecb53e87fc6c9c` |
| M1511 ep34 decoder semantic hammer | `d32017624a41e2ba0ecb41a13316fcc48aaf1bdc70788780c3530a7952a8f508` |
| M1519 C1 promotion failure forensic | `dd99079d32a50e6341717672440e0de7b55581e0f954ea01d6648c771ce5618b` |
| M1479 C3 corrected result hammer | `4e09a4710481db9be7816a097667662361c37963a22467774401810489fc6c6d` |
| M462R2 G8 result | `3c80bb7c037f58a3f6dbabd553a822ebc4b4f887bd076530dc29a52f53c5c09f` |
| M386 G12 result | `84652be51d1b66bbe9e750809b4963ec565ddbc366c0dfaa8f6d083353f7e3b7` |
| M501 overlap review | `09013dcea337a7775c69d7282e4e8882ac5e63abc1c7fd38f9c49ec9d9013eb9` |
| protected docs/359 | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

