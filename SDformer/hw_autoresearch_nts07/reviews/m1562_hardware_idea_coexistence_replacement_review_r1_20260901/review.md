# M1562｜现有硬件机制共存、覆盖与替换第一性原理审阅

日期：2026-09-01（Asia/Shanghai）  
性质：只读证据审阅；没有运行 GPU、VCS、EDA、远端任务或性能模拟器  
对象：C1 finite-capacity 1RW product-capture、C2 typed-signed K8、C3 Fixed-T10 ATLIF、RQTB、TSBG、S2 CCBS、S1 ABCG  
裁决：**保留 C1/C2/C3 三贡献骨架；TSBG 只能成为 C2 的 exact memory specialization；S2 可作为 C1/C2 之前的 optional lossy fetch gate，但必须对 C1-enabled 强基线报告增量收益；S1 与 S2 默认只能选一个正文有损模式。RQTB 与 C3 数学正交，但不能声称物理完全正交。**

本审阅只新增本目录，不修改任何旧 source/result/contract、`ucli.key`、论文或 `docs/359_DATE终局冻结_20260813.md`。审阅时 `docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 1. 结论先行

### 1.1 四个直接答案

1. **TSBG 是 C2 memory specialization，不是第四个执行器。**它改变的是 typed-signed source 对 weight row 的组织、驻留和广播方式；bundle 展开后仍由 C2 K8 issue、bank service、Acc24 context 和 completion 协议执行。它替换普通逐 token row-fetch/descriptor-scan 模式，不替换 C2 算术和 Acc24。只有当 bundle builder、row buffer、bank conflict、多个 destination context、tail/commit 全收费后仍过门，才升级为 C2 子机制。
2. **S2 可以在 fetch 前与 C1 串联，但不能把它和 C1 的 exact product-capture 收益相乘。**正确顺序是 `activity/group bound -> S2 debt decision -> surviving block fetch -> C1 lookup/capture or residual product -> psum update/commit`。S2 的分母必须是 C1 已开启的 exact 路径，只计 C1 仍会付出的 residual fetch/product/update。更关键的是，当前 S2 的 `O16` 与 C1 的 96-output/1152-bit parent row 不天然对齐；若 weight/parent storage 不能按 O16 独立关读，S2 只能省局部 compute/update，不能声称省整行 fetch。
3. **S1 与 S2 默认是二选一的正文有损模式。**两者都在 fetch-before-compute 阶段消费同一个 destination/output-tile error budget，并维护 debt。串联两个独立 epsilon 会重复扫描、metadata、端口和误差预算。理论上可用一个共享 debt ledger 做“先 S2 block、后 S1 source”的层级门控，但必须给联合界和同资源收益；当前没有这份证据，因此论文和 RTL 只选一条。现有优先级是 S2；S1 保持 capture piggyback/fallback ablation。
4. **RQTB 与 C3 在数学和状态语义上正交，不是物理完全正交。**RQTB 合并 attention score 等值类，拥有 K/class/slot/window-retirement 状态；C3 执行 Fixed-T10 ATLIF，拥有 temporal coefficients、phase/neuron state。两者不覆盖同一计算，也不替换对方，但可能共享 SRAM 端口、phase controller、clock/power domain，且 attention-local ATLIF 在图上可紧邻 RQTB。因此只能写“functionally orthogonal and phase-composable”，不能写“zero-cost physical coexistence”。

### 1.2 现在保留和替换什么

| 机制 | 最终身份 | 是否保留 | 替换/覆盖关系 |
|---|---|---|---|
| C1 1RW product-capture | exact Conv backend | **保留，C1 主体** | 不被 S2 替换；S2 只可在其前面做 optional gate |
| C2 typed-signed K8 | exact shared source execution fabric | **保留，C2 主体** | 不被 TSBG/S1/S2 替换 |
| C3 Fixed-T10 ATLIF | exact temporal/neuron island | **保留，C3 主体** | 不复活 rank-3/PAFT headline |
| RQTB | exact attention support | **保留但降为 supporting operator** | 不与 C3 合并成同一数学机制；不占第四贡献 |
| TSBG | exact C2 row-memory/broadcast mode | **优先完整 fast-kill** | 通过后替换 C2 普通逐 token row-fetch 模式；不替换 K8/Acc24 |
| S2 CCBS | optional lossy C1/C2 prefetch gate | **条件保留** | 若准入，替代 G11 类逐 source debt，并成为唯一正文有损模式 |
| S1 ABCG | optional lossy analog-boundary gate | **降为 piggyback/fallback** | S2 准入后不再并列；S2 失败且 S1 独立过门时才替补 |

## 2. 一条可执行数据路径

建议的统一相位/数据路径不是七个并列岛，而是三条 execution phase 加可选前端：

```text
activation / event / temporal state
    |
    +-- [optional lossy policy: S2 OR S1, epsilon=0 is exact]
    |        decision must finish before independently suppressible weight fetch
    |
    +-- Conv path: C1 finite-capacity 1RW product-capture -> residual product -> psum/commit
    |
    +-- FC path: TSBG exact row-memory mode -> C2 typed-signed K8 -> shared Acc24/commit
    |
    +-- temporal/attention phases:
             RQTB score-class service -> attention-local or global C3 Fixed-T10 ATLIF
```

这张图有三条纪律：

- phase-composed 只表示按图顺序复用资源，不表示局部倍率相乘；
- exact backend 始终存在，S1/S2 的 `epsilon=0` 必须回到同一 exact transaction stream；
- final commit、terminal/close、barrier 与 checkpoint/quant identity 不得因 skip/bundle 被隐式删除。

## 3. 单机制资源与身份

| 机制 | 决策/执行阶段 | 核心状态与 metadata | 主要共享资源 | checkpoint / quant 身份 |
|---|---|---|---|---|
| C1 | surviving Conv source 后，parent/product lookup 与 residual product issue | 1RW parent scratch、row directory/order tag、2-bit liveness、source mask、response FIFO、psum valid | 240-KiB-class storage、weight/PWP port、psum/commit | 不改 checkpoint；最终 checkpoint 改变 activity/product trace 时必须重放。exact 数值仍须绑定正式 quant/Acc 语义 |
| C2 | typed source decode 后，最多 K8 unique-bank issue 与 Acc24 update | source tuple、bank/tag、six-slice Acc24 context、terminal/completion | weight banks、descriptor FIFO、Acc24/writeback | 不改 checkpoint；workload/codeword/bank trace需对最终 checkpoint 重绑 |
| C3 | T10 temporal phase / neuron update | 10x10 layer coefficients、phase、bias/threshold、neuron state | coefficient/state SRAM、mult/add pool、phase controller | Fixed exact 模式不改 checkpoint；coefficient、bias、threshold、scale 必须对最终 checkpoint/quant 重绑 |
| RQTB | attention score classify/quotient/Shiftmax/value recovery | K-store、score class、multiplicity、slot/window retirement | attention SRAM、histogram/update port、phase controller | exact quotient 不改 checkpoint；等值类和局部倍率必须对最终 Q/K trace 重绑 |
| TSBG | FC source/weight row fetch 与 C2 ingress 之间 | bundle header、support/sign/nonunit/codeword、destination context、row-buffer tag | C2 descriptor SRAM/queue、weight row port、K8 ingress | 不改 checkpoint；M1558 当前只对 captured diagnostic codeword/contributor exact，`hardware_quantization_authority=false`，不能称 model-bit-exact |
| S2 | group activity已知后、weight block fetch前 | `M(G,O)`、`A(G)`、destination-owned epsilon/debt、block/tag/bank key | metadata port、weight bank enable、C1/C2 ingress、debt SRAM | 不要求重训，但形成独立 epsilon/policy identity；必须 paired AEE。`epsilon=0` exact；bound 必须使用正式量化/权重身份 |
| S1 | raw/analog source读取后、source descriptor/weight fetch前 | `beta(j,O)`、magnitude、destination-owned epsilon/debt、source/tile tag | ingress comparator、beta port、C2 descriptor builder、weight gate | 不要求重训，但形成独立 epsilon/policy identity；必须 paired AEE。硬件定点 beta/x 语义未闭前不能宣称 runtime-certified |

## 4. 逐对共存/替换审阅

完整 21 对也在 `pairwise_matrix.csv`。这里保留会影响架构决策的解释。

### 4.1 C1 与其余机制

- **C1 × C2：正交算子、分相复用。**C1 是 bottleneck Conv 的 parent/product capture；C2 是 FC typed-source issue。两者可共享上层 phase/tag 和 SRAM fabric，但不能假设同时占用同一 Acc/port。组合周期只能顺序求和。
- **C1 × C3：图阶段串联、状态独立。**C3 产生或消费 temporal activation，C1 处理 Conv product；共享 coefficient/activation/psum 存储时必须仲裁，不存在算法覆盖。
- **C1 × RQTB：图 scope 分离。**attention quotient 与 Conv parent capture 不重算同一个 product；共存只受 SRAM/phase 物理资源影响。
- **C1 × TSBG：当前 scope 分离。**TSBG 只面向 FC1/FC2 跨 token row reuse；不能扩成 Conv bundle 后再声称与 C1 都是独立贡献。
- **C1 × S2：可串联但有 overlap。**S2 先拒绝 block，C1只处理保留 block。准入必须对 C1-enabled exact baseline；不得把 S2 skip 的 product 再计作 C1 capture 收益。`G16xO16` 到 C1 `O96` 要求六个 O16 bank-enable 或等价独立物理读；整条 1152-bit row仍被读时，weight-fetch saving 为零。
- **C1 × S1：当前冻结 scope 基本分离。**S1 只允许 raw event head 与 patch analog residual，不应扩到已证 binary/layer-code bottleneck 复活 G7。若未来同一 source进入C1，规则与S2相同：先 gate、后 C1，按增量计费。

### 4.2 C2 与其余机制

- **C2 × C3：明确 producer/consumer 串联。**C3 的定点输出可编码成 typed-signed source，C2负责 K8/Acc24；scale、saturation、terminal、token/phase tag 必须成为接口合同。C3 的 neuron state 和 C2 的 Acc24不能混作同一 state。
- **C2 × RQTB：逻辑正交、物理条件共存。**当前 RQTB 自有 slot/K-store；若未来用 C2跑 Q/K projection，那只是共享 Linear backend，不代表 RQTB core 被 C2覆盖。
- **C2 × TSBG：替换关系发生在 memory mode，不在执行器。**TSBG row-fetch/broadcast 取代普通逐 token row-fetch/scan，展开后的 tuple 必须逐项等于 C2原输入。bundle value不同只能复用 weight row，不能复用 product。若 expander/row-buffer 成为 max()，TSBG应降为 traffic/energy support。
- **C2 × S2：串联且有粒度 adapter。**S2 `G16` block 对应两个 K8 issue wave；`O16` 对 C2 96-output服务是六个 slice。drop 必须同时取消两波和相应 destination slice update；keep 的请求仍走原 C2。若 metadata读取占据weight/descriptor关键端口，必须计 stall。
- **C2 × S1：串联。**S1在 typed tuple生成/发射前丢 source，保留 source再由K8调度。source减少不保证cycle减少：K8 occupancy/tail可能恶化，因此必须用 executable recurrence，不用 source-drop 比率代替加速。

### 4.3 C3、RQTB 与新候选

- **C3 × RQTB：功能正交、可相邻串联。**RQTB改变 attention score service，C3执行 ATLIF temporal transform；attention-local ATLIF仍归C3类。独立 state/lifetime 阻止二者互相覆盖，但共享宏/端口/phase control 仍需收费。
- **C3 × TSBG：producer/consumer 关系。**尤其 sn2/ATLIF 到 FC2 时，C3输出可进入TSBG bundle，再展开给C2。TSBG不能改变 temporal coefficient或neuron state，只能复用weight row。
- **C3 × S2/S1：条件共存。**若门控位于C3之后的FC2 fetch，C3保持exact；若门控FC1输入并改变C3上游，local `Linf` bound不是跨T10 state或AEE bound，必须以paired forward验收。不能把local debt直接写成光流AEE上界。
- **RQTB × TSBG：scope分离。**一个是attention score class，一个是FC row reuse；只共享memory/transport时才冲突。
- **RQTB × S2/S1：默认用 target manifest 隔离。**正文的S1/S2不进入attention core，避免把极小Amdahl的attention再次有损剪枝。未来若门控Q/K projection，必须重新证明其误差与RQTB quotient/Shiftmax组合界，当前不准入。

### 4.4 新候选互相之间

- **TSBG × S2：可串联。**TSBG是exact source/row组织，S2是optional lossy fetch gate。理想实现让bundle直接给出`A(G)`/popcount，在不完全展开的情况下做S2决策；S2保留块才触发row fetch/broadcast。S2的性能分母应是TSBG-enabled exact baseline，bytes分类要区分TSBG省source/row重复读与S2省整块weight读。
- **TSBG × S1：能串联但可能互相吃收益。**S1逐source beta scan 若要求先完全展开bundle，会消掉TSBG的row reuse/transport收益。若保留S1，只能集成在bundle expander并共享一次code scan；否则优先TSBG，S1降级。
- **S2 × S1：默认互斥。**二者同stage、同destination debt owner、同weight-fetch目标。只有单一共享debt、联合误差界、统一metadata端口和incremental same-resource收益全部成立，才允许层级组合；当前不得双headline、双epsilon或倍率相乘。

## 5. S2 与 C1 串联的强制物理门

S2 最容易被误写成“在 C1 前再省一遍”。要晋级，至少回答以下五项：

1. **Granularity gate**：`G16xO16` 的 drop 是否能关闭一个真实 SRAM bank/burst？若 C1/weight store仍读取O96或1152-bit整行，weight bytes不得记为下降。
2. **Residual-work denominator**：baseline必须开启C1 exact product-capture、dead-write和zero-source；S2只计这些机制后剩余的fetch/product/update。
3. **Parent-state semantics**：skip不能伪造parent hit、不能破坏liveness/close；terminal与destination commit独立前进。
4. **Port charge**：`M(G,O)`/debt read与C1 directory/parent/weight端口冲突必须进入周期。
5. **No multiplication**：最终组合只重放统一transition machine；不得把C1 `1.759x` 与S2局部比值相乘。

因此当前裁决是 **architecturally composable, performance not composable by arithmetic**。S2仍未准入；M1555只条件授权修复reference后的紧凑capture。

## 6. S1/S2有损模式选择

| 轴 | S1 ABCG | S2 CCBS | 选择含义 |
|---|---|---|---|
| 粒度 | individual analog source | `GxO` block | 同一硬件默认只启用一套debt policy |
| 当前scope | raw event head + patch analog residual | patch/FC/C1候选block | S2覆盖潜力高；S1 Amdahl低 |
| metadata | per-source/tile beta | one max-weight per block + debt | S2需保持 `>=8x` 小于G11且 `<=2%` weight bytes |
| 物理优势 | 精细但scan/port贵 | 一次决定整块fetch，前提是bank粒度对齐 | S2优先，S1 fallback |
| 风险 | threshold/pruning prior强；可能只drop source不省cycle | bound过粗；可能读整行后才发现“skip” | 谁先过paired AEE + same-resource门谁入主表 |

推荐状态机只有 `LOSS_MODE={EXACT,S2,S1}`，而不是同时开启S1+S2。`EXACT`是零预算；DATE主表最多报告exact与一个selected lossy mode，另一个只作negative/fallback ablation。

## 7. 最终论文三贡献归属

### C1｜Capacity-constrained exact product capture

主体：有限容量、single-ported 1RW parent/product capture、dead-write/liveness/completion。  
可挂载：S2若最终准入，作为shared fetch interface上的optional certified prefilter；它不是C1的第二个贡献。  
现有数字边界：C1 CPU same-ledger四层Conv点约`1.759x`，不是RTL/system speedup；宏/时序/功耗边界按当前独立证据如实列出。

### C2｜Typed-signed shared-accumulator source service

主体：typed signed K8、unique-bank issue、shared Acc24、terminal/completion。  
可挂载：TSBG exact row-memory/broadcast specialization；S1或S2中最终胜出的一个optional lossy frontend。  
现有数字边界：K8对等带宽K1x8是`1.016728x` directed component cycles、`4.541078x` throughput/mm2、logic area `-77.6104%`；三者必须同句，不能恢复K8对单K1的旧headline。

### C3｜Exact Fixed-T10 temporal service and operator composition

主体：Fixed-T10 ATLIF的phase/state/coefficient exact service，以及与C1/C2的thin phase/tag composition。  
Supporting：RQTB作为attention-side exact operator/ablation，保持自身slot/K-store state，不单列第四贡献。  
现有数字边界：C3已有28-nm 3ns logic-only、hold-closed component证据，但没有throughput/speedup/power；RQTB只报local attention core约`1.183x`和full-envelope约`1.000911x`的口径差。

**不能写成六个贡献。**TSBG、S2/S1、RQTB分别是C2 memory mode、shared optional lossy policy、attention support。论文贡献标题只保留C1/C2/C3。

## 8. 下一步执行裁决

1. **先闭合TSBG exact fast-kill。**它不需要AEE且是C2内部替换，风险最低；但当前M1558只是source+synthetic，仍需独立source hammer、生产capture和同容量ordinary row-buffer baseline。
2. **随后只重筛S2 16x16。**先修activity-relative safe reference，再检查O16可独立关读；若不能关真实weight bank/burst，立刻降级compute-only/negative，不进入AEE或RTL。
3. **S1只piggyback。**若S2失败，且S1先过metadata/port/weight-byte门，再作为唯一有损fallback进入paired AEE。
4. **不改C1/C2/C3/RQTB主体。**新机制只做submode和ablation，不再开发第四执行岛。

## 9. 声明边界

- 本审阅不产生cycle、traffic、energy、AEE、RTL或PPA新结果；
- logical/functional coexistence不等于macro-inclusive physical coexistence；
- 所有local ratio不得相乘；
- TSBG当前exact只限captured diagnostic codeword/contributor；
- S1/S2均未准入，`epsilon=0`以外必须独立policy identity与paired AEE；
- final checkpoint不会因这些runtime机制被修改，但C1 trace、C2 workload/codeword、C3 coefficient、RQTB score classes和S1/S2 metadata都必须对最终checkpoint/quant authority重新绑定。

