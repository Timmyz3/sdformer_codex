# 算法+硬件统一建议清单（三份调研合并，r1 2026-09-17）

作者：主会话合并（CPU-only，未触 GPU/EDA/RTL；未动编号文档与 docs/359/362/366）。
合并来源（三份原始文档为准，本文只做交叉综合）：

| 流 | 目录 | 内容 |
|---|---|---|
| ① co-design 范式 | `reviews/idea_survey_codesign_r1_20260917/` | 只收"算法+硬件同时提出"的工作；五条耦合范式 P1–P5；竞品撞车警示；空白判定 |
| ② Motion 注意力化简 | `reviews/motion_attention_simplification_r1_20260917/` | 现网 h60 逐段拆解；**Q 梯度断流**发现；化简候选 A1/A2/A3 + CPU 证据 |
| ③ 泛顶会 idea | `reviews/idea_survey_2026_r1_20260917/` | 四板块检索；筛选律；8 个候选 M1–M8（含 4.0 双腿对照）；外部锚 |

---

## 0. 一句话结论

三条独立路径（外部范式、本地代码解剖、外部候选）收敛到**同一硬件公共载体**：「**enable 位图作为一等 issue/clock 操作数**」（掩码进描述符字段 → 直接门控 FC/ATLIF/decoder 的 issue 与时钟），
以及**同一条筛选律**：天然中间稀疏已判死，只有"契约层新稀疏 + 阶段级对象切换"能活。
DATE 4.0 门槛（docs/433：新算子合同 + 改存储/执行对象）的候选收敛为 **M1 CGRD / M2 RCM / M3 TAP 三首选 + M4 BSF-PM 并列第四 + H1/H2 待验**。

---

## 1. 筛选律（三份合并后的第一判据，来源：③ §1.5）

本地 `[prof]` 事实：H67 中间 output-site 空置仅 `0.1117%`、跨帧自然 source-work 仅 `2.7%`、ATLIF delta/early-stop 自然收益 `0.0676%`。因此：

> **任何依赖"现有中间特征里天然就有的零/冗余"的跳过，都已被本地实测判死。**
> 能活的只有两类：**(A) 契约层新稀疏**——由训练/合同引入、本来不存在的稀疏；
> **(B) 阶段级对象切换**——跳过"整层/整槽/整 tile/token 行"的**执行对象**，而非逐操作数零。

范式流①的 P1（稀疏可预测性→确定性负载）与 P2（跳过粒度↔存储对象粒度）正是这条律的外部分身：外部强工作（FireFly-T、Workload-Balanced Sparsity、SPARTA、Sparse by Command）全部符合 A/B 两类。

---

## 2. 4.0 门槛候选·统一总表（跨三份，按"外部锚强度 × 精度风险 × 冲突"排序）

| 排名 | ID | 来源流 | 机制一句话 | 新算子合同 | 新存储对象 | 新执行对象 | 外部锚 | 精度风险 | 冲突/依赖 | 验证状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **M1 CGRD** | ③ | 逐 tile 收敛门控精化深度（整数流增量 ≤ε 即停）+ 精确证书 | 收敛深度证书 | per-tile 深度图 + 流增量寄存器 | level-enable issue 门 | **强：CICC 2026 硅级**（0.20×/0.08× 操作量，本机 PDF 一手） | 低-中（ε 事后标定） | 无；**副产品补齐 G1 decoder trace** | 未验证 |
| 2 | **M2 RCM** | ③ | 运动 regime 当 command，小门控网预测 per-tile 执行掩码，掩码进描述符并跨层传播 | regime-掩码合同 | 掩码寄存器组 + 跨层台账 | tile manager 一拍跳过、不发 DMA | **强：MICRO 2026**（延迟 −51~59%，FPGA 实测） | 中（需联合训练） | 无（与 H1/H2 正交，可叠加） | 未验证 |
| 3 | **M3 TAP** | ③ | token 早停 + **原地保持**（IR-Arc），enable 位 = clock/issue enable | hold 语义进算子合同 | per-slot hold 使能位 + hold 寄存器 | block 级 early-stop；FC/ATLIF 共用 enable 位图 | **强：ICLR 2026**（training-free，事件任务已验证） | 中-高（迁移未证） | 须与 RQTB 目录解耦 | 未验证 |
| 4 | **M4 BSF-PM** | ③ | 守卫融进 M935 既有 exact 父匹配 bit-round（PADE 式，无新预测器） | exact 守卫（无新语义） | per-lane 2-bit 区间寄存器 | bit-round guard 阶段 + lane clock-enable | 中：HPCA 2026（负载不同） | **零（exact）** | 无；并入既有 exact 线 | **CPU 最先可验（对削率重放）** |
| 5 | M7 TPS | ③ | 时间槽证据饱和即停（halt 位图按槽下发） | halting 证书 | 证据寄存器 + halt 位图 | 槽推进控制器 | 中：WACV/C-STEP/ASTER | 中-高 | **与 D1/B2 训练线抢身份**；与 LQ4/TLQ-5 须写明对象差异（槽计数 vs 槽记录） | 未验证 |
| 6 | M6 MWM | ③ | 时空冗余 token 合并 + 显式重数（H81 已有精确整数归一化） | 重数合同 | 合并索引图 + 重数字段 | 合并/解合并 + 收缩 issue 宽度 | 中：NeurIPS/CVPR 2026 | 高（有损） | 与 M3 互补、需 DSE | 未验证 |
| 7 | M8 SENS | ③ | `{flow, support, confidence}` 输出契约；低支持 tile 保持上帧流值 | 支持域合同 | support 位图 + 流场 hold | support 门控 tile issue | 中：CICC/SENTRY | 中（hold 语义） | 与"跨帧 feature cache"NO-GO 相邻，须严格切割 | 未验证 |
| 8 | **H1 TLQ-5** | ①(agent A) | T=5 时间商记录文件 + run-length 广播执行器 | 商记录文件合同 | 记录文件（run-length 结构） | run-length 调度单元（长度已知→固定延迟） | 中：P1 范式（FireFly-T/SPARTA 同族） | 低 | 无 | **待 CPU 最小验证** |
| 9 | H2 XWIN-RB | ①(agent A) | 重叠滑窗 + 滚动分母 + 跨窗目录 | 跨窗目录合同 | 目录指针 | 整对象跳过 | 中：P2 范式 | 低 | 无 | **待 CPU 最小验证** |
| 10 | LQ4 | ②(agent B) | 时间四元组 stencil（4 槽统计记录跨 pair 保留） | stencil 合同 | 4 槽统计记录 | — | 中 | 低 | 继承 C1 实测 FAIL 后的位置 | 未验证 |
| 11 | M5 SIP | ③ | 有符号抑制预状态预测"不发放" | inhibit 描述符类 | 抑制寄存器 | 旁路通路 | 中：TVLSI 2026 | 中 | **与 ATLIF early-stop 判死相邻；显式依赖 H12 先修好** | 未验证 |
| — | DRC | ②(agent B) | 双角色构造合同 | 备选 | — | — | — | — | 备选（见 ② 原文档） | 未验证 |

**若只做三个：M1 + M2 + M3**（③ 的推荐，本文认同）；**M4 并列第四**（零精度风险，CPU 验证成本最低）；**H1/H2 作为本地已有协议先验，CPU 验证可与 M1/M4 并行**（互不依赖）。

---

## 3. 算法侧修复/化简（非 4.0 候选，但是多个候选的前提）

来源 ②（本地解剖，CPU 证据已过 1 ulp）：

1. **Q 梯度断流（最高优先级发现）**：h60 的 TX 分数前端全为 bool 掩码（不可导）+ SC 分支被 `mu=0` 乘死 → **注意力对 Q 梯度恒为 0**（实测 `dq ≡ 0`），与 ATLIF 死神经元诊断中 `sn_q` 近死模块互印证。已补进 `neuron_autoresearch/CLAUDE_ATLIF_DEAD_NEURON_DIAGNOSIS_20260917.md` §2.D。
   - 含义：**H12 + 双向 rate 反馈只解 ATLIF 一半**；Q 侧恢复学习还需打开梯度通路。
2. **A1 dedup-SC-off**（删死路径，前反向双精确，风险最低）→ 建议直接进 GPU 队列（t49 后 5ep 对照，期望位级同分）。
3. **A2 shared-ternarize**（事件化去重，前向精确；反向少 STE 单位，需补偿或重训）。
4. **A3 canonical-popcount**（4 统计量规范式 = RTL MSSB5 同构；前向 1 ulp、Q7 整型域精确；**会恢复 `dq`**）——性质上就是"算法侧机制升级"，与 4.0 双腿同构：软件改写后 model↔RTL 可整型域逐项对拍。属 GPU 对照实验，不属纯工程化简。
5. 注意 `docs/44:128` center→quant 与 RTL 顺序分歧未关，写"部署等价"前必须先关。

**依赖链**：M5（及一切依赖"神经元可学"的候选）← H12 梯度修复 ← A3 或 mu>0 提供 Q 梯度。**训练侧修复是这些候选的前置，不是平行项。**

---

## 4. 三份之间的相互印证与冲突

**印证**
- ① P1/P2 ↔ ③ 筛选律 A/B 类 ↔ ①竞品工作（FireFly-T byte 级写、Sparse by Command 跨层掩码传播、Sparsity Tax bitmap=clock-enable）三处独立收敛到同一载体：**enable 位图即一等 issue/clock 操作数**。建议三条 M 候选共用一套"掩码寄存器 + 跳过状态机"，避免被评审读成三个 matcher。
- ① P3（门控→功耗域）与 ③ Sparsity Tax 的"bitmap 直接当 clock-enable"是同一范式的软件/硬件两侧表述。
- ① P4（阈值→数字可控可学习）↔ ② 的 Q 梯度/ATLIF 修复 ↔ ③ M5 依赖：**训练侧阈值与门控的可学性是共同瓶颈**。

**冲突/边界**
- M7 与 D1/B2（T>2 合同）训练线抢同一身份，须排 D1 之后；M5 与已判死的 ATLIF early-stop（0.0676%）相邻，CPU 上界不过门即降附录。
- ③ 全文外部文献为 `[lit]` 摘要级（WebFetch 被环境拒绝），**任何 `[lit]` 数字不得直接进论文主表**；仅 CICC 2026 PDF 为 `[lit-一手]`。
- C1（Local5 跨 pair 统计平面）已实测 FAIL（G4 23.42%<60%），LQ4 为其诚实继承者，不复活 C1 口径。

**竞品警示更新（合并后的必引清单）**
- **CICC 2026 28nm 光流加速器（北大，U-Net 混合 SNN-ANN + DLS 层跳过 + 变宽压缩）**：与 ASNA-Flow 并列，**同一应用域（事件/光流加速）**的第二个硅级先例；差异：我们 = spiking Transformer + 时间商/跨窗结构。必须引用并对比（其 DLS 同时是 M1 的外部锚）。
- 原 ① 清单维持：ASNA-Flow（最接近竞品）、28nm Spiking ViT、FireFly-T、SPARTA、SDformerFlow-TETCI。
- 空白复核：**仍未见"事件光流专用 spiking-Transformer ASIC"**（③ 本轮检索未推翻）。

---

## 5. 推荐行动次序（严格 CPU 先行；GPU 队列待 t49 结束后由用户批准）

**CPU（即刻，零风险、互不依赖，可并行）**
1. **M4-S1**：冻结 51.84M 行账本的 bit-round 守卫对削率重放（成本最低 + 零精度风险 + exact 文化契合）。
2. **M1-S1**：decoder/深层 cycle 占比与 ε 上界（读冻结 profile；**副产品直接补 DATE G1 decoder trace 缺口**）。
3. **H1/H2 CPU 验证**：run-length 分布/p95 调度单元、跨窗命中率与目录面积模型（agent A 已列）。
4. **A1**：deploy 旋钮逐位比对 + 端到端 0-diff（② 已给脚本与步骤）。
5. **M2-S1 / M3-S1 / M5-S1**：per-tile 工作分布与掩码上界、IRToP enable-off 代理界、抑制旁路率 vs 0.0676% 门。

**GPU 队列（一次一个，不抢卡）**
H12 G1（今日已建目录）→ B2 重跑 → ② A1/A2/A3 探针（5ep）→ M1 免训练 ε 标定 → M3 training-free 探针 → M2 门控 MLP short。统一判据：valid825 相对锚点 `≤×1.01` 且周期上界 ≥10%（或组件能量 ≥15%）。

**EDA**：本轮不新增；M4 过 CPU 门后只允许并入既有 exact 匹配线（不新开模块线）。

---

## 6. 诚实边界

- 三份来源文档的红线声明各自有效；本文未新增任何实验。
- ③ 的外部文献 `[lit]` 为检索摘要级；① 同（WebSearch 为主）。数字进论文前须二手一手核对。
- ② 的 CPU 微基准受同机 t49 dataloader 噪声影响（单段 ±2×，比值稳定）。
- 全部 M/H/LQ 候选均为**提案**：未跑 GPU、未写 RTL、未训练；收益方向为预测而非结果。
