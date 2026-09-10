# Codex 接手：调研包（本 Agent 不实现）

日期：2026-09-07。作者：Grok 调研会话。  
**本文件是调研交付，不是施工许可。** Codex 负责按合同做测量/overlay/RTL；不要把这里的倍率写成论文加速比。

Grill 合同（已锁定，含用户三处修改）：

- 一篇 TCAS-II，一个协同对象的**两面**（算法面+电路面），篇幅偏硬件。
- 可改算法、可重训；必须仍是 **SNN Transformer 事件光流** + **脉冲驱动 QK**（无 softmax）。
- valid825 AEE **≤ 1.259** 且优于 SDformerFlow PSN **1.5848**。
- 不要求 1RW；旧机制可重测，**不用 kill list 直接扔掉**。
- 先全部筛、再套效果，不预先只留一个赢家。
- 主工艺 TSMC 28HPC+；FPGA ZCU102 加分、先不绑对象。
- 一张 A800 可全量；换注意力/神经元从 ep34 微调，换骨干/窗/T 可从 SDformerFlow 起。

本 Agent 已按两份 idea-skill **原文走完流程**（见 `orchestra/`、`kdense/`）。  
**没做：** overlay 微调、valid825 新跑、新 RTL、EDA。那些是 Codex。

---

## 1. 两套 skill 收口状态

| 工序 | 状态 |
|---|---|
| grill-me 决策树 | 完成，见本文件合同 |
| Orchestra Phase 1 发散 | 完成，`orchestra/PHASE1_DIVERGE.md`（F1–F9 + R01–R20） |
| Orchestra F8 干系人 | 完成，`orchestra/F8_STAKEHOLDERS.md` |
| Orchestra Phase 2 收敛 | 完成，短名单 5：I001(+I002)、I003、I004、I005、I007 |
| Orchestra Phase 3 打磨 | 完成，流程赢家 I001；**不是**投稿锁定；P01 未签 |
| K-Dense 10 步台账 | 完成，`kdense/WORKFLOW.md` + `session.json` |
| K-Dense 独立预测/不确定性 | 完成（不再用占位句） |
| K-Dense 对抗全文模板 | 完成，`kdense/ADVERSARIAL_REVIEWS.md` |
| K-Dense 文献后再开 | 完成，I009–I013；`kdense/LITERATURE_REOPEN.md` |
| K-Dense CLI | `validate_register.py` valid、0 errors/0 warnings；`evaluate_matrix.py` `decision: null` |
| K-Dense D001 | **simulation** = M0–M2；电路面若 T0<~2% 改 I004；待 P01 签字 |
| 32 卡 + S1 记分板 | 仍在 `README.md` / `SCOREBOARD.md`（未删除） |
| 选唯一投稿标题 | **不做**（grill 合同）。Orchestra 只打磨一条*测量*计划 |
| 实现 | **明确不做** |

文献窗：新候选 2024-01 至今；经典工作只当对照。检索日 2026-09-07，有界，状态多为 `search-incomplete`，**不是**「从未有人做过」。

---

## 2. 给 Codex 的工作方式

1. 读本文件 + `SCOREBOARD.md`，不要 `codex exec resume` 旧巨会话。
2. 每条候选先做 **可复核测量**，再谈 RTL。
3. 组件倍率不相乘；无 VCS+DC 同负载不写 RTL 加速比。
4. ATLIF 是 `z=θ×g`，θ 连续。
5. 不改 `docs/359`。
6. 生产 RTL 默认不动；新东西放 `ideafromai/research/` 或隔离树，用户点头再合。

建议 Codex **先测量、后实现** 的顺序（仍不是标题预选）：

| 序 | ID | 为什么先做 | Codex 具体动作 |
|---|---|---|---|
| 1 | B1/B2/B3 | 100 样本 QK 已有方向，但身份是 ep35 | 用 **ep34** 打同一套 token 级 K 零 / dirty / leaf-needed |
| 2 | A4 | ep34 CPU B8 已 3.89× 序列化 | 同资源 RTL 门，不要把 CPU premodel 当 VCS |
| 3 | A6/D1 | 加法减量 20.8%/42.2% 相对 FTP | 接持久 Y，**同资源 vs 直接 FTP** |
| 4 | E2/B4 | 最便宜的网络面 | stage2 **一块** 线性 QK 或 QKFormer vs Motion-XOR，10 帧 AEE |
| 5 | A3 | 旧 PAFT 身份 AEE≈1.47 过不了 1.259 | 只有在 ep34 上重训才有资格 |

---

## 3. 已有硬证据（可引用路径，勿重发明）

| 证据 | 路径 | 身份 |
|---|---|---|
| QK 100×12 普查 | 本目录 `scoreboard.json` → `attention_census` | **ep35 路径，非 ep34** |
| 原始 packed Q/K | `hw_autoresearch_nts07/system_handoff/received/.../trace_qk_100sample_12block/` | ep35 |
| TSBG B2/4/8 | `results/tsbg_ep34_same_io_b2_b4_b8_quickkill_r1_20260902/result.json` | ep34 `4bbaf7fc` |
| S2 零成本上限 | `results/m1713_ep34_s2_fc_patch_zero_cost_upper_bound_fastkill_r1_20260901/` | ep34 |
| PAFT valid825 | `results/m247_...` 与 τ=1 的 1.469→1.498 | **PAFT ep4，非 ep34** |
| Prosperity 完整层 | `ideafromai/research/complete_transfer_20260907/c1_full_layer_r1.json` | sample0 一算子 |
| C2 时间共享 RTL | `c2_temporal_shared_protocol_20260907/records/functional_r2/` | 功能过 |
| 晚知 θ 包 RTL | `hardware_mechanisms_20260906/records/functional_r3/` | 功能过 |
| 源顺序加法 | `fusion_delivery_20260907/source_order_r1.json` | 诊断有理数 |

QK 普查摘要（3,105,000 token）：K 双 T 全零 56.0%；dirty 60.9%；仍需打分叶 43.9%；S1 几乎可空，**S3 最没空**（leaf 75.6%）。单样本 96% skip 作废。

---

## 4. 文献后再开的三条（独立轮之后补的）

| ID | 两句 | 先验 | 文献状态 |
|---|---|---|---|
| B11 / I009 | FireFly-T 用 SRAM 字节写做 3D 注意力布局。我们窗是 T×15×15，可以试字节/位平面写，不必 1RW。 | FireFly-T IEEE TC 2026，KV260 | `support-located` 布局；光流窗未做 |
| B12 / I010 | SpiLiFormer 侧抑制注意力（ICCV 2025，arXiv:2503.15986）。可当 Motion-XOR 的「抑制共静默」训法，不是新 ALU。 | SpiLiFormer ICCV 2025 | `search-incomplete` |
| C5 / I011 | 去 LayerNorm / 硬件可折 running-BN。已有 running vs no_running 精度分裂。 | 本地 PAFT BN 实验；NF-SpikingVTG | 本地 `mixed`（1.47 身份） |
| I012 | 三项 popcount **分别**在 support=0 时关门（overlap 均值 0.013）。挂在 I001 上，不是第二条标题。 | `T1_T4_ep35.json` | `no-direct-evidence-located` on ep34 |
| I013 | 编码器特征图时间跳过（CICC DLSS 类比），**显式不是** Motion-XOR 分数跳过。 | Zhang CICC 2026 | `support-located`；不当本短文对象 |

模拟 CIM（ASTER 2511.06770）不当 PPA 主线。SpikePool 用 pooling 换 SSA，违反脉冲 QK，只作对照。

---

## 5. 逐卡调研卡（pitch / 先验 / 反对 / 文献 / Codex）

先验不是 kill list：抄全、对照、重测都可以。

### 簇 A 重测

**A1 Prosperity 抄全**  
Pitch：H67 的瓶颈卷积仍有重复乘积；把官方 Prosperity 链按 θW 抄全，看还剩什么可换。  
先验：Prosperity HPCA’25。本地官方外层 product/bit **2.33×**，mem stall≈0。  
反对：审稿会说这就是 Prosperity；双口也救不了 stall≈0 的模型。  
文献：`support-located`。Codex：不要重写官方 simulator；要新数字必须同资源对照 `run_fc`。

**A2 父值提升**  
Pitch：叶子组把父值在寄存器里抬一层。  
先验：SumMerge / Prosperity 父林。增量 **+1.58%** 加法。  
反对：增量小于扫描代价。Codex：除非换布局，否则不必重做。

**A3 PAFT**  
Pitch：Hamming 正则换更可捕获的 pattern。  
先验：Phi。m247 running 相对对照 +0.57% AEE，但是 **PAFT-ep4 AEE≈1.47>1.259**。  
反对：旧身份直接桌退新精度门。Codex：**必须 ep34 重训** 才继续。

**A4 TSBG**  
Pitch：同行多消费者只送一次权重行。  
先验：Gustavson / Eyeriss 广播。ep34 CPU：B8 序列化 **3.89×**、权重字节 −80%。  
反对：创新弱；CPU≠RTL。Codex：同资源 RTL/VCS，禁止把 3.89× 写进摘要。

**A5 S2**  
Pitch：按块跳过剩余工作。  
先验：token/block skip 一族。FC2 零成本上限 <1.15。  
反对：上限不够。Codex：FC2 停；FC1/patch 没有 payload+AEE 就不要 RTL。

**A6 时间共享部分和**  
Pitch：同时间发放的源先加再多播，保留 θ。  
先验：共享归约 / LoAS / RSR。8 槽相对 FTP −20.8%/−42.2% 加法；RTL 功能过。  
反对：加法≠周期；无 Y 端口。Codex：接 Y，打 FTP 同资源。

**A7 每 bank 一模式**  
Pitch：每 bank 冻一个 10-bit 模式，免字典。  
反对：bank 映射不是捕获地址。Codex：先统一映射再谈周期。

**A8 晚知 θ 包**  
Pitch：BN 分界未到时只留能证明正确的残差和 θ。  
先验：Gist；区间消费。五配置功能过。  
反对：合成区间不是网上 BN。Codex：接真实 BN/PSN。

**A9 运动默认态**  
Pitch：用运动预测当实值默认，只传例外。  
先验：Graham nonzero ground state；Motion-aware Event Suppression 2026。  
反对：无 AEE。Codex：先 10 帧训，AEE 不动再硬件。

### 簇 B 注意力

**B1 Motion-XOR 三 popcount**  
Pitch：分数是 overlap+共静默+时间对端 XOR，没有乘法。  
先验：FireFly-T 只做 QK AND-PopCount；α-XNOR 无 K_peer。普查 overlap 均值 0.013、motion 1.53。  
反对：注意力旧信封 ~0.6%；系统加速可能为 0。Codex：ep34 普查 + 叶周期，不写整网 FPS。

**B2 dirty-lane**  
Pitch：Q/K 相对 t0 没变则复用分数。  
先验：DeltaCNN/DLSS。100 样本 token dirty 60.9%，leaf 仍需 43.9%。  
反对：Shiftmax 分母可能是行级，token skip 会塌。Codex：必须行级分母实验。

**B3 K=0 跳 V**  
Pitch：K 当 V，K 全零则不打分不取 V。  
先验：不是 empty-tile。K 双 T 零 56%。  
反对：S3 只有 24% K 零。Codex：和 B1/B2 同一普查。

**B4 线性 QK / SDSA 换核**  
Pitch：公开 SDformerFlow 是 SDSA/QK，H67 的 Motion-XOR 可换成线性脉冲 QK 看 AEE 和电路是否更简单。  
先验：QKFormer NeurIPS’24（线性 Q-K，无 V/softmax）；SDformerFlow SDSA。  
反对：AEE 可能回到公开 1.58 附近。Codex：stage2 **一块**，10 帧，从 ep34 微调。

**B5 STSA**  
Pitch：时空联合脉冲注意力。  
先验：Spiking ST-former 2025；STAtten。反对：T_w=2 可能太短。

**B6 SLI+ACF**  
Pitch：邻域 depthwise 补 SSA 在稀疏时丢掉的局部运动。  
先验：arXiv:2608.19238。反对：多一条通路，五页难讲两个核。

**B7 SQKFormer MABN**  
Pitch：膜电位自适应 BN + 通道增强 QK。  
反对：和现网 PSN/ATLIF BN 合同冲突。Codex：先对齐 BN，再训。

**B8 LRF-SSA**  
Pitch：局部感受野神经动力学注意力，挂现成 SSA。  
先验：ICLR 2026 2603.19290。Codex：插件，ep34 微调 10 帧。

**B9 FireFly-T 引擎**  
Pitch：把 FPGA 二值注意力引擎改成三项 popcount。  
先验：FireFly-T TC 2026。反对：他们做 QK^T，我们做 Motion-XOR。

**B10 双 spike 跳零**  
Pitch：Q 与 K 都是 spike，跳双零。  
先验：2501.07825 Spike-driven Transformer 加速器。Q 密度 1.7%，几乎是跳 Q。

### 簇 C 神经元/前端

**C1 混合 T ATLIF**  
Pitch：膜留在岛内，对外 1-bit；T=10 与 T=2 两套调度。  
反对：「first mixed-T」不成立（Chen/Chang 已 mux T=4/2/1）。Codex：当 C3 升级，不写 first。

**C2 动态 T**  
先验：STISA TSC。反对：光流边界对短 T 敏感。

**C3 Spiking Patches**  
先验：2510.26614。反对：改输入身份，必须从 SDformerFlow 重训。

**C4 θ 折进 W**  
冻结就是 z=θg。改合同 = 新 AEE。不要悄悄当 int8 载荷。

### 簇 D 大份额算子

**D1 LoAS FTP**  
先验：LoAS MICRO’24，arxiv:2407.14073，官方是 PyTorch/profile 不是完整 ASIC RTL。  
反对：FTP 是对照不是自动标题。Codex：混合 T=2/10 纤维是否比直接 FTP 同资源更好。

**D2 RSR++**  
当对照。显式整行字典已停（stage0 −2.9% 状态）。

**D3 解码器岛**  
份额大。只有完整 decoder 表才能进系统行。有 ep34 D0 shard，未闭合。

**D4 双端稀疏**  
FireFly-S。本地 N:M：FP32 无精确零块，必须重训。

**D5 一织物多算子**  
架构选项，等 A1/D1/D3 有共同 ISA。

### 簇 E 网络替换

**E1** SDSA vs Motion-XOR 消融（算法面最干净）。  
**E2** 只换 stage2 为 QKFormer（最便宜）。QKFormer：NeurIPS’24 2403.16552，线性 Q-K。  
**E3** SDT-v3 块。  
**E4** 改窗/T，从 SDformerFlow 重训。

---

## 6. 对抗总表（skill 要求的「不是发明者的反对」）

| 风险 | 伤哪些卡 |
|---|---|
| 注意力 0.6% 信封 → 岛再快也没有系统摘要 | B1–B10、B11 |
| 「抄全先验」被当成标题 | A1、A4、D1 |
| 旧身份 AEE 混进 ep34 门 | A3 |
| token skip ≠ 行级 Shiftmax | B2 |
| CPU/加法减量写成 RTL 加速 | A4、A6、D1 |
| 模拟 CIM / FPGA 当 28 nm PPA | ASTER、ZCU102 |
| 五页里两个核 | B6+B1，或 C1+A1 |

---

## 7. 本 Agent 停在这里

调研交付（更新于 T1 窗粒度之后）：

1. `README.md` — 候选台账与合同  
2. `SCOREBOARD.md` / `scoreboard.json` — S1 数字  
3. `T1_T4_ep35.json` — token vs **15×15 窗** dirty（窗干净仅 9.9%）  
4. `CODESIGN_OBJECTS.md` — 8 个可投稿拼法（O1–O8），含 T0 依赖  
5. `MEASUREMENT_CONTRACTS.md` — M0–M8，Codex 从这里开工  
6. `SOURCE_LEDGER.md` — 有界文献  
7. **本文件** — 总交接  

把本目录丢给 Codex。**第一件事是 M0+M1（ep34 份额与 QK），不是写 RTL。**
