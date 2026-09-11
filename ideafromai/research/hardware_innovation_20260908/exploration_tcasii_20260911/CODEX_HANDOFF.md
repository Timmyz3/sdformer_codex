# Codex 交接 — Grok 调研（2026-09-11）

给 Codex：只读本文件 + 下面「必读」路径。不要改 `tcasii/main.tex`、不要改生产 nts07，除非用户另说。Grok **不接管 RTL**。

根目录：

```
/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/
```

下文相对这个根。硬件测量树：

```
…/algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/
```

---

## 0. 身份（冻结，写进任何 RTL/注释/论文句）

必读：`IDENTITY_ATLIF.md`

- 官方 AT-LIF：\(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\)。推理 \(W\leftarrow\theta W\)。层间是 **0/1 脉冲**。
- **脉冲路径** = 吸完后的二值 GeMM（select-add / AC）。Prosperity / Gustav NRV-as-psum / FireFly-S Bitmap AND 都是必须抄全的 **A**。
- **残差 / PED / I24** = **另一条连续张量**。不要写成「连续 AT-LIF 幅值被两个 MAC 共用」。
- 阈值前 T10 PSN mix \(H=WX\) 是连续，再阈值，再吸收。
- 作废：不可吸收 int8、HBG-RP Absorb=NO 当身份、MX3P 换神经元、workflow-1 的连续 θg 双 MAC。

---

## 1. 决策（所有轮次合成）

| ID | 处置 | Codex 做什么 |
|---|---|---|
| **G1 / H1** 吸收切口上类型化 last-use：同一生产者 `{binary GeMM done, PED/I24 done}` 才释放行 | **修订，不是标题** | **先测**，不画新 RTL |
| **H2** 8088 等待类 | **实验** | **你现在的主任务**（见 §4） |
| G2 光流几何 / 空间局部 skip | **停标题** | 只作 ASNA-Flow / EventShiftFlow / ExSpike 对照 |
| G3 全域投影 BN | **卫生** | Stage B 必须记账；不是 X |
| G4 训练二值支撑密度 | **停标题** | **不要**把仅有的 paired AEE recovery 花在这里 |
| Overlay / skip spikeformer FPGA | **停标题** | 同刊 ESTU 已占 |
| CIM / AIMC / ReRAM | **A 然后停标题** | 可抄 skip 语义，不能当标题 |
| IAND 残差 | **负对照** | 我们保留 PED |

没有选出标题机制。广扫后仍 **未定位** 吸收切口双 last-use 的硅或编译器 IR。

---

## 2. 目录地图

| 路径 | 给谁看 | 内容 |
|---|---|---|
| `IDENTITY_ATLIF.md` | 必读 | 锁定身份 |
| `CODEX_HANDOFF.md` | 必读 | 本文件 |
| `round2_absorb/REPORT.md` + `DECISION_LOG.md` | 必读 | 吸 θ 后重挖；G1–G4 |
| `round3_gap/REPORT.md` + `DECISION_LOG.md` | 必读 | FireFly-T / DATE / IAND / 8088 观察 |
| `round3_gap/literature/R3L4_full_chain_obs.md` | 必读 | **已核对 JSON 的数字** |
| `round3_gap/hypotheses/H_8088_wait_class.json` | 必读 | 8088 四条候选假说 |
| `round3_gap/adversarial/ADV_H1.md` | 建议 | 审稿人怎么打 H1 |
| `adversarial/ADV_ESTU_same_journal.md` | 必读 | 同刊碰撞。文中「连续 θg」合同句 **过时**，对象仍是二值 SSA skip |
| `round4_wide/REPORT.md` + `DECISION_LOG.md` | 建议 | 顶会广扫结论 |
| `round4_wide/literature/V0_seed_catalog.md` | 建议 | 广扫种子表 |
| `round4_wide/literature/V1_isscc_jssc_cicc.md` | 需要硅对照时 | ISSCC/JSSC/CICC ≥20 条 |
| `round4_wide/literature/V3_date_dac_iccad_fpga.md` | 需要 FPGA/EDA 时 | DATE hybrid、ICCAD 3D、SPARTA、COBRA… |
| `round4_wide/literature/V4_tcas_tvlsi_tcad.md` | 写信件前 | 同刊/CASS 碰撞表 24 条 |
| `round4_wide/literature/V7_local_unread_p0.md` | 需要 2025–26 新文时 | 本地未读 p0 精读 ≥34 条 |
| `round4_wide/literature/V10_compiler_fusion.md` | 排程/IR | LoopTree 等 **没有** 分型双 last-use IR |
| `round2_absorb/literature/R2L*.md` | 脉冲 GeMM A | Prosperity/Gustav/FireFly/PSN/PED |
| `literature/L6_event_of.md` | 事件 OF 算法 | 禁当前帧 flow 神谕 |
| `full_chain/README.md`（测量树） | 硬件 | Stage B NOT_CLOSED |

**不要当证据：** `web/DEEP_RESEARCH_WORKFLOW.md`（workflow-1，错误身份）；round-4 的 deep-research-5/6（verifier 全失败）。  
**可用但 Partial：** `round2_absorb/web/DEEP_RESEARCH2.md`、`round3_gap/web/DEEP_RESEARCH4.md`。

Round-1 `independent/` `fusion/` 按旧身份，只作档案。

---

## 3. 必须抄全的 A（不是贡献句）

**吸完后二值 GeMM：** Prosperity 产品稀疏；GustavSNN CPTB/NRV 当 **GeMM 部分和**（不是 LIF 膜）；LoAS FTP；FireFly-S Bitmap AND；Phi PWP；Bishop SAC/AAC；ESTU group-4 skip；FireFly-T 稀疏引擎；Li 2501.07825 地址编码 SDSA；ELSA mini-batch Gustavson；SegFold 动态 Gustavson。

**阈值前 T10：** PSN + da4ml CSE。lifting 的 159+35 RNE 不是标题。

**残差代数：** SDT membrane-shortcut（加在 SN 前，脉冲保持二值）。Spike-IAND 用 IAND **删掉** ADD → 保留 PED 时是 **B**。

**Overlay 控制：** FireFly-T 双引擎；DATE 2025 **层切分** dense 输入 / sparse 其余。都不是 G1。

**事件 OF 硬件（禁「首个」）：** SENECA FireNet；TrueNorth OF；plane-fit FPGA；EventShiftFlow；ASNA-Flow 摘要（空间局部性已占）。

**BN：** SDformerFlow eval 关闭 running-state 跟踪 = 本网 native proj BN 软件合同。RISCSparse 冻结 BN **不能** 直接套到 live `10×96×120×160`。

**算法双路径已有人写过：** SymbolicLight V1（2605.21333）二值门 + 连续残差流（语言模型、dense kernel）。不要声称切法本身是 X。

---

## 4. Codex 主任务：8088 wait-class（不要先画 RTL）

观察（已在 JSON/MD 核对，表 **不可相加**）：

| 构造 | 文件 | 数字 |
|---|---|---|
| 源核 always-ready | `schedule_compare_same_port/source_service.md` | 6914 → 6170（−10.76%） |
| 源核长背压 | 同上 | **两端 8088**；FIFO 满等待 1159 vs **1903** |
| 两段写回 ready | `two_stage_writeback/result.json` | 6938 → 5354（−22.83%）；blocked **全 8088** |
| 整数消费者 | `consumer_service_result.json` | 758777 → 714889（−5.78%） |
| 整帧预约 | `full_chain/README.md` | lifting **−6.3981%，NOT closed**；FP 预览+sn2 ≈ **49%** |
| delayed-V | `full_chain/consumer_lifetime_result.json` | `arithmetic_saving=0`；BN 硬件未闭合；55.3MB vs 18.4MB ≠ 净服务 |
| preview-V | `preview_v_schedule_result.json` | ordinary **10396** slot |
| AEE | 既有 valid825 | 1.219801338 vs 1.232979368，Δ+0.013178；相对 +0.005 **失败** |

**假说（`round3_gap/hypotheses/H_8088_wait_class.json`）：**

1. **C1 阻塞接收 FIFO**（优先）：源核 8088 是测试波形（896/1024 unready）。无穷 sink 两端都应掉下 8088。
2. **C2 PED last-use（G1）**：只应出现在 **双消费者核**，`s_done` vs `r_done` 滞后非零。
3. **C3 BN 屏障**：源核 tag 应为 0；投影之后才有。
4. **C4 结构/指令**：改 last-use 政策 8088 不动。

**请交付：**

1. 源核：无穷 sink vs 现有 blocked 波形，两端学生，同一资源点。
2. **双消费者核**（gate + PED/I24，不是源-only）：stall tag `{fifo, spike_BP, PED_BP, BN, other}` 加总到总等待。
3. `s_done` / `r_done` 周期滞后直方图。
4. 不要把 6.4% 预约、−22.83% ready、−5.78% 整数消费者加在一起当净服务。
5. 同端口净服务门：**≥15%**；AEE 绝对 ≤1.259 且相对 ordinary ≤+0.005。

若 C1 成立而双消费者核 PED_BP 不是质量 → **停 H1 布局**。  
若双消费者核 PED last-use 是质量且服务 ≥15%、AEE 过门 → 再修订 H1 **一页**机制。

---

## 5. 明确禁写

- 首个事件光流硬件  
- CIM / OpenROAD-as-PPA / 模块动物园  
- 「我们跑了 Prosperity / FireFly-T / DATE hybrid」当标题  
- 连续 θg 双 MAC  
- 删 35 次 RNE、delayed-V、preview 转置当 X  
- 当前帧最终/中间 flow 当 r1 skip 神谕  
- 并行 ISCAS 论文（用户约束）  
- 发明 Yosys/OpenROAD 硅 PPA；组件比相乘当 FPS  

---

## 6. 调研完成度（诚实）

| 轮 | 状态 |
|---|---|
| Round 2 吸 θ 重挖 | **完成** |
| Round 3 缺口精读 + 独立再构想 | **完成** |
| Round 4 顶会广扫 | **决策包完成**；V1/V3/V4/V7/V10 有全文级表。V2/V5/V6 未落盘。ASNA-Flow / ERAFT FPGA / Fang 正文 / COMPASS / SPARTA PDF **仍 unresolved**。摘要已够用来停 G2 和 overlay 标题。 |
| 8088 wait-class | **未测** — 交给你 |

Grok 使用 scientific-brainstorming / hypothesis-generation（Kassis et al., 2026, arXiv:2609.00065）。
