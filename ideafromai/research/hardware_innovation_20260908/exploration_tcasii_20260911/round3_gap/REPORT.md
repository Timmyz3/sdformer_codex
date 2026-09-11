# Round 3 报告：缺口精读 + 证据后再独立构想

**日期：** 2026-09-11  
**性质：** 提案与文献核对。不是测量结论。  
**身份：** `../IDENTITY_ATLIF.md`（用户锁定：`{0,θ}`，θ 吸入下一层 W，层间二值）。  
**方法：** scientific-brainstorming 文献后再开独立轮（Kassis et al., 2026, arXiv:2609.00065）。

Round 2 留下：G1 修订、G2/G4 停标题、G3 卫生。本轮只补未精读先验，并在新证据包上重开独立构想。不接管 Codex RTL。

---

## 1. 新先验（抄全 A，不当标题）

| 论文 | 打开深度 | 机制对象 | 相对 H1 / 本地 r1 |
|---|---|---|---|
| **FireFly-T** 2505.12771 全文 | 本地 txt + ar5iv | 双引擎 overlay：稀疏 spike×W 卷积 + 二值 AND-PopCount 注意力；残差是 **SN 前膜** AXI 口 | overlay 控制 **A**。**不是** 同一生产者的二值 GeMM + 连续 PED 双 last-use |
| **Spike-IAND** 2503.19643 全文 | 本地 txt | 残差 ADD→IAND 以保持全脉冲；LIF 展开 mux `111/101/000`（T=4/2/1） | **负对照**（我们保留 PED）。T 展开是因果 LIF，**不是** 非因果 T10 PSN |
| **DATE 2025 hybrid** 2411.15409 | arXiv HTML 全文 | 稠密核吃 **输入层**（非二值、非稀疏），稀疏核吃其余发放卷积 | overlay **A**。**层切分**，不是 G1 |
| **ASNA-Flow** TVLSI 2025 | **仅摘要** | 事件 OF 异步神经形态；宣称 OF **空间局部性稀疏**；28 nm 104 FPS / 7.9 mW / 0.3 pJ/SOP | 空间局部跳过 **标题级已占**。G2 继续停。全文未打开 |
| **ERAFT FPGA** ISCAS 2025 | **仅摘要** | **帧** RAFT，VCK190，Middlebury 86 fps @ 640×480 | **不是** 事件、不是 SNN |
| **EventShiftFlow** 2605.28312 全文 | 本地 + HTML | 1-bit occupancy 假设 popcount；Artix-7 原型 | 又一个事件 OF FPGA **A**。不是 SNN、不是稠密 DSEC |
| **SENECA** 2407.20421 | 已有 L6 | FireNet 已是 event-OF 硬件 | 禁止写“首个事件光流芯片” |
| **FlexSpIM** ISCAS 2025 | 本地 author 全文 | 数字 CIM，W 与膜统一存储，WS/OS | **A 然后停标题**（CIM 禁） |
| **SpikePool** 2510.12102 全文 | 本地 + HTML | GPU 上 max-pool 替 SSA | 算法 A，无加速器 |
| **Fang CICC 2024** | **仅摘要** | 3D AC-only，反 hetero SNN+CNN MAC 核 | 二值 AC GeMM **A**。PDF 未打开 |
| **ESTU** TCAS-II 2025 | 已有 AAM | 二值 SSA skip + 分类 + mW | 同刊碰撞。ADV 里“连续 θg”合同句已过时，对象仍是二值 overlay skip |
| **SDformerFlow** 2409.04082 | 本地 | eval 关闭 BN running-state 跟踪；deformed 1×1 PED | 与原生投影 BN **同一软件合同**（G3）。PED 是算法已有，不是硬件双 last-use |

**仍未打开全文：** ASNA-Flow 正文、ERAFT FPGA PDF、Fang CICC/JSSC PDF、COMPASS MICRO PDF。摘要 ≠ 机制精读。

Workflow-1 把“不可吸收连续 θg 双 MAC”当 X：**作废**，不要再用。Workflow-2 与本身份一致。Workflow-4（Partial）全文：`web/DEEP_RESEARCH4.md`。补充：FireFly-T 仍把膜阈值当运行参数加载，**不是** θ→W 吸收切口；它的“双 last-use”是 SN 后二值注意力 + SN 前膜残差（SDT 类 A），不是吸收切口上的二值 GeMM + PED。ERAFT/SpikePool/EventShiftFlow 在该工作流核验包里未过线；本轮已独立打开 SpikePool 与 EventShiftFlow 全文。

---

## 2. full_chain 观察（不是净服务）

核对过 JSON/MD：

- 源核 `source_service`：always-ready 6914→6170（−10.76%）；长背压 **两端 8088**；FIFO 满等待 1159 vs **1903**（lifting 等得更多）。
- 两段写回 `two_stage_writeback`：6938→5354（−22.83% ready）；blocked **全 8088**。与上表 **不可相加**。
- 整数消费者 758777→714889（−5.78%），另一构造。
- 整帧预约 −6.3981%，**未闭合**；FP 预览+sn2 约占该构造 49%。
- delayed-V：`arithmetic_saving=0`，BN 硬件未闭合；55.3 MB vs 18.4 MB 是寿命不是服务。
- preview-V ordinary 10396 slot。
- AEE 1.219801338 vs 1.232979368，相对门失败。

**候选解释（不是发现）：** 源核 8088 更像是测试的阻塞接收波形，不是 PED last-use。G1 若存在，应出现在 **双消费者核**，不在源-only tile。

---

## 3. 独立再构想（48 条 → 8 簇）

六路视角（电路 / 架构 / 编译 / 双路径 / 事件 OF / 训练）在 **同一证据包、不看 G1–G4 目录** 下重开。独立文件：`independent/R3Q01…06.md`。

结果：**没有新的标题对象。** 多数卡片收敛到 H1（与 G1 同一对象），其余落进 overlay 复皮、空间跳过、BN 卫生、训练密度（已停）。

详见 `fusion/FUSION_R3.md`、`DECISION_LOG.md`。

---

## 4. 现在硬件从哪入手（仍交给 Codex，不扩 RTL）

1. **脉冲 GeMM：** Prosperity / Gustav NRV-as-psum / FireFly-S / Phi / Bishop 当对照底座抄全。
2. **Overlay 控制：** FireFly-T + DATE 当 A 抄全；不要当 X。
3. **唯一还对准已测洞的问题：** 把 8088 **按等待类切开**（FIFO / 脉冲消费者 / PED 消费者 / BN），对象是 **双消费者核**，不是源-only。H2 是实验，不是信件。
4. 直方图若显示 PED last-use 才是质量，再修订 H1 一页机制；若是 FIFO 或 BN，**停 H1 布局**。
5. 15% 过了且 AEE 仍 +0.013，才允许一次 paired recovery，对象是精度，不是 FireFly 式稀疏训练。

---

## 5. 明确不要做

- 不可吸收 int8 / 连续 θg 双 MAC（身份错）
- “我们跑了 Prosperity / FireFly-T / DATE hybrid”当标题
- 光流空间局部性当标题（ASNA-Flow 摘要已占）
- 首个事件光流硬件
- CIM、OpenROAD-as-PPA、模块动物园
- 删 35 次 RNE、delayed-V、preview 转置当 X
- 把唯一 paired recovery 花在二值支撑密度

独立 6×8 条在 `independent/`。文献 `literature/R3L*`。网页抓取 `web/R3W1_date_asna_eraf.md`。对抗 `adversarial/ADV_H1.md`。
