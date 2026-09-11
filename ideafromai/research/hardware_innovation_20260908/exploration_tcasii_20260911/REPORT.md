# 探索报告：硬件从何入手，怎样才可能冲击 TCAS-II 强录用

**日期：** 2026-09-11  
**性质：** 提案与文献核对，不是已验证发现。  
**入口：** 本目录 `README.md`。

本轮按 scientific-brainstorming：先冻结观测，六路**独立**构想（不看 idea 目录），再精读本地全文 + 2025–2026 会刊，再融合，再对抗审。没有自动选出“赢家”。

## 0. 先回答你的迷茫

硬件不要从 GrokBot 18 个模块或再扫 245 张卡开始。  
**从 Codex 正在测的那个洞入手：** 源核 always-ready 能少 22.83%，长背压两边都是 **8088**，完整链还没闭合，相对 AEE **+0.013** 没过 +0.005。

TCAS-II 五页只容一个机制。强录用需要同时：

1. 审稿人能看见**完整先验**（Prosperity / GustavSNN / LoAS / ESTU）被抄全当对照；  
2. 有一个他们论文**定理盖不住**的对象；  
3. 同端口同状态同背压下净服务 ≥15%，且 AEE 门过。

本轮**没有**定位到已经能写进标题的 X。对抗审把四条融合岛都降成 revise / hygiene / stop / pause。这比再发明一个新名字更接近强录用路径。

同刊对照：ESTU（TCAS-II 2025，DOI `10.1109/TCSII.2025.3626209`）已经是**二值 SSA + 活动跳过 + 分类精度 + mW**。再投“又一个可跳过的脉冲 Transformer 加速器”会被当换皮。信件必须测 ESTU 没测的合取：**连续 θg × 双消费者 × DSEC valid825 AEE × 同端口净服务**。

## 1. 本轮实际读了什么

独立构想 6 份 × 各 8 条（`independent/`）。  
全文精读（本地 txt/pdf，不是卡片）：

- Prosperity HPCA 2025（`2503.03379`）：二值 GeMM 行的 EM/PM 前缀复用；**明确不做 intersection、不做第二前缀**。  
- ExSpike FPL 2026（`2606.20414`）：邻位 AND 压缩，正好是 Prosperity 拒绝的 intersection；残差 SRAM 仍是脉冲。  
- GustavSNN HPCA 2026：CPTB + 片内膜 + NRV；与 Prosperity **并列先验，不是叠加**。身份是二值、因果 LIF、单消费者。  
- LoAS MICRO 2024：FTP + 伪累加校正；正文写 Gust/OP **不适合** dual-sparse。  
- FireFly-S、SCNN、Bishop：双侧稀疏 / 笛卡尔积 / AAC 都绑定 `s∈{0,1}`。  
- ISSCC 23.2 ConvFormer/CFMP、LoopTree、RISCSparse：层融合与 retain/recompute；RISCSparse 把 BN 折成冻结 `Y=aX+B`，与本网**实际 batch 统计**相反。  
- VENOM、BitFair、SpikePack、IAND-Former：全部只能当 **CONTROL**。  
- 事件光流：BAT/SciFlow/EV-FlowNet 是 GPU 精度；唯一带硬件的 SENECA 是分类级 FireNet。

2025–2026 新出现、本地八个 ID 里没有的：**Phi ISCA 2025、ESTU TCAS-II 2025、FireFly-T IEEE TC 2026、da4ml TRETS 2026、DATE 2025 hybrid SNN、CICC 2026 ROM-LTE**。没有找到“连续 θg × 双消费者 × 事件光流 SNN-Transformer 芯片”。

后续 `deep-research` 工作流（状态 Partial）又核对了：Phi 两级 pattern、C-Transformer HDSC（JSSC）、Fang CICC 2024 三维时间并行阵、Kao 亚阈 8T CIM。CIM 仍出局。**CFMP 是级联特征图剪枝，不是层融合**；层融合是 ConvFormer 的 LFS。ASNA-Flow（TVLSI 2025）是列表外的事件光流硬件。详见 `web/DEEP_RESEARCH_WORKFLOW.md`。

## 2. 独立构想收敛到四条岛，对抗后的地位

| 岛 | 一句话 | 对抗结论 | 你该不该当标题 |
|---|---|---|---|
| **F1 双完成 last-use** | 编译后的 T10 活值要等 gate∧PED∧BN-stat 都用完才退休；占用是连续 θg 的并集，不是二值 NRV | **revise**；新颖 1、性能 1 | 先测 8088 是什么等待，再决定能不能立题 |
| **F2 付费全幅投影 BN** | 原生 proj BN 是 10×96×120×160 的真实 batch 统计，不能给小窗免费 μ/σ | **hygiene**；新颖 1 | **必须记账，不是 X** |
| **F3 整数 lifting 删掉 35 次 RNE** | dyadic/CSD QAT 让源图没有中间舍入 | **stop 该布局**；新颖 1、性能 1 | 不要把仅有的 paired recovery 花在这里 |
| **F4 前缀足够的 T10** | 训练让后缀对两个消费者都是精确 0 | **pause**；新颖 2、性能 1 | 8088 被证明是可前缀等待之前不要开训 |

F3 被停的理由已经很硬：系数已是 q12；da4ml 已经吐出 159 次加减；全合并探针 241>159+35；通用 `round→sat` 已经吃掉 always-ready 里的 13.56%；ALU 变短时 FIFO 满等待 **1135→2719**。删 RNE 会加深背压，而且相对 AEE 已经 +0.013，再收紧字母表大概率更差。

F1 仍是**唯一对准已测洞**的方向，但审稿人可以合法说这是 credit 流控或 Gustav NRV 换了谓词。要活下来必须先量两件事（提案）：

1. 8088 里有多少是 PED/BN 未就绪，多少是 FIFO 深度伪影。  
2. `support(θg)∪support(PED)` 是不是真的比 ordinary 稀疏；并集很可能更密，那 F1 会变成节流而不是加速。

## 3. 强录用路径（仍然是提案）

**不要换主岛去抄 Prosperity 当标题。** Prosperity 的复用定理建立在二值行相等/子集上，连续 θg 直接拆掉。本地 G4 ExSpike 融合已经更慢。

**不要把 lifting 的 CSE 或两级融合当 X。** 那是 A。

可执行顺序（和 Codex 主线对齐，不抢它的档）：

1. **让 Codex 把 full_chain 做完**（它正在做）。这是 F1/F2 的分母，不是新 idea。记下：原生 BN 是全幅 batch 统计。  
2. **拆 8088**：同一资源点上，源 FIFO 等待 vs PED ready vs BN 统计屏障，分项。这决定 F1 是修订还是停。  
3. 若并集占用真的更稀、8088 下降、完整链 ≥15%、AEE 仍过门：标题句只能是  
   *“continuous-θg dual-consumer last-use on a compiled T10 source, with Gustav/LoAS as copied controls.”*  
   不是 CPTB，不是 product sparsity，不是 OP-STW。  
4. 若 15% 过了但 AEE 仍 +0.013：这时才允许**一次** paired recovery，对象是精度不是再发明电路。  
5. Prosperity / FireFly-S / VENOM 继续当**完整迁入的对照**，第二队列，不并行铺开。

C2 目前仍没有过创新门的新主机制。TCAS-II 不必硬凑两刀。

## 4. 明确不要做

- GrokBot 三刀、Card G/H/I、MX3P、二值 ATLIF 岛  
- 把 ESTU/FireFly-T/Bishop 的跳过故事换名  
- 用 OpenROAD/Yosys 当 PPA  
- 把源核 −22.83% 和消费者 −5.78% 相加  
- 在完整链分母出来之前开生产 RTL 或改 `main.tex`

## 5. 身份锁定（用户确认，2026-09-11）

**AT-LIF 就是 \(\{0,\theta\}\) 二电平；推理时 \(\theta\) 吸进下一层 \(W\)。这是对的，也是本课题要的身份。** 详见 `IDENTITY_ATLIF.md`。

吸完之后层间是 0/1 脉冲。Prosperity / Gustav / FireFly-S 对这条路径是合法完整先验。  
GrokBot「不可吸收 int8」是另一份草案，与官方 AT-LIF 冲突，不能当冻结身份。  
残差 / PED / I24 仍是另一条连续张量，不要写成“AT-LIF 带着模拟幅值被两个 MAC 共用”。

PSN 原文出口仍是二值 \(S=\Theta(H-B)\)。tdBN 推理用滑动均值折进卷积，不是 live 全幅 batch 屏障。投影 BN 的全幅统计是本地捕获观测，不是 AT-LIF 论文的贡献句。

## 6. 局限

- 精读覆盖的是本地 150 篇 txt 里的优先子集 + 会刊扫描，不是全领域穷尽。  
- 对抗分是判断，不是录用概率。  
- Codex 完整链数字还在变；本报告以 2026-09-11 冻结观测为准。

本探索使用了 Scientific Agent Skills 的 scientific-brainstorming 与 hypothesis-generation 流程（Kassis et al., 2026, arXiv:2609.00065）。
