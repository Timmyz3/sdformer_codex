# 逐卡审阅记分板（接手 Codex，2026-09-11）

审阅范围：GrokBot 隔离 RTL 树的自有刀 + `survey_ab_fusion_20260910/idea_cards/` 全部 **245** 张文献卡 + grok46 / Card G–I 机制清单。  
方法：每卡只交 A（完整先验）/ B（本网未解决洞）/ X（相对 A 的真增量）。停布局，不杀家族。卡面自评分、iverilog PASS、OpenROAD DRC=0 **不是** TCAS-II 证据。

身份约束：Motion C12 / H67 / ep34；ATLIF 是**连续 θg**，不是二值发放。主研究线仍是 patch r1 可学习 T10 lifting。精度门绝对 AEE≤1.259、相对同预算对照 ≤+0.005；lifting 当前 +0.01318，完整链未闭合。

## 一句话

**文献卡里没有标题级 X。GrokBot 三刀信件也不能当标题。**  
真正还站得住的是：把 Prosperity / Gustav / ExSpike / CFMP / FireFly-S / VENOM / LoAS-FTP **抄全当底座**，在 lifting + 真实双消费者链上找净服务；GrokBot 树里只留账本/容量胶水，不要再扩阈值比较器动物园。

## 交付文件

| 文件 | 内容 |
|---|---|
| `ALL_IDEA_CARDS.csv` | 245 张文献卡逐卡表（已并 Codex 未完成的 113 行 + 补审 132 行） |
| `review_grokbot_native.csv` | 39 条 GrokBot/Grok46 自有机制 |
| `GROKBOT_NATIVE_VERDICT.md` | 打开 SV 后的字面电路与停刀清单 |
| `RUBRIC.md` | 本轮打分规范 |
| `review_part_b.csv` 等 `review_batch*.csv` | 分批底稿 |

证据深度几乎全是 `card only`。打开了 GrokBot `.sv` 的标为 `card+rtl tb`。没有把卡内 PDF 路径当成“本轮已精读原文”。

## GrokBot 自有刀（先看这个）

打开 RTL 之后，信件三刀都是比较器：

| 自称 | 实际电路 | 决定 |
|---|---|---|
| OP-STW（刀1） | `wake = (\|Δflow\|>TH) \|\| (event_cnt>TH)` | **停主刀**。Codex 已测残差卷积因果运动差分相对 bit-skip 是额外功 |
| HBG-RP（刀2） | `g=\|amp\|>EPS; p=g?amp:0`，约 24 cells | **停主刀**。死区/时钟门，不是 dual-rail；int8 payload 是另开合同，ep35 捕获还是二值 |
| OGEC×PRRC→exact（刀3） | `exact_en=match_ok`；三级递减计数；容量 FSM | **停架构声称**。账本∩容量可留作控制胶水 |
| TDE3 / TMA / CFP / SCI / BiSAT / BUI | 1D 邻域年龄、相等计数、`sat(peak)`、`255-\|Δ\|`、fwd==bwd 平均 | **未进** `c1s_top`/`c2s_top`。停 first-HW 叙事，降为对照 |
| ECP / MW / SMAM / ADP / STH / SP / MFBD / ARM | 阈值或透传 | 停布局 |
| grok46 MX3P / dirty-lane / 二值 ATLIF 岛 | 明确依赖二值发放 | **停**。与本刊连续 θg 冲突 |
| Card I Top5 | 只有草图 | 不要实现 |

独立树自评 ~3.2/10 已经偏松。OpenROAD 18/18 DRC=0 只证明模块能布，没有 PDN/SPEF，没有 AEE，没有与旧 C1/C2 同资源对照。

可留的非标题约束：**连续 θg 不要吸进下一层 W**；PRRC 计数器 + exact 容量 FSM 可当控制平面。

## 245 张文献卡

novelty：0 = 83，1 = 132，2 = 30，**3/4 = 0**。  
没有一张卡达到标题级增量。novelty=2 的意思是“完整迁入后可能成为强对照或底座，仍不是 X”。

### 必须抄全的底座（保留完整迁移 / 强底座）

这些是 A，必须按原方法迁完再谈 X。已失败的是**具体布局**。

| 家族 | 代表卡 | 本地已知边界 |
|---|---|---|
| Prosperity 部分和复用 | `arxiv_2503.03379.md` | 当前 C1 融合比 Prosperity 慢；停该布局，不杀家族 |
| ExSpike / APEC / SpikeX | `ExSpike_APEC.md`, `arxiv_2505.12292.md` | 停旧 G4；完整多消费者共享未试完 |
| GustavSNN CPTB/NRV | `GustavSNN.md` | 2×4 切片仅功能；强对照后 class 名约 1.70%；完整链未闭 |
| LoAS FTP | `LoAS.md` | **停静态共享布局**（加法 −8.96%，周期 +0.23%）；FTP 家族保留 |
| CFMP / ConvFormer | `arxiv_2512.17555.md`, `isscc2025_convformer.md` | 分解/同 mask 是 A；私有尾本地未胜过共享 |
| FireFly-S 双侧稀疏 | `arxiv_2408.15578.md` | 要联合量化剪枝+训练；旧 s2b3 负例不能否全族 |
| VENOM / Phi / Bishop | `VENOM.md`, `Phi.md`, `arxiv_2505.12281.md` | 完整格式/训练未迁；共 mask 不是完整方法 |
| 条件计算 / 早停底座 | BitFair、ASTER、HeatViT、SpAtten | 分类熵/删 T ≠ 本网门证书 |
| 事件光流执行对照 | SNE、neuromorphic OF ANN vs SNN、MemFlow | 任务对照，不是免费 oracle |

### 明确旁路（不要进主岛）

- 二值 ATLIF / MX3P / 纯 mask-add 当本刊身份
- 模拟 CIM / AIMC / 空泛存内计算当数字 28 nm 贡献
- FlashAttention / vLLM / xformers / NVDLA 整 SoC 替换当前执行岛
- 分类 token 早退、全局匹配相关体、RAFT 金字塔换任务
- GrokBot 18 模块动物园、OpenROAD 当 PPA

### 待核全文（18）

多数是 `unresolved_*` 题名卡：有线索、没有可执行 A。补全文前不得开 RTL，也不得计“已读 245 篇论文”。

### 已停止的具体布局（文献侧）

目前表里显式“停止该布局”的主条目是 **LoAS 静态共享**。更多停止写在 `decision_and_reason` 后半句（Prosperity G4、持续检查、私有 R56 尾、精确 warp 差分、全合并 lifting 图等），见 Codex `CURRENT_LINES_AND_PLAN_20260910.md`。那些是实验负结果，不是本轮新测。

## 和当前主线怎么接

主候选不变：**可学习 T10 lifting + 完整常量编译 + 真实 θg/双消费者**。  
文献审阅的作用是把分母补齐，不是另开 245 条线。

允许并行准备、但不得抢 lifting 完整链档期的底座：

1. Gustav 供数 / NRV∩W / 同 k-ID 屏障（完整物理链，不是 2×4）
2. Prosperity 部分和，但是要换多消费者/有限执行，不再用已慢的 G4
3. 敏感 patch 上的结构剪枝（VENOM/HiNM/FireFly-S 全流程），不要在可删的 s2b3 上打磨
4. 训练诱导时间稀疏（Delta 层思想），对照必须含旧状态读写；已停的 1.53× warp 不算数

GrokBot C1*/C2* **不要合并进 nts07**。隔离树可以留作阈值胶水博物馆，不再为信件加模块。

## 本轮没有做、也不该从卡片推出的事

- 没有新 valid825、没有新 RTL、没有改 `main.tex`
- 没有把 245 张卡当成 245 篇精读
- 没有因为某卡 novelty=2 就开工训练

## 建议的下一步（仍按 Codex 原计划）

1. 把 lifting 源核接到真实前驱/后继的完成与 ready 依赖（`full_chain/`，Codex 只写了 `capture_reference.py` 就额度耗尽）
2. 同资源净服务仍要单独过 15% 门；源侧 22.83% 含通用指令融合，长背压会被吃光
3. 相对 AEE +0.013 未过 +0.005 门之前，不晋级性能 RTL，也不写标题句
4. Prosperity/Gustav 保持第二队列，等完整链分母出来再决定是否重开未试接口
