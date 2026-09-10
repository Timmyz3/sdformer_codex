# A+B 叠层深度调研：GustavSNN 模式迁移到本工作

日期：2026-09-08（Asia/Shanghai）  
范围：事件光流 Spikeformer（Motion C12 ep34）· 只投 TCAS-II  
方法模板（对齐 GustavSNN）：

> **完整底座 A**（已有论文/已闭环对照）  
> **+ 未解决问题 B**（A 没吃掉、且钉在本网剩余最贵算子上）  
> **→ 可测增量 X**（同资源净收益；否则停）

硬约束沿用仓库纪律：θg 且 τ≠θ；非因果 T10 PSN；生产 nts07/主稿只读；禁止把 CPU 服务拍写成 RTL 加速比；不把已否证路线复活成标题。

---

## 0. GustavSNN 的模式到底是什么（校准）

GustavSNN **不是** Prosperity 的直接升级版，而是：

| 层 | GustavSNN 实际做法 |
|---|---|
| 祖先/外壳 | tick-batch（SNN 加速器族已有） |
| 底座 A | **Gustavson SpMM + 局部膜**（相对全局 Vmem 搬运） |
| 未解决 B | neuron-centric tick-batch **吃不掉时间稀疏与搬运** |
| 增量 X | **CPTB 列并行 + NRV 跳空行**（再对照 Prosperity） |

Prosperity 自己也示范叠法：ProSparsity 可挂到 LoAS。  
对本工作：**选一个已闭环 A，找 A 在 ep34/粗头后工作量上仍留的洞 B。**

---

## 1. 本网现状：什么已死，什么还贵

### 1.1 已否证（不得再当标题）

| 路线 | 证据 | 处置 |
|---|---|---|
| 旧 C1 Prosperity 融合 / C2 静态共享 | 同资源无正增量或加法≠周期 | 只作对照 |
| 时间类别 vs 打包时间（弱对照下 14–20%） | 强分块+共同 NR4 后仅 0.58%～1.70%；三原行下类别更慢 | 撤回贡献 |
| 当前 NR4 费用训练版 | 匹配边际后真实组 ≤ 打乱组 | 停该版本 |
| 共同 C16 作主机制 | 不胜普通 hidden50 | 可选布局 |
| 支持集投影共享 / 全 T10 严格提前界 | 额外 ~1%–1.7% | 停 |
| code→code FFN 开 RTL | 概念薄 + bank 穿不过 | 暂不开 |

### 1.2 剩余最贵（粗头 `preds.2` 后，点积代理）

| 算子族 | 剩余份额 | 含义 |
|---|---:|---|
| patch 卷积（残差为主） | **~34.8%** | 最大头，机制覆盖不足 |
| FFN（FC1+FC2） | **~26.0%** | S2 六块占 FFN 大头 |
| 其中六个 S2 FC1 | ~11.4% | 当前主战场 |
| PSN 连续算术（另计） | FFN 相关约占 PSN 的 ~39% | 必须与突触同设计 |

强控制必须保留：完整 Gustav 服务模型、普通 **hidden50**（valid825 AEE 1.1647）、**row 2:4**（1.1636）。

---

## 2. 推荐底座池 A（按可叠性）

| ID | 底座 A | 已解决什么 | 对本网仍留的洞 B |
|---|---|---|---|
| A1 | **GustavSNN** GP+CPTB+NRV+局部 S | 时间批、空行跳过、局部状态、双稀疏交（§VII-B） | 非因果 T10 消费者；θg；广播组并集请求；有损共同完成 |
| A2 | **Prosperity** ProSparsity | 模式复用部分和 | 连续 θg / 与 GP 广播并存时增量已薄 |
| A3 | **HYTE + Buffets** | 分块/容量/信用多播 | 不改变本网稀疏结构本身 |
| A4 | **HiNM / VENOM / CRISP** | 输出聚类、两级 N:M、块导航 | 未按本网 T10+FC2 消费损失与物理 64b 字优化 |
| A5 | **Avalanche** | SpMM 完成回收、输入复用 | S 在 T10 消费前不能释放 |
| A6 | **SparseInfer / BitFair / DejaVu** | 预测后跳权、提前终止、层/神经元路由 | 正 θg、非因果多 T、共享请求并集 |
| A7 | **Flextron / 嵌套宽度** | 动态宽度、费用路由 | 共享源并集可能≈满宽 |
| A8 | **S3Net / 子流形稀疏前端** | 事件坐标稀疏卷积 | 不是 θg Spikeformer；需重训 stem |
| A9 | **FireFly-T / ASTER（旁路）** | Spike attention 稀疏引擎 / 层跳过 | 注意力份额小；CIM/FPGA 故事不同 |
| A10 | **ERAFT / 流预测唤醒（算法侧）** | 光流语义 tile wake | 未接入 SNN-Transformer 调度 |

**默认主底座：A1。** A2 仅对照。A4–A6 是最值得叠的稀疏/条件执行先验。A8/A10 对准 patch 大头。

---

## 3. A+B 短名单（按建议优先级）

评分：新颖增量 / 适配 / 本地证据（已有数字）/ TCAS-II 潜力。主观研究分，非录用概率。

### ★ P0-1　广播域消费误差剪枝（推荐先做算法门）

- **A** = HiNM 重排聚类 + VENOM/CRISP 两级结构 + Gustav 请求前跳块  
- **B** = 现有幅值/W² 分组不能保证「同一 NRV 广播组」删同一物理源字；hidden50 虽强但直接砍消费者  
- **X** = 在固定 H8（或真实广播组 G）内，用 **完整 T10 门 + 真实 FC2 损失** 选可共同删除的 C16 物理字，H 重排同步 W1/τ/θ/W2  
- **挂点** = 先 s2b3，再扩 S2；最终要能迁到 patch 才有系统意义  
- **对照** = 原分组 / 完整 HiNM(二阶) / 消费者分组 / hidden50 / row2:4；同 50% 槽、同 train32  
- **杀门** = 不胜 HiNM 或 hidden50；或只涨 W² 不降门/FC2/流误差；或并集仍读满 C384 且无周期余量  
- **分** = 新颖 5.5 / 适配 8 / 证据 2 / 潜力 **7**  
- **为何像 Gustav** = 完整继承结构稀疏执行链，只改「按本网消费者定义的洞」

### ★ P0-2　共享请求组的有损共同完成（条件执行）

- **A** = Gustav 共享请求同步 + SparseInfer（预测后跳权）+ BitFair（学习提前终止）  
- **B** = 严格后缀界在单 (p,h) 可免 ~5.3% 逻辑 W，但 H8 等最慢后只剩 ~1.0%——**共享源字被最慢消费者钉死**  
- **X** = 在固定检查点（如 C192）预测整组 T10 门字，**一组接受/继续**；接受则发预测 θg（含预测非零），未接受继续原算  
- **挂点** = BN1 固定、τ 可编译的 s2b3 FC1→PSN  
- **对照** = 完整 Gustav / 独立每神经元预测+相同组关闭 / SparseInfer 式 / hidden50  
- **杀门** = 净服务不胜同精度静态窄层与独立预测；或预测开销≈再做一次 PSN；或 AEE 爆  
- **分** = 新颖 5.5 / 适配 6.5 / 证据 2 / 潜力 **7**  
- **注意** = 有损；不能写成无损早停。DAC’25 DJP/GRASP 全文未核到前，不作「无人做」宣称

### P1-3　依赖生存期感知剪枝（Avalanche×非因果 S）

- **A** = Avalanche 完成回收 + Gustav NR4/W 驻留  
- **B** = FC1 最后贡献到达后 S 仍被非因果 T10 占用，不能按 SpMM last-use 释放；现有滚动 2-live 相对最佳固定配对几乎无增量（C16 择优仅 ~0.07%）  
- **X** = 训练掩码时约束「沿公共 C 顺序同时未完成的广播组数」，使短生存期与完整 T10 同时成立  
- **杀门** = 免费供数上界已无余量；或同 AEE 不胜固定配对/hidden50  
- **分** = 新颖 6 / 适配 6 / 证据 1（静态探针偏负）/ 潜力 6  
- **建议** = 先算乐观上界；上界不够就停，不必先训练

### P1-4　`F_live>1` 公平重测类别表示（窄否证/复活）

- **A** = 完整 Gustav（含多输出上下文）  
- **B** = 现杀类别路径时强制 F_live=1，可能低估「更小状态→更多并发上下文」  
- **X** = 同学生、同 F_cache/端口，搜索合法 F_live，只比较类别 vs 时间  
- **杀门** = 相对强对照仍 <5% 或不稳  
- **分** = 新颖 3（复活探针）/ 适配 8 / 证据 3 / 潜力 4  
- **定位** = 一天级探针，不是主创新

### P1-5　Patch 最贵 residual 的结构稀疏/子流形前端

- **A** = S3Net 稀疏坐标链 或 CRISP/TB-STC 结构剪枝 + 普通滑窗/Gustav 卷积迁移  
- **B** = patch 仍是第一贵（~34.8%），当前机制几乎全钉在 S2 FC1  
- **X** = （a）只剪/跳最贵 `patch residual conv1`；或（b）训稀疏 stem，保持到首次 θg 再进固定窗  
- **对照** = 等预算窄稠密 stem / stride2 / 原 stem；必须计索引、scatter、halo  
- **杀门** = 跨 T 支撑很快铺满；或窄稠密同精度更便宜；或 AEE>1.259  
- **分** = 新颖 5–6.5 / 适配 7 / 证据 1 / 潜力 **7.5（若做成）**  
- **风险** = 训练身份重；但是唯一对准最大头的叠层

### P2-6　注意力无损跳过旁路（不成主标题除非份额上来）

- **A** = FireFly-T AND-PopCount / 本仓库 Motion-XOR 叶  
- **B** = ep34：K 单 token 零 ~83.5%，配对相同 ~70.5%，复用零开销上限 ~35%；但 K=0 **不能**删分数项（改 Shiftmax 分母）  
- **X** = D1 只跳 gated-K 输出乘积；D2 行内分数 memo；禁止整窗 dirty 一刀切  
- **杀门** = 计入比较/保存/调度后系统周期份额仍 <2–3%  
- **分** = 新颖 6（MX3P 故事）/ 适配 7 / 证据 5（普查有）/ 潜力 5（短文旁路）

### P2-7　光流语义 tile wake（OP-STW 重生，因果版）

- **A** = ERAFT/TCAS-I 方向预测唤醒 + 事件前端  
- **B** = 零跳过不是光流语义；GrokBot 旧 OP-STW 因用本网输出/峰值被否  
- **X** = **禁止**用本网最终 flow；只用上一帧已完成 flow / 事件 TDE / 外置小预测器  
- **杀门** = 相对强零跳过无净省；或运动探针式费用>1（历史差分 tile 曾 1.5×）  
- **分** = 新颖 7（若因果且硬件闭环）/ 适配 6 / 证据 1 / 潜力 6  
- **定位** = 高概念、高风险；不宜与 P0 并行开很多

### 明确不推荐当 A+B 主线

| 想法 | 原因 |
|---|---|
| Prosperity∪APEC 再融合 | 已测负 |
| 纯抄 Gustav 无 X | 复现工程，不是 TCAS-II 贡献 |
| 动态宽度先小后大（Flextron 直迁） | 概念差，且 hidden50 门槛高 |
| 模拟 CIM / 未核全文的 DAC 口号 | 证据链与工艺合同不合 |

---

## 4. 建议的两周试点（只做门检，不砌 8×8）

**Week 0 纪律**  
所有实验分母 = `完整 Gustav 服务模型 + hidden50/2:4`；不过门不开 2×4 以外 RTL。

**Day 1–3：P0-1 消费误差分组（s2b3）**  
四轴同槽同步；diverse10 AEE + 物理源字/NR4/PSN 服务。  
过门：同精度下物理字或服务显著优于 HiNM 与幅值 C16，且不输 hidden50 太多。

**Day 2–4：P0-2 共同完成预测（并行小预测器）**  
C192 检查点；独立损失 vs 组费用损失。  
过门：请求数/S 存活下降且 valid10 AEE 过门，并胜过独立预测。

**Day 3：P1-4 F_live 窄测**  
半天级；复活或钉死类别路径。

**Day 5–7（仅 P0 有苗头时）：**  
最小 2×4 PE 切片只挂过门的 X；否则改开 P1-5 patch 探针，而不是扩 8×8。

**明确不做：** 完整 8×8 Gustav 复刻、叶子 PPA、改 main.tex、六层同时剪枝大训。

---

## 5. 若只能选一个投稿句（预写）

**首选（若 P0-1 或 P0-2 过门）：**

> Under a full Gustavson-style sparse supply baseline, we co-design the remaining shared-request / noncausal-T10 consumer bottleneck of an event-OF spikeformer FFN by &lt;broadcast-aware prune | group-complete speculation&gt;, showing same-resource gains beyond structured pruning and independent predictors.

**若 FFN 岛继续空、patch 过门：**

> We target the dominant patch residual front-end of a frozen event-OF spikeformer with architecture-aware sparsity, under equal-budget dense and Gustav-migrated convolution controls.

---

## 6. 结论

1. **Gustav 模式可用，但要对准「A 已优化后仍在的洞」。**  
2. 当前洞不再是「再做一个时间类别码」，而是：**广播并集请求、非因果共同完成、以及未动的 patch 大头。**  
3. 最像 Gustav 的下两条是 **P0-1（消费误差结构剪枝）** 与 **P0-2（共享组有损共同完成）**；patch 是系统潜力最大但训练最重的备选。  
4. Codex 若继续无 X 砌满 Gustav RTL，路径不正确；本调研给出的是可执行的纠偏菜单。

