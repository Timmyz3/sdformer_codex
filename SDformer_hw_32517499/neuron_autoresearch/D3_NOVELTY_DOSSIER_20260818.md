# D3 新颖性档案（Novelty Dossier）：A3S 各向异性 stencil（方向场偏移 ±Δ）+ 方向感知唯一门

日期：2026-08-18。对象：D3 合同（CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md 的 D3 部分）与 D3 实现（D3_LOCAL5_A3S_IMPLEMENTATION_20260818.md）。
方法：WebSearch 中英多查询（14 轮，2026-08-18）+ WebFetch 精读 OPCM 论文 + 内部文献核对（hw_autoresearch_nts07/docs/439、docs/433 门槛、Local5 线 277/320 系列）。所有引用标注 arXiv 号/会议名/DOI；检索不到即标注"未检索到"，无虚构文献。未碰 GPU、未改任何现有文件。

---

## 0. 一句话结论

**D3 的三件套（方向场位图 2bit/pixel + Q7 网格精确位移 ±Δ=8 档（K2）+ 方向感知唯一门 1→3 类分裂（K5））在"方向场进入注意力分数域做门控"的交集上未检索到占位（算法侧无、硬件侧无）；但"方向选择性/方向场"是事件视觉里几十年的老主题（方向选择性神经元、TDE/sEMD 电路、方向先验光流估计），且"分数+偏置"模式本身有先例（SWIN relative position bias、FlashBias NeurIPS'25）——创新幅度质疑是真实压力，必须靠 K1-K5 恒等式与"新存储对象+新执行对象"三件套正面回应。新颖性风险等级：中（偏高，高于 D1）。**

---

## 1. D3 合同要素（对照审查用基线）

| 类别 | D3 要素 | 一句话 |
|---|---|---|
| 新算法算子合同 | A3S 方向场分数偏移 | Local5 5-lane 分数加方向场偏移：对齐 lane +Δ、正交 −Δ、self 0；Δ=1/16 = 8 个 1/128 档**网格精确位移**（K2：200 组 clamp 外与 Q7 量化 commute 逐位相等）；Δ=0 与现网 Local5 门逐位一致（K1：10 组随机平面 torch.equal，注入式训练锚点） |
| 新存储对象 | 方向场位图 + 双权重槽 | 2bit/pixel E/W/N/S 码（3×3 时域 XOR 梯度 argmax，复用现网 `_binary_temporal_k_xor_popcount` 流水）+ 对齐/正交双权重槽（Δ 固定参数，无需训练）；450bit/窗 <1% 现网存储增量 |
| 新执行对象 | 方向感知唯一门执行 | ident-K 目的地唯一门从 1 组 → 3 偏移类 {self 0, +Δ, −Δ}（K5：11250 个目的地全部 1→3；折叠 self 为 2 类权重）；非 ident-K 目的地方向场 2bit 查表决定 lane 分组；raw16 广播 ×2（诚实成本已记账） |
| 不动项 | Swin 窗口 (2,15,15) 与全部模型参数 | checkpoint 与 Local5 ep44 锚点直接兼容（valid825 对比口径不变，锚点 AEE 1.2819@ep44）；bsa_attention.py 纯追加 0 删除行 |

实测数字（check_d3 脚本 K1-K5，ALL PASS）：方向场 E/W 轴语义对齐 76-85%（K3）；运动承载像素上对齐 lane 量化分数 argmax 命中率 91.2% vs 基线 0.0%（K4 诚实指标）；ident-K 唯一门 1→3 类全分裂（K5）。

---

## 2. 文献检索记录（2026-08-18，WebSearch 14 轮 + WebFetch 1 篇）

| # | 查询主题 | 关键命中 | 结果 |
|---|---|---|---|
| 1 | anisotropic stencil accelerator hardware direction-aware kernel | **无结果** | "各向异性/方向感知 stencil 加速器"未检索到硬件占位 |
| 2 | event camera direction field prior gating computation | **OPCM（arXiv:2511.12961）**、Greatorex CVPR 2026 Findings（突触门控 SNN 光流）、Frontiers 2015（TDE 方向选择性） | 方向场先验存在于**软件光流估计**域；无注意力分数、无硬件算子 |
| 3 | direction-selective neuron SNN event camera hardware | Pignari et al. ICDL 2025（sEMD 四基元方向群体）、Paredes-Vallés TPAMI 2019（无监督方向选择性）、TDE 电路（Frontiers 2015） | 方向选择性输出分类/检测；**无注意力分数门控** |
| 4 | attention accelerator hardware score gating identity groups spiking | BSA（NeurIPS 2025，Shiftmax）、3D spiking transformer accelerator（arXiv 2024-12） | "identity groups/gate 分裂"未检索到；BSA 是项目自身基础 |
| 5 | stencil accelerator hardware 5-point HPC | Tenstorrent Wormhole stencil（arXiv:2605.07599）、IWOCL 2026 StencilStream GPU 后端、FPGA stencil DAQ（PMC8621947） | HPC 5-point stencil 加速器全部**各向同性**；无方向场 |
| 6 | optical flow attention spiking SNN event camera 2025/2026 | ASNA-Flow（IEEE 11142472）、SPECK、SENECA、Nature Comms Eng review | 事件光流硬件无注意力分数方向门控；ASNA-Flow 无注意力 |
| 7 | SA-CNN event camera direction classifier | 未检索到（仅有不相关命中） | 该方向检索失败，标注"未检索到" |
| 8 | flow-guided attention video transformer | **FGST（ICML 2022）**、FGT（视频补全，2022）、Flow-Guided Deformable VSR（2023） | **最接近的算法先例**：光流偏移采样 keys；ANN、键选择、需 flow 网络 |
| 9 | event camera ASIC direction of motion chip | sEMD/TDE 电路、SPECK 预处理层（极性合并/池化/旋转，**无方向场**）、CsPbBr3 方向神经元 | 方向检测在**前端传感器/独立分类**层；SPECK 预处理层无方向场 |
| 10 | anisotropic kernel event camera denoising | 未检索到 | 标注"未检索到" |
| 11 | Swin relative position bias / attention with bias hardware | **SWIN RPB（ICCV 2021）**、**FlashBias（NeurIPS 2025，arXiv:2505.12044）**、昇腾 aclnnMaskedSoftmaxWithRelPosBias 融合算子 | **"分数+偏置"是成熟模式**（静态学习型偏置）；FlashBias 是快速精确计算法（低秩压缩感知） |
| 12 | event camera temporal XOR binary direction | 未检索到 | "3×3 时域 XOR 梯度 argmax 方向场"同式硬件提取未检索到（技术本身简单，非核心主张） |
| 13 | event transformer direction selective motion gating | MAT（AAAI 2025，运动掩码引导稀疏注意力）、GS-SpikeFormer（2025，二值门控注意力）、SpikingVTG（NeurIPS 2025，SFG 门控） | 运动掩码/门控存在于 ANN/SNN 注意力；**均为 token/掩码粒度、学习型，非方向场分数偏移** |
| 14 | directional bias attention score SNN gate splitting | 未检索到 | "方向场驱动注意力分数偏移 + 唯一门按方向类分裂"未检索到 |
| — | WebFetch arXiv:2511.12961 精读 | OPCM 全文确认 | 方向先验 = 惯性传感器 3D 速度投影（外部线索）约束 CM 轨迹优化；软件、浮点、无网格、无注意力、无 stencil |

**未检索到的占位**（诚实声明）：
- "各向异性/方向感知 stencil 加速器"（硬件，方向场驱动的 lane 差异化计算）——未检索到（HPC stencil 加速器全部各向同性，专项查询 #1/#5）。
- "事件光流/SNN 注意力中方向场驱动**注意力分数偏移 ±Δ**（Q7 网格精确位移）"——未检索到（算法侧 FGST/MAT 是键采样/掩码粒度；硬件侧无）。
- "注意力加速器唯一门按运动方向分裂为多偏移类"（ident-K 1→3 类执行对象）——未检索到。

---

## 3. 逐篇对照表（工作 / 会议 / 对象 / D3 边界）

| 工作 | 会议/arXiv | 方向/门控对象 | D3 边界一句话 |
|---|---|---|---|
| **OPCM**（Karmokar & Beksi） | arXiv:2511.12961（2025-11） | 事件光流 contrast maximization 的方向先验：惯性 3D 速度投影方向图约束轨迹优化 | 域内最接近的"方向场先验"；但先验来自**外部惯性传感器**（非算子自含）、作用于**轨迹优化**（软件、浮点、无量化网格）；D3 方向场来自**算子内 K 时域 XOR 差分**、作用于**注意力分数域**（Q7 网格精确、硬件 2bit） |
| **FGST**（Lin et al.） | ICML 2022 | ANN 视频去模糊：光流偏移 (Δx,Δy) 采样 keys（稀疏注意力） | 最接近的"运动先验进注意力"；但机制是**键采样/删 token**（全局稀疏），需**光流网络**提供连续偏移，ANN 域；D3 是**固定 5-lane stencil 的分数偏移 ±Δ**（不删键、无 flow 网络、SNN 二值注意力） |
| **FlashBias** | NeurIPS 2025，arXiv:2505.12044 | "attention with bias"（SWIN RPB 等）的快速精确计算 | 证明"分数+偏置"模式不新（对抗性先例）；但 FlashBias 的偏置是**静态学习型**、计算方法是低秩压缩感知（通用 ANN）；D3 的偏置是**数据驱动（运动方向）2bit 偏移**、固定 Δ、Δ=0 逐位锚点、硬件档位选择器 |
| **SWIN relative position bias** | ICCV 2021 | 学习型静态逐偏移偏置加在分数上 | 同"分数+偏置"槽；RPB 与运动无关、需训练、每偏移一格一个学习参数；D3 无学习参数（Δ 固定）、随方向场逐像素变化、Δ=0 恒等档 |
| **Greatorex et al.** | CVPR 2026 Findings | SNN 突触门控的免训练事件光流估计 | 算法层模型方法（突触门控做 flow 回归）；无注意力分数、无加速器算子、无存储/执行对象 |
| **Pignari et al.** | IEEE ICDL 2025（Zenodo 15831646） | sEMD + 四基元方向群体（上/下/左/右）分类运动方向 | 方向分类是**任务输出**；D3 方向场是**注意力分数门控的输入**（执行对象），不改变任务输出语义 |
| **Paredes-Vallés et al.** | IEEE TPAMI 2019，arXiv:1807.10936 | 层级 SNN 无监督涌现方向/速度选择性（STDP + 延迟线） | 神经科学启发的选择性涌现（前馈检测器）；无注意力、无分数、无硬件合同 |
| **Brosch & Neumann** | Frontiers in Neuroscience 2015，DOI 10.3389/fnins.2015.00137 | TDE 方向选择性电路（含抑制侧翼） | 方向选择性电路本体是老技术（承认）；对象是**前端检测**，非分数域执行 |
| **MAT**（Xu et al.） | AAAI 2025 | 事件去模糊：运动掩码引导运动稀疏/感知注意力 + 门控 | ANN、掩码粒度 token 删减；D3 是 5-lane 分数偏移、SNN、方向场 2bit |
| **Bishop** | ISCA 2025，arXiv:2505.12281 | TTB 打包 + ECP 裁剪 | 禁止粘贴清单成员；无方向场、无分数偏移（D1 档案 §3 已核） |
| **ASTER** | arXiv:2511.06770 | PIM spiking transformer（层跳 + 时间步缩减） | 时间维裁剪；无方向场 |
| **Prosperity** | HPCA 2025，arXiv:2503.03379 | 产品稀疏（相同输入行复用内积） | 空间行级复用；无方向维 |
| **Zhang et al.** | CICC 2026，DOI 10.1109/CICC65509.2026.11509564 | 事件光流 U-Net 加速器：冗余推测 + 输入流相似跳过 | 同任务域硬件最近作；**无注意力、无分数、无方向场**（帧级相似跳过） |
| **RADiT** | DAC 2025 | ANN DiT 时间步相似度块级复用 | 时间步粒度近似；无方向 |
| **HPC stencil 加速器**（Wormhole/IWOCL/FPGA） | arXiv:2605.07599 等 | 各向同性 5-point stencil 数据流/访存优化 | 同一"stencil"词但无方向场、无注意力、无分数对象 |
| **BSA**（项目自身基础） | NeurIPS 2025 | 双极自注意力 + Shiftmax | 无方向偏置；D3 在其分数网格上做 ±Δ 位移 |
| **SDformerFlow**（项目基线） | arXiv:2409.04082 | 3D 窗口 (2,15,15) spiking 注意力 | Local5 的母基线；无方向场 |
| **GS-SpikeFormer / SpikingVTG / 3D spiking MoE** | 2025（MDPI/IJCAI/NeurIPS 等） | 二值门控注意力 / 显著门控 / MoE 条件计算 | 门控均为学习型、非方向场驱动、无存储/执行对象合同 |

**禁止粘贴清单专项核对**（PADE/SpAtten/Transitive Array/FuseMax/TeAAL/FLAT/Prosperity/Bishop）：逐项以 D1 档案 §3 的机制精读为据复核，**全部不含方向场/方向偏移分数/按方向分裂的 gate**——PADE 是位平面稀疏跳过、SpAtten 是 token 剪枝、Transitive Array 是位片行前缀复用、FuseMax/TeAAL 是 einsum 映射、FLAT 是数据流、Prosperity 是空间行产品稀疏、Bishop 是 TTB 打包。D3 不触碰其中任何机制成分。

---

## 4. 与最接近三篇的深边界（审稿人最可能引用）

### 4.1 vs OPCM（arXiv:2511.12961）——"方向场先验已是事件光流旧技术"杀法
1. **先验来源**：OPCM 的方向图来自**惯性传感器 3D 速度**投影（外部线索，需 IMU/位姿）；D3 方向场来自**注意力算子自身的 K 序列**（时域 XOR popcount → 3×3 空间差分 argmax，2bit），自包含、零外部输入、零学习。
2. **作用面**：OPCM 约束 CM 的**轨迹优化空间**（软件、浮点连续值）；D3 作用在**注意力分数域**（Q7 1/128 网格，K2 位移与量化 commute，8 档精确）。
3. **执行性质**：OPCM 是算法/训练侧方法（无存储对象、无执行对象）；D3 是加速器算子合同（方向场位图 450bit/窗 + 方向感知唯一门 K5），符合 docs/433 三件套门槛。
4. **目标**：OPCM 目标是提高光流**估计精度**；D3 目标是**硬件门质量的结构性再分配**（K4：对齐 lane winner 命中率 0%→91.2%），不声称改任务精度（AEE 对比只是回归检查，通过线 ±1%）。

### 4.2 vs FGST（ICML 2022）——"光流引导注意力已有"杀法
1. **机制**：FGST 用光流偏移**采样 keys**（Ω = {k at (i+Δx, j+Δy)}），是"改注意力拓扑"（键集合变化、删 token、全局稀疏）；D3 的 5-lane 拓扑**不变**（self+N+S+W+E 固定 stencil），只改**分数值**（±Δ），任何 token 都保留。
2. **代价结构**：FGST 需要**光流估计网络**（连续稠密偏移）；D3 的方向场是 4 个 2bit 比较器的硬件提取（实现说明 §1：无新乘加路径），Δ 是固定档位选择器。
3. **域**：FGST 是 ANN 视频去模糊（浮点注意力）；D3 是 SNN 二值注意力（Q7 网格、Shiftmax 门、K⊙gate 流水）。
4. **语义**：FGST 的流先验可错（像素级偏差需 FGSW 窗口变体兜底）；D3 的 K1 锚点（Δ=0 逐位恒等）保证任何方向场错误**不劣化**现网行为——可注入式安全属性是 FGST 没有的。

### 4.3 vs FlashBias / SWIN RPB（NeurIPS 2025 / ICCV 2021）——"分数+偏置是老模式"杀法
这是**最诚实的对抗**：D3 承认"分数加偏移"是注意力里的成熟槽位（RPB 是静态学习偏置，FlashBias 把这类计算快速精确化）。D3 的边界：
1. **偏置的动态性**：RPB/FlashBias 的偏置对每（位置偏移）是**固定的学习参数**（静态查表）；D3 的 ±Δ 是**数据驱动**的——随方向场逐像素取 {+Δ, 0, −Δ}，且方向场由算子内 K 的运动结构实时产生（非学习、非静态）。
2. **对象**：FlashBias 是"计算注意力"的**通用软件/数值方法**（低秩压缩感知）；D3 是**硬件存储对象 + 执行对象**（方向场位图、方向感知唯一门分裂），不改注意力矩阵计算法本身。
3. **量化语义**：D3 的 Δ 是 Q7 网格上的**档位位移**（K2：与量化 commute 逐位相等），不是浮点标量——硬件上是整数加法（+8/−8 档），这在 RPB/FlashBias 中不存在（它们都是浮点/学习型）。
4. **锚点**：K1 的 Δ=0 逐位恒等 + 注入式 warmup（Δ 0→8 档线性渐增）在"分数+偏置"文献中无对应——没有先例要求偏置具备"与无偏置版本逐位等价"的构造性回滚属性。

---

## 5. 与项目内部贡献的边界（防"自我重复"杀法）

| 内部对象 | 出处 | D3 边界 |
|---|---|---|
| Local5 unique-gate（source-owned 唯一门） | 现网合同（QS → FCSR → unique-gate → TCFM5，创新 3.1，docs/439） | unique-gate 是**目的地维度**对象：ident-K 目的地 1 个唯一门（71.6% 非静默）；D3 是**方向维分裂**：同一目的地 1→3 偏移类（K5 全分裂），门空间 G0={g(d)} → G3={g(d,c)}, c∈{self,+Δ,−Δ}，|G3|=3|G0|——执行对象结构变化，非换名（见 §6 杀法 3） |
| docs/439 硬件侧裁决"Local5 3.1 封顶、4.0=NO、再筛无新 exact 物化对象" | hw_autoresearch_nts07/docs/439 | 该裁决针对"再拆 RTL/再贴 Prosperity"（QAT、系数融合、dyadic、wavefront、Memo、2-wide、第三 stencil 均被否）；D3 不是第三 stencil（**不加 lane**），是方向场驱动的**分数偏移 + 唯一门分裂**——新算法合同在先（K1-K5 全过），正是 docs/439 留的"算法新线"路径 |
| H67/H82/H86 | 现网合同 | D3 纯追加（0 删除行、322 追加行），不动 Motion-XOR、不动 class-major/member-delta 目录 |
| BSA（双极注意力/Shiftmax） | 项目基础（NeurIPS 2025） | D3 在 BSA 的 Q7 分数网格上工作，方向场偏移是 BSA 之外的算子扩展；Δ=0 时 BSA 路径逐位不变 |
| D1（Motion T>2 时间商） | 同批草案 | D1 是时间维商 + RLE 广播；D3 是空间 stencil 方向场偏移——正交维度，互不触碰（D3 实现内 motion 恒 0） |

---

## 6. 审稿人预答辩（DATE 最可能三条杀法）

### 杀法 1："方向场门控是光流后处理的旧技术——方向选择性神经元/TDE/sEMD/方向先验估计几十年前就有了"

**反驳（引用 K1-K5 恒等式与检索记录）：**
1. **承认前件、拒绝结论**：方向选择性是事件视觉的老主题（TDE 电路 Frontiers 2015、sEMD 四基元群体 ICDL 2025、无监督选择性 TPAMI 2019、OPCM 方向先验 2511.12961、突触门控 CVPR'26F）——但检索记录（§2，#2/#3/#8/#13）显示所有这些工作的方向场对象是**前端检测器输出、光流轨迹优化约束、或任务输出分类**；**"方向场进入注意力分数域做偏移门控"在算法侧和硬件侧均未检索到占位**。
2. **对象与作用面不同**：TDE/sEMD 输出"运动方向"本身（分类语义）；D3 的方向场是**执行语义**——它不产生任何可见输出，只重新分配 Q7 分数网格上的门质量（K4：对齐 lane winner 命中率 91.2% vs 基线 0.0%）。这是"计算结构"的差异不是"实现细节"。
3. **与 OPCM 的关键差分**：OPCM 需要**外部 IMU/3D 速度**（架构依赖）、作用在浮点轨迹优化；D3 方向场由**算子内 K 的时域 XOR 差分**自产生（实现说明 §2：复用现网 `_binary_temporal_k_xor_popcount` 流水，4 个 2bit 比较器），硬件成本 <1% 存储增量、无外部传感器。
4. **K3 语义账**：方向场与运动轴对齐 76-85%（E/W 移动条），且与 C1 统计平面（q1[p+1]==k1[p] 79.36% 平面）正交可裁决（合同验证实验 3）——方向场是**算子内可证伪的机制**，不是借用外部光流后处理的现成场。

### 杀法 2："±Δ 偏移只是分数微调，创新不足——SWIN bias/FlashBias 早就做过分数加偏置"

**反驳（正面回应合同排序里"创新幅度是分数微调级"的质疑）：**
1. **区分"分数加偏置"与"方向场驱动的分数位移"**：SWIN RPB / FlashBias（NeurIPS 2025，arXiv:2505.12044）的偏置是**静态学习参数**（每位置偏移一格一个值、需训练）；D3 的 ±Δ 是**数据驱动、零学习参数、固定档位**（对齐 +Δ/正交 −Δ/self 0，由 2bit 方向场逐像素选择）。偏置的"动态性 × 量化精确性"组合在文献中未检索到。
2. **K2 是合同级事实**：Δ=1/16 在 Q7 1/128 网格上恰为 8 档，**与量化 commute 逐位相等**（200 组 clamp 外）——这不是浮点微调，是**整数域档位选择器**（实现说明：无新乘加路径，固定偏移加法）；"微调"在硬件上没有可迁移的对应物，D3 有（+8/−8 计数）。
3. **K4 证明偏移是"再分配"不是"微调"**：微调改变所有分数的幅度/尺度；D3 使对齐 lane 的量化分数 argmax 命中率从 **0.0% → 91.2%**（运动承载像素）——胜者被结构性改写，2^s 门动态范围（s∈[0,1]，max 2x）约束下再分配天然有界（K4 修正，诚实指标已钉死）。0%→91.2% 不是"微调"语言能描述的效应。
4. **K1 是"新算子合同"的充分证据**：Δ=0 与现网 Local5 **逐位一致**（10 组随机平面 torch.equal + forward 级逐位对比，实现 §5）——参数微调从不会与基线逐位等价；D3 定义了"算子系统 A3S ⊇ Local5"的**构造性包含关系**与**注入式安全迁移路径**（warmup 0→8 档，short 配置 1224 步 ≈ 1 epoch）。这是 docs/433"新算法算子合同"的严格形式：分数函数改变（方向偏移）+ 新存储对象（方向场位图）+ 新执行对象（方向感知唯一门）三件齐全，且三者均可在 Δ=0 处退化为现网——"微调"改变参数值，D3 改变**分数函数、存储、执行三个硬件对象**。
5. **诚实成本已记账**：raw16 广播 ×2、gate-plane +1 slot、450bit/窗方向场——总增量 <3% 位账（合同 §D3-4），即"以 <3% 位账买方向敏感执行对象"，交换结构本身是合同内容。

### 杀法 3："ident-K 分组就是 Local5 已有 unique-gate 的换名——你们只是把 1 组拆成 3 组"

**反驳：**
1. **unique-gate 是目的地维度对象**：现网 Local5 的 source-owned unique-gate 定义是"ident-K 目的地 1 个唯一门"（71.6% 非静默），门与**运动方向无关**——同一目的地无论 K 来自哪个方向都是同一门值。
2. **D3 分裂的是方向维**：同一 ident-K 目的地按方向场分裂为 3 个偏移类（K5：11250 个目的地**全部** 1→3，无例外），门空间 |G0|=n → |G3|=3n（折叠 self 为 2n）——**执行状态的离散化分裂**，不是命名变更。换名的判据是"对象集合不变"；D3 的门集合、gate-plane 槽位、广播数（×2）全部改变（合同 §D3 位账）。
3. **分裂的驱动者是方向场**：非 ident-K 目的地新增"方向场 2bit 查表决定 lane 分组"执行流水（实现说明 §1），这在 unique-gate 的现有流水（目的地身份判定）中不存在——新增了**第二判定维度**（方向），执行器从"1 判定 1 门"变为"2 判定 3 门"。
4. **分裂的语义可证伪**：K4 证明分裂后对齐类在运动像素上以 91.2% 概率成为胜者类——如果只是换名，门值分布不会改变；实测 winner 重分配是门质量的**行为变化**证据。
5. **内部裁决的自我修正**：docs/439 曾判"Local5 再筛无新 exact 物化对象"（针对再拆 RTL）；D3 回应的是"分数函数 + 方向场存储 + 分裂执行"三个**新物化对象**（K2/K5 给出 exact 物化的数学保证：网格精确、全分裂），不在该裁决覆盖的"QAT/系数融合/dyadic/wavefront/Memo/2-wide/第三 stencil"名单内。

### 附加杀法（备用）："方向场用 3×3 时域 XOR 梯度 argmax 提取，这个方向场本身没新意"
**反驳**：承认提取技术本身是简单的（XOR popcount + 空间差分 + argmax，专项查询 #12 也未检索到同式硬件实现，但技术复杂度确实低）——**但 D3 的新颖性主张不在方向场提取**，而在"方向场进入**注意力分数域**（K2 网格精确位移）并驱动**执行对象分裂**（K5）"的算子合同三件套；方向场只是一个 2bit 输入信号，正如 RPB 的 index 表本身也不新、新在它对注意力的作用方式。写作层将明确此叙事，避免被引导到"方向场检测器"战场。

---

## 7. 诚实结论：新颖性风险等级与补救方向

### 风险等级：**中（偏高，高于 D1）**

**降级风险因素**：
- "方向选择性/方向场"是事件视觉**几十年老主题**（TDE 电路、sEMD、四基元方向群体、无监督方向选择性、OPCM 方向先验、CVPR'26F 突触门控）——比 D1 的"时间冗余"主题拥挤度更高，审稿人第一印象"又一个方向场工作"是最大风险。
- "分数+偏置"模式有直接先例（SWIN RPB、FlashBias NeurIPS'25）——"±Δ 只是分数偏置"的指控有真实文本支撑。
- 项目自身合同排序点名"创新幅度是分数微调级"（DRAFTS 排序推荐 D1 > D3 > D2 的理由）；docs/439 硬件侧裁决 Local5 3.1 封顶——内部先例也构成降级压力。
- K4 修正本身说明门质量再分配受 2^s 动态范围约束有界——诚实指标（winner 命中率）与主流硬件指标（流量/能耗）语言不同，需要论文写作转化。

**支撑等级的因素**：
- "方向场驱动注意力分数偏移 ±Δ（Q7 网格精确、Δ=0 逐位锚点）+ 方向感知唯一门 1→3 类分裂"的**交集未检索到占位**（算法侧 FGST/MAT 是键采样/掩码粒度；硬件侧全空，专项查询 #1/#5/#14 均无）。
- K1（Δ=0 逐位恒等）与 K2（网格精确位移与量化 commute）构成**构造性安全锚点**——"可注入式算子合同"（从现网逐位等价出发渐增注入）在注意力加速器文献中未检索到先例。
- K5 的全分裂（11250/11250）与 K4 的 winner 重分配（0%→91.2%）是**执行对象改变**的硬证据——不是统计噪声，是 100% 确定性分裂。
- D3 满足 docs/433 三件套（新算法算子合同 + 新存储对象 + 新执行对象），且不动 Swin 架构、checkpoint 直接兼容（valid825 口径不变）——这是"分数微调"指控的反证：微调不需要新存储对象和新执行对象。
- 禁止粘贴清单八项全部不触碰方向场/分数偏移/方向门分裂。

### 为什么 Δ 偏移 + 方向感知唯一门构成"新算子合同"而不是"参数微调"（对合同排序质疑的正面回答）

1. **对象层级**：参数微调改变 W/b 的**数值**，算子、存储、执行全部不变；D3 同时改变**分数函数**（±Δ 方向偏移，K2 整数档位）、**存储对象**（方向场位图，450bit/窗新增文件）、**执行对象**（唯一门 1→3 类分裂，K5 全分裂、raw16 广播 ×2、2bit 查表流水）——三处都是"对象"级变更，不是数值级变更。
2. **K1 的区分力**：微调不存在"与基线逐位等价"的构造；D3 的 Δ=0 档是 torch.equal 级别的逐位恒等（算子级 + forward 级双路径验证），这在数学上定义了 A3S 与 Local5 的**包含关系**（A3S 是 Local5 的保真扩展），是"新算子"的形式化判据（类似"新函数 f_Δ 满足 f_0 = f_local5 且 f_Δ ≠ f_local5（Δ>0）"）。
3. **K4/K5 的行为证据**：winner 命中率 0%→91.2%（行为级改变）+ 11250/11250 全分裂（结构级改变）——微调的效应是渐变标量，D3 的效应是**离散类别的结构化重排**。
4. **验证协议**：合同验证实验 3（C1 统计平面 dump 裁决方向场与 q/k 相关收敛）是"机制归因"的裁决设计——如果 AEE 改善不能被方向场语义解释，D3 自我否决；微调类工作没有这种自证伪协议。
5. **诚实边界**：D3 不声称"大幅精度提升"（AEE 通过线只是 ±1% 回归检查）；其贡献是"**以 <3% 位账换取方向敏感的执行对象与门质量再分配**"——如果 DATE 审稿人要求性能增量叙事，D3 需要与 Local5 的 AEE 账并排呈现并承认收益主要在"门质量/执行对象"维度。这正是本文档 §7 补救方向 3 要处理的风险。

### 补救方向（按优先级）
1. **写作层（必做）**：related work 必须**主动消化**方向选择性文献（OPCM/FGST/TDE/sEMD/TPAMI'19/CVPR'26F/SWIN RPB/FlashBias），用 §3 对照表逐条差分"对象（分数域 vs 前端检测/轨迹优化/键采样）× 动态性（数据驱动 vs 学习型）× 量化（Q7 档位 vs 浮点）"；把 K1 写成"保真扩展定理"（Δ=0 逐位等价 → 注入式安全），这是 D3 独有的形式化资产。
2. **实验层（必做）**：合同验证实验 1（short，Δ warmup 0→8）与实验 3（C1 统计平面 dump 裁决）必须在提交前完成；补 Δ 敏感度扫描（Δ=2/4/8 档的 winner 命中率与 AEE 曲线），证明 8 档选择非过拟合。
3. **叙事层（建议）**：主打"**数据驱动、量化精确、可注入的方向敏感注意力执行**"（direction-driven, quantized-exact, injectable attention execution），避开"方向检测器"（前端战场）与"分数偏置"（FlashBias 战场）两个既有标签；位账（<3% 增量换方向敏感执行对象）与 K4（0%→91.2% winner 重分配）放同一张表。
4. **红线检查（完成）**：PADE/SpAtten/Transitive Array/FuseMax/TeAAL/FLAT/Prosperity/Bishop 逐项核对（§3），D3 不触碰其任何机制成分；未碰 GPU；未改任何现有文件；本档案只新建。

---

## 8. 引用清单（全部经检索确认或直接精读）

1. Karmokar, P. P. & Beksi, W. J.: Inertia-Informed Orientation Priors for Event-Based Optical Flow Estimation — arXiv:2511.12961（2025-11，WebFetch 精读确认：方向先验来自惯性 3D 速度、约束 CM 轨迹优化、无注意力/无 stencil）
2. Lin et al.: Flow-Guided Sparse Transformer for Video Deblurring（FGST）— ICML 2022（光流偏移采样 keys）
3. Wang et al.: FlashBias: Fast Computation of Attention with Bias — NeurIPS 2025，arXiv:2505.12044
4. Liu et al.: Swin Transformer（relative position bias 模式）— ICCV 2021
5. Greatorex et al.: Event-Based Optical Flow Leveraging Precise Event Timing — CVPR 2026 Findings（SNN 突触门控免训练光流）
6. Pignari et al.: Spiking motion direction through object motion sensitivity — IEEE ICDL 2025（Zenodo 15831646；sEMD + 四基元方向群体）
7. Paredes-Vallés et al.: Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation — IEEE TPAMI 2019，arXiv:1807.10936
8. On event-based optical flow detection（TDE 方向选择性电路）— Frontiers in Neuroscience 2015，DOI 10.3389/fnins.2015.00137（作者名未在检索摘要中确认）
9. Xu et al.: Motion-Adaptive Transformer for Event-Based Image Deblurring（MAT）— AAAI 2025
10. Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning — ISCA 2025，arXiv:2505.12281
11. ASTER: Attention-based Spiking Transformer Engine for Event-driven Reasoning — arXiv:2511.06770
12. Prosperity: Accelerating Spiking Neural Networks via Product Sparsity — HPCA 2025，arXiv:2503.03379
13. Zhang Tao et al.: A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation, Bit-Width-Aware Compression and Similarity Detection — CICC 2026，DOI 10.1109/CICC65509.2026.11509564（D1 档案已精读）
14. RADiT: Redundancy-Aware Diffusion Transformer Acceleration Leveraging Timestep Similarity — DAC 2025
15. ASNA-Flow: An Efficient Asynchronous Neuromorphic Accelerator for Real-Time Event-Based Optical Flow — IEEE 2025（11142472）
16. Stencil Computations on Tenstorrent Wormhole — arXiv:2605.07599；StencilStream GPU 后端（IWOCL 2026）；FPGA stencil DAQ（PMC8621947）——HPC 各向同性 stencil 对照
17. Wang et al.: Bipolar Self-Attention for Spiking Transformers（BSA）— NeurIPS 2025（项目基础）
18. SDformerFlow: Spatiotemporal swin spikeformer for event-based optical flow estimation — arXiv:2409.04082（项目基线）
19. 内部：docs/433（4.0 三件套门槛）、hw_autoresearch_nts07/docs/439（Local5 3.1 封顶裁决与算法新线接法）、D1_NOVELTY_DOSSIER_20260818.md（禁止粘贴清单机制核对依据）
20. 未检索到（诚实声明）：各向异性/方向感知 stencil 加速器；方向场驱动的注意力分数偏移；注意力唯一门按方向类分裂；SA-CNN 方向分类（查询失败）；同式"3×3 时域 XOR 梯度 argmax"硬件方向场
