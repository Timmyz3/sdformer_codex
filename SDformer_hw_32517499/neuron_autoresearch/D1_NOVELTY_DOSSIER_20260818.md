# D1 新颖性档案（Novelty Dossier）：Motion T>2 时间商 + 5-slot 商文件 + 时间维 run-length 广播

日期：2026-08-18。对象：D1 合同（CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md 的 D1 部分）与 D1 实现（D1_MOTION_T5_IMPLEMENTATION_20260818.md）。
方法：WebSearch 中英多查询（12 轮）+ 项目 docs/445、docs/433、docs/19、docs/142、docs/164 内部文献盘点 + 直接阅读 Zhang et al. 2026 CICC PDF。所有引用标注 arXiv 号或会议名；检索不到即标注"未检索到"，无虚构文献。

---

## 0. 一句话结论

**D1 的三件套（T=5 五元组时间商 + 规范融合式冻结分数 + 5-slot 可逆商文件 + 时间维 run-length 广播执行）在"分数粒度 × 时间维 × 无损构造语义"的交集上未检索到文献占位；但"时间维冗余利用"主题本身拥挤（Bishop/Prosperity/RADiT/SimVidT/CMC/Zhang CICC/ASTER/DSTSP/STAS/TC-SNN），包装与写作是主要风险。新颖性风险等级：中。**

---

## 1. D1 合同要素（对照审查用基线）

| 类别 | D1 要素 | 一句话 |
|---|---|---|
| 新算法算子合同 | T=5 五元组时间商 + 规范融合式 | `s_t = min(RNE16(64·o_t + sz_t + 16·m̄_t), 162)` 冻结为唯一规范（I1：拆解式在 RNE 平局奇偶翻转处全域 2.74% 差 1 档）；每位置 5 槽、4 条运动边 m_t=popcount(K_{t-1}⊕K_t) |
| 新存储对象 | 5-slot 时间商文件 | 每位置 5 个 (o_t, sz_t, m_t) 记录，记录→分数双向可逆（I5：物理域内反解 (s−m̄)→(o,r) 完全唯一，无 s%4==3，I2）；存储增量 <2%（净增 15bit/1125 tokens） |
| 新执行对象 | 时间维 run-length 广播执行 | eq 边（P(s_t==s_{t+1}|eq)=1，构造性精确，I6）下同一 Q7 类码沿 T 广播；eq=0.979 时每位置独立门 1.084/5（−78.3%）；T=4 对照 1.063/4（−73.4%） |
| 不动项 | Swin 窗口 (2,15,15) 不动 | T=5 分组在算子内跨窗完成，checkpoint 与 Motion ep35 锚点直接兼容（valid825 对比口径不变） |

---

## 2. 文献检索记录（2026-08-18，WebSearch 12 轮）

| # | 查询主题 | 关键命中 | 结果 |
|---|---|---|---|
| 1 | PADE 2512.14322 | HPCA 2026，predictor-free sparse attention，bit-serial QK + BUI-GF guard filtering + stage fusion | 无时间维商、无 RLE 广播；禁止粘贴清单成员，D1 不触碰其对象 |
| 2 | Transitive Array 2504.16339 | ISCA 2025，GEMM 结果复用：TransRow 子集偏序 Hasse 图前缀复用 | 空间维行内/行间复用，无时间维、无注意力分数对象、无 RLE/商文件 |
| 3 | FuseMax/TeAAL 2406.10491 / 2304.07931 | MICRO 2024 / MICRO 2023，einsum 映射与声明式建模 | 无时间商 |
| 4 | spiking transformer 加速器时间冗余 | Bishop（2505.12281）、ASTER（2511.06770）、HKUST 3D 阵列（JSSC 2025） | 见对照表 |
| 5 | 事件光流 transformer 时间窗 | SDformerFlow（2409.04082）、ASNA-Flow（IEEE 2025）、DCR-EFlow | SDformerFlow 时间窗 T=2；事件光流注意力 T>2 未见 |
| 6 | attention 加速器 RLE | SOFA（MICRO 2024）、HOLES/SPARK/INSPIRE、Blaze、Libra | **"run-length 用于注意力分数沿时间维广播"未检索到**；RLE 仅见于权重/位级编码 |
| 7 | FLAT | MICRO 2023 数据流优化（fusion/tiling）；**检索未能确认 arXiv 号**（项目 docs/142、164 有引用） | 无时间商 |
| 8 | Prosperity | HPCA 2025（2503.03379），产品稀疏：相同 spike 输入行→复用内积结果，TCAM 双缓冲 | 空间行级结果复用；见对照表 |
| 9 | 时间商/槽分解 | STAS、DSTSP（算法侧时间步剪枝）、SMixer | **"temporal quotient/槽位分解 + RNE 融合式"未检索到** |
| 10 | SNN 时间压缩/RLE | TC-SNN/PTC-SNN（1909.04757，Frontiers 2020，加权 spike 串压缩 16:1）、Intel 权重 RLE 专利（US 2018/0107919）、ISCAS 2009 零游程 RLE、SpikePack（ICCV 2025） | RLE 是编码老技术（表示性质）；D1 的广播是执行性质 |
| 11 | 视频 transformer 时间冗余加速 | CMC（ASPLOS 2024）、SimVidT（IEEE，arXiv 未检索到）、RADiT（DAC 2025）、EVA、ORBIS、LIPAR | 帧/token 粒度近似相似性跳过；见对照表 |
| 12 | 事件光流硬件 2025/2026 | Zhang et al. CICC 2026（项目 PDF 直接精读）、ASNA-Flow | Zhang 无注意力；见对照表 |

**未检索到的占位**（诚实声明）：
- 注意力/SNN 加速器中"逐位置跨时间槽分数商 + 分数冻结为唯一规范（含 RNE 平局语义）"——未检索到。
- "时间维 run-length 广播执行"（以分数相等为条件复用已冻结门，无损构造语义）——未检索到。
- 事件光流 SNN 注意力中 T>2 时间窗 + 每槽运动边商——未检索到（SDformerFlow 为 T=2，见 §5）。

---

## 3. 逐篇对照表（工作 / 会议 / 对象 / D1 边界）

| 工作 | 会议/arXiv | 复用/压缩对象 | D1 边界一句话 |
|---|---|---|---|
| **Bishop** | ISCA 2025，arXiv:2505.12281 | Token-Time Bundle：空间 token×时间点打包为调度容器，密度驱动路由 + 权重复用 + ECP 有界误差裁剪 Q/K/V + AND-accumulate 注意力阵列 | TTB 是打包/调度容器，裁剪是有误差界的近似；D1 是窗口内**逐位置 5 槽分数**的**无损构造广播**（eq 边 P(s_t==s_{t+1}|eq)=1），对象粒度与语义均不同 |
| **RADiT** | DAC 2025（IEEE 11133190） | ANN Diffusion Transformer 相邻 timestep 块级输出相似 → 重用整块结果（DTSM/CCU 阈值判定，有精度损失补偿） | 对象是 ANN 生成模型块级输出、判定近似；D1 对象是 SNN 二值注意力窗口内 Q7 分数类码、判定构造性精确、由运动边 XOR 驱动 |
| **Zhang et al.（事件光流加速器）** | CICC 2026，DOI 10.1109/CICC65509.2026.11509564 | Hybrid U-Net（SNN 编码 + ANN 解码）：MaxPool/ReLU 冗余操作推测（71.3%）、权重 MSB 符号位位宽感知压缩（84.7% EMA）、**输入流相似检测**（δ>10 时 >90% 相似 → 跳过帧级计算） | 同任务域最接近；但对象是卷积 U-Net 的输入流帧级跳过，**无注意力、无分数、无窗口**；D1 是注意力窗口内 5 槽分数商 + RLE 广播，且零精度损失（Zhang 的相似跳过含阈值近似） |
| **Prosperity** | HPCA 2025，arXiv:2503.03379 | 产品稀疏：同一 tile 内相同 spike 输入行 → 复用内积结果（TCAM 双缓冲产品稀疏表） | 家族相似（同操作数→同结果）但维度不同：空间行 vs 时间槽；对象不同：内积结果 vs 已冻结 Q7 分数+门；机制不同：TCAM 运行时查表 vs 构造性 eq 判定+广播 |
| **Transitive Array** | ISCA 2025，arXiv:2504.16339 | GEMM 位片行（TransRow）子集偏序前缀复用（Hasse 图 + Scoreboard，运行时 XOR 差位） | 复用对象是位片行部分积累（空间结构），注意力仅作为普通 GEMM；无时间维、无分数商文件、无 RLE（有 prefix/suffix bitmap，非时间维） |
| **PADE** | HPCA 2026，arXiv:2512.14322 | QK 位平面级稀疏跳过（BUI-GF 无预测器 guard filtering）+ 阶段融合 | 对象是位串行 QK 计算的稀疏跳过（删信息、近似）；D1 不删 token、分数冻结、无损广播；D1 不触碰其对象 |
| **SpAtten** | ISCA 2021，arXiv:2012.09852 | token/head 渐进剪枝 + top-k engine + progressive quantization | token 级删信息近似剪枝；D1 分数零近似、无剪枝预测器 |
| **FuseMax / TeAAL** | MICRO 2024，arXiv:2406.10491 / MICRO 2023，arXiv:2304.07931 | attention einsum 映射/声明式稀疏建模，序列长无关片上缓冲 | 数据流映射层；无时间商、无分数类码执行对象 |
| **FLAT** | MICRO 2023（arXiv 号检索未确认） | attention 数据流 fusion/tiling（softmax 片上融合） | 数据流优化；无时间维商 |
| **SimVidT / CMC** | IEEE / ASPLOS 2024 | ANN 视频 transformer 帧/token 时空相似性 → 跳过冗余计算（CMC 为 CODEC 辅助矩阵压缩） | 帧/token 粒度近似相似性消除；D1 是分数粒度无损广播，SNN 二值注意力专用 |
| **ASTER** | arXiv:2511.06770 | PIM spiking transformer：层跳过 + 时间步缩减（贝叶斯优化）+ 时空稀疏数据流 | 时间维裁剪在层/时间步粒度、推理时决策；无窗口内分数商 |
| **DSTSP / STAS** | 论文解读页/arXiv 列表（arXiv 号未检索完整） | 算法侧时间步剪枝（SIV 强度）+ 时空自适应计算（A-SSA 二维 token 剪枝） | 训练/推理算法（动态稀疏化、有精度影响）；D1 是硬件执行合同，无损 |
| **TC-SNN / PTC-SNN** | arXiv:1909.04757，Frontiers in Neuroscience 2020 | spike train 加权时间压缩（计数保持，16:1），用于 LSM/CNN 加速器 | 对象是 spike 串表示压缩（表示性质）；D1 对象是分数执行（执行性质），且是注意力窗口内 |
| **SpikePack** | ICCV 2025 | spike 序列压成整数（zip 形式）免解压计算 | spike 序列表示压缩；非注意力分数、非时间维广播 |
| **ASNA-Flow** | IEEE 2025（11142472） | 事件光流异步神经形态加速器：时间稀疏 + 事件驱动 + 空间局部性 | 无注意力（卷积/稀疏 SOP 计算）；对象不同 |
| **SDformerFlow（项目基线原论文）** | arXiv:2409.04082 | 3D 窗口 T×H×W 注意力，**T=2**（2×9×9 / 2×15×15）；spiking QK 线性注意力 | T>2 时间窗 + 每槽分数商不在其设计空间；D1 是其时间维直接扩展（I4：T=5 均匀边剖面 ≡ H67 T=2 锚点兼容） |

---

## 4. 与最接近三篇的深边界（审稿人最可能引用）

### 4.1 vs Bishop（ISCA 2025）——"SNN attention 时间打包已有"杀法
Bishop 的 TTB 把空间 token × 时间点的 spike 数据打包为 bundle 做密度路由、权重复用与 ECP 裁剪。D1 的边界：
1. **对象粒度**：TTB 的复用单位是 bundle（多 token × 多时间点）的调度与**权重**复用；D1 的复用单位是**同一空间位置跨 5 个时间槽的 Q7 分数类码**（每位置 5 个 (o_t,sz_t,m_t) 商记录）。
2. **语义**：Bishop ECP 是**有误差界的近似裁剪**（trims redundant Q/K/V with bounded error）；D1 的广播是**构造性无损**——eq 边由"同一 Q7 类码"定义，P(s_t==s_{t+1}|eq)=1 是恒等式不是统计近似（I6），无任何精度损失。
3. **时间维数**：Bishop 没有 T=5 五元组商、没有逐槽规范融合式分数、没有 4 条运动边 XOR 结构。D1 的分数含运动项 16·m̄_t（H67 兼容），Bishop 无运动结构。

### 4.2 vs RADiT（DAC 2025）——"时间维结果重用已有"杀法
RADiT 对 ANN DiT 相邻 denoising timestep 做块级特征相似检测（DTSM/CCU，阈值判定、精度损失补偿）后重用结果。D1 的边界：
1. **判定性质**：RADiT 的相似检测是**近似的**（需 Dynamic Threshold Scaling 补偿精度）；D1 的 eq 判定是**构造性精确**（同码→同分数，双向可逆 I5），无补偿模块。
2. **对象与驱动**：RADiT 无运动边；D1 的每槽分数由运动边 m̄_t=popcount(K_{t-1}⊕K_t) 显式驱动——时间商是**运动结构的编码**（4 条边、88.9% 时间边覆盖 I7），不是通用相似度。
3. **域**：ANN 扩散生成 vs SNN 事件光流注意力（Q7 163 档网格、Shiftmax 门、K⊙gate 流水）。

### 4.3 vs Zhang et al.（CICC 2026）——"事件光流加速器冗余跳过已有"杀法
这是任务域（事件光流硬件）内最近的工作，项目 docs 有 PDF。D1 的边界：
1. **对象**：Zhang 的 redundancy speculation 作用于 **MaxPool/ReLU 的非激活值**（71.3% 冗余操作）与**输入流帧级相似**（δ>10、>90% 相似 → 跳过）；网络是 Hybrid U-Net，**不含 attention、不含分数**。D1 作用于注意力窗口内每位置的 5 槽分数与门。
2. **粒度**：帧/层粒度 vs 窗口内逐位置分数粒度。
3. **语义**：Zhang 的 similarity detection 有阈值与漏检风险；D1 无损（I5/I6）。Zhang 不主张"时间商文件/广播执行/可逆重建"。
4. **运动边**：Zhang 的相似检测是输入流统计；D1 的运动边 m_t 是注意力算子内的**构造性运动量**（K 的时序 XOR popcount），与分数同域、可逆。

---

## 5. 与项目内部贡献的边界（防"自我重复"杀法）

| 内部对象 | 出处 | D1 边界 |
|---|---|---|
| H67 Motion（T=2 pair 商，RQTB） | 现网合同 | D1 是 T=2→T=5 的直接扩展：I4 证明 T=5 均匀边剖面 ≡ H67 T=2（m=2 配对翻转构造逐位一致），16·m 项兼容，checkpoint/锚点直接续训——是"扩展且向后兼容"，不是"重复五次"（见 §6 杀法 1） |
| H82 temporal quotient descriptor（class-major 目录） | docs/433（CONDITIONAL_PROFILE_GATE_SUPPORT_ONLY_NO_RTL） | H82 的 quotient 是**每 occupied class 一个 descriptor**（class_id+k_mask+pair_last，class-stationary 目录）；D1 的 5-slot 商文件是**逐位置 5 槽 (o_t,sz_t,m_t) 记录 + 分数冻结 + 时间维广播执行**——对象（位置×槽 vs class）、结构（可逆重建 vs 目录差分）、执行（广播 vs gather）三不同。docs/433 明确 H82 quotient 不是独立贡献，D1 正是把"时间商"升级为独立 4.0 算子合同的候选 |
| Local5 source-owned unique-gate | 现网合同 | D1 不动 Local5 任何路径（0 删除行、纯追加）；Local5 是空间唯一门，D1 是时间维广播门——正交维度 |

---

## 6. 审稿人预答辩（DATE 最可能三条杀法）

### 杀法 1："T>2 只是把 T=2 的商重复五次，增量式扩展不构成新算子合同"

**反驳（引用 D1 恒等式与实测）：**
1. **不是五次 pair，是 5 槽共享的运动商结构**：T=5 每槽分数由规范融合式 `s_t=RNE16(64·o_t+sz_t+16·m̄_t)` 单独决定，且 4 条运动边 m_t 是**窗口内共享**的构造性运动量（I7：时间边覆盖 55.6%→88.9%，可见边从 5/9 增至 8/9，新增 3 条边的信息是 T=2 结构里不存在的）。
2. **I1 是硬合同发现**：拆解式 RNE16(64o+sz)+m̄ 与融合式在 RNE 平局商奇偶翻转处全域 2.74% 差 1 档——这证明"时间扩展"不是机械重复，分数函数本身必须冻结为唯一规范（硬件与部署同式）。任何"重复五次"的实现都会在 2.74% 的槽位上产生不同的分数值。
3. **I2/I5 给出唯一性定理**：槽位分解 s_t=4·o_t+r_t 在物理域（容斥界 max(0,q+k−32)≤o≤min(q,k)）内 r∈{0,1,2} 唯一、s%4==3 不存在；5-slot 商文件记录→分数**双向可逆**（0.00% 退化）。这是 T=2 商文件不具备的复合结构（5 记录 + 4 边 + 交叉可逆性）。
4. **执行对象是新的**：T=5 的 RLE 广播（1.084/5，−78.3%）与 T=2 的逐 pair 执行在数据流上不同类——广播执行器沿 T 复用同一 Q7 类码，每位置独立门数从 5 降到 1.084（I6），T=4 对照 1.063/4（−73.4%）。
5. **锚点兼容是优点不是平庸**：I4 证明 T=5≡H67（T=2）的均匀边剖面逐位一致，valid825 对比口径不变——扩展与兼容同时成立，这正是"不动 Swin 架构"的合同约束（docs/433 门槛：新算法算子合同 + 改硬件存储/执行对象，D1 三件全满足）。

### 杀法 2："时间维 run-length 是 image/video codec 的老技术搬到注意力"

**反驳：**
1. **承认 RLE 编码是旧技术，但 D1 的贡献不是编码**。D1 的新执行对象是**广播执行器**（执行语义：同一 Q7 类码沿 T 只发射一次门），省的是**门/exp-add 流量**（−78.3%），不是存储位；存储对象是 5-slot 商文件（可逆重建，I5），其目的不是压缩而是**分数冻结 + 重建唯一性**。
2. **与 codec RLE 的对象不同**：codec RLE 编码的是像素/零游程/权重/位片（TC-SNN 1909.04757、Intel 专利、ISCAS 2009 均为表示压缩）；D1 编码的是**同一空间位置跨 5 个时间槽的注意力分数序列**——分数是 Q7 163 档网格上的融合式值，广播条件由构造保证（eq 边 P(s_t==s_{t+1}|eq)=1，I6），是执行调度性质而非存储表示性质。
3. **与视频 transformer 时间冗余加速器（CMC/SimVidT/RADiT）的边界**：这些在帧/token 粒度做**近似相似检测**（需精度补偿）；D1 在分数粒度做**无损构造广播**，无检测器、无补偿——比它们"更精确、更细粒度、更底层"（窗口内逐位置）。
4. **文献对照表的直接证据**：§3 表中 17 项均无"注意力分数沿时间维的 RLE 广播执行"（专项查询 #6 确认：attention 加速器 RLE 仅见于权重/位级编码，未检索到分数时间维 RLE）。

### 杀法 3："eq=0.979 是数据特性，不是合同贡献；换个数据集数字就没了"

**反驳：**
1. **合同贡献是"eq 边 → 广播"机制，不是 0.979 这个数字**。I6 恒等式给出：eq 边是构造性精确（P(s_t==s_{t+1}|eq)=1），广播执行在任意 eq 率下**语义无损**——机制贡献与 eq 率无关。
2. **敏感度内嵌在合同里**：T=4 → 1.063/4（−73.4%）、T=5 → 1.084/5（−78.3%）两组账已在合同内（验证 I6）；即使实测 eq 率显著低于 0.979，门流量下降仍保持在 ~73% 量级（T=4/T=5 两个窗口都成立）。0.979 只是 H82 rank-1 锚点的**估计值**，不是机制的充分条件。
3. **可证伪性设计**：合同验证实验 3（运动边分布 dump 裁决）直接以"rank-1 带 Q 标签 dump 实测 T=5 槽位分数 RLE 广播率 vs Bernoulli 独立假设界（I6b）"为裁决标准——如果实测广播率不显著超过独立假设，D1 的执行账就自我否决。这符合 DATE 4.0"可证伪"门槛，且是文献中罕见的（PADE 依赖预测器精度、RADiT 依赖阈值补偿，都无类似的自证伪裁决设计）。
4. **与 PADE/SpAtten 的关键差异**：那些需要**预测器**（运行时稀疏猜测）才能生效；D1 无预测器——eq 边由同一类码的构造恒等式给出，执行器只需"run-length 判定 + 广播"两条流水（合同 §4）。

### 附加杀法（备用）："相同输入→相同结果的复用（Prosperity 产品稀疏）已经覆盖了 D1"
**反驳**：Prosperity（HPCA 2025）复用的是**空间维**相同 spike 输入行产生的**内积结果**（TCAM 查表、tile 内行级）；D1 复用的是**时间维**同一位置的**已冻结 Q7 分数与门**（含运动边商），且复用判定是构造性（无 TCAM 运行时匹配，见 docs/209 对 Prosperity 的既有边界裁决："不做运行时 matcher；复用类由 gate 与固定 relation 直接给出"）。维度、对象、判定机制三不同。若审稿人继续追问"时间维版本是否平凡"——用 §6 杀法 1 的 5 条回应。

---

## 7. 诚实结论：新颖性风险等级与补救方向

### 风险等级：**中**

**降级风险因素**：
- "时间维冗余利用"是 2024-2026 的高活跃主题（Bishop ISCA'25、RADiT DAC'25、SimVidT、CMC ASPLOS'24、Zhang CICC'26、ASTER、DSTSP/STAS、TC-SNN），审稿人第一印象是"又一个时间冗余工作"。
- "RLE/时间压缩"的编码老技术标签容易被贴（虽然 D1 的对象是执行不是编码）。
- 事件光流硬件域已有 Zhang et al. CICC'26 的"输入相似性跳过"先例（虽然对象不同）。

**支撑等级的因素**：
- "窗口内逐位置分数商 × 时间维 × 无损构造广播"的交集**未检索到占位**（专项查询 #6/#9 均空）。
- I1 的 RNE 平局规范发现（2.74% 差 1 档）与 I2/I5 的唯一性定理是文献中未见的形式化——"把分数函数本身（含舍入平局语义）钉成硬件合同"没有先例。
- 运动边驱动的 T>2 分数商在 SNN 注意力加速器中未见（Bishop/Prosperity/Transitive Array 均无运动结构）。
- T>2 时间窗注意力在事件光流 SNN 域未见（SDformerFlow 为 T=2，检索确认）。

### 补救方向（按优先级）
1. **写作层（必做）**：论文 related work 用 §3 对照表把 D1 与 Bishop/Prosperity/RADiT/Zhang 四家逐条差分（对象粒度：窗口内分数 vs 行/块/帧/束；语义：无损构造 vs 近似/有界误差；维度：时间槽 vs 空间行/时间步）。把"分数冻结为唯一规范（I1）"写成算法-硬件合同定理（含 RNE 平局语义），这是形式化资产。
2. **实验层（必做）**：验证实验 3（运动边分布 dump 裁决，I6b vs Bernoulli 界）必须在提交前完成——它是"eq=0.979 是数据特性还是机制"的唯一裁决。补 eq 率敏感度扫描（报告 T=4/T=5 双窗、不同 eq 率下的门流量，弱化对单点 0.979 的依赖）。
3. **叙事层（建议）**：主打"无损时间商执行"（lossless temporal-quotient execution）而非"时间冗余压缩"，与所有近似/裁剪/打包工作拉开语义距离；存储增量 <2% 与门流量 −78.3% 的账（I6）放在同一张表，强调"以 2% 存储换 78% 执行流量"的交换结构是新的。
4. **红线检查（完成）**：PADE/SpAtten/Transitive Array/FuseMax/TeAAL/FLAT/Prosperity/Bishop 逐项核对（§3），D1 不触碰任何禁止粘贴对象的机制成分——D1 无预测器、无 token 剪枝、无位串行 GEMM、无 TCAM 匹配、无近似裁剪、无 bundle 调度。

---

## 8. 引用清单（全部经检索确认或直接精读）

1. Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning — ISCA 2025，arXiv:2505.12281
2. PADE: A Predictor-Free Sparse Attention Accelerator via Unified Execution and Stage Fusion — HPCA 2026，arXiv:2512.14322
3. Transitive Array: An Efficient GEMM Accelerator with Result Reuse — ISCA 2025，arXiv:2504.16339（ar5iv HTML 已精读机制细节）
4. FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design — MICRO 2024，arXiv:2406.10491
5. TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators — MICRO 2023，arXiv:2304.07931
6. SpAtten: 动态 token/head pruning — ISCA 2021，arXiv:2012.09852
7. Prosperity: Accelerating Spiking Neural Networks via Product Sparsity — HPCA 2025，arXiv:2503.03379
8. Zhang Tao et al.: A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation, Bit-Width-Aware Compression and Similarity Detection — CICC 2026，DOI 10.1109/CICC65509.2026.11509564（项目本地 PDF 直接精读）
9. FireFly-T: High-Throughput Sparsity Exploitation for Spiking Transformer Acceleration with Dual-Engine Overlay Architecture — IEEE TC 2026 / arXiv:2505.12771
10. ASTER: Attention-based Spiking Transformer Engine for Event-driven Reasoning — arXiv:2511.06770
11. SDformerFlow: Spatiotemporal swin spikeformer for event-based optical flow estimation — arXiv:2409.04082（D1 基线原论文，T=2 窗口确认）
12. TC-SNN / PTC-SNN: Boosting Throughput and Efficiency of Hardware Spiking Neural Accelerators using Time Compression — Frontiers in Neuroscience 2020，arXiv:1909.04757
13. RADiT: Redundancy-Aware Diffusion Transformer Acceleration Leveraging Timestep Similarity — DAC 2025（IEEE 11133190）
14. CMC: Video Transformer Acceleration via CODEC Assisted Matrix Condensing — ASPLOS 2024
15. SpikePack: Enhanced Information Flow in Spiking Neural Networks with High Hardware Compatibility — ICCV 2025
16. ASNA-Flow — IEEE（2025-08 在线，IEEE 11142472）
17. SimVidT（IEEE，arXiv 号未检索到）；FLAT（MICRO 2023 数据流优化，arXiv 号未检索确认）；DSTSP/STAS（arXiv 号未检索完整）；HKUST 3D-array SNN accelerator（JSSC 2025）
18. 内部：docs/433、docs/445、docs/19（NTS11 文献调研）、docs/142、docs/164、docs/209（Prosperity/Bishop/FLAT 边界既有裁决）
