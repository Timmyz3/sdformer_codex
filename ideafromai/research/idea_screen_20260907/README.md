# 协同对象候选筛查台账（不选赢家）

日期：2026-09-07。合同来自 grill-me。本文件是 **proposal**，不是 finding。

**调研包（给 Codex，本目录不负责实现）：**

- [`CODEX_HANDOFF.md`](CODEX_HANDOFF.md) — 总交接  
- [`CODESIGN_OBJECTS.md`](CODESIGN_OBJECTS.md) — 8 个协同对象  
- [`MEASUREMENT_CONTRACTS.md`](MEASUREMENT_CONTRACTS.md) — M0–M8  
- [`SOURCE_LEDGER.md`](SOURCE_LEDGER.md) — 文献  
- [`T1_T4_ep35.json`](T1_T4_ep35.json) — 窗粒度 dirty  
- [`orchestra/`](orchestra/) — Orchestra `brainstorming-research-ideas` Phase 1–3（20 条发散 → 5 条短名单 → 流程赢家 I001）  
- [`kdense/`](kdense/) — K-Dense `scientific-brainstorming` 10 步台账 + CLI（`decision` 为 null；D001 待 P01 签字）

**S1 记分板：** [`SCOREBOARD.md`](SCOREBOARD.md) · [`scoreboard.json`](scoreboard.json) · [`screen_all.py`](screen_all.py)  
QK 普查：ep35 路径 100 样本 × 12 块 packed Q/K（1200 npz），**不是** ep34 封存表。

**相对上一版合同的三处改动：**

1. 不要求 1RW 友好；双口 SRAM 编译器在 TSMC 28 上可用。
2. 不先晋级唯一标题对象；先全部筛出来、套到现网上看效果。
3. 不用 kill list 淘汰；旧 C1/C2/TSBG/PAFT 等可以在新条件下重测。

精度硬门仍有效：valid825 AEE **≤ 1.259**，且须优于 SDformerFlow PSN **1.5848**。任务锁死 SNN Transformer 事件光流；注意力必须仍是脉冲驱动 QK（或等价无 softmax）。

文献窗：新候选 2024-01 至今；经典工作只作对照，不自动否决。检索日 2026-09-07，有界搜索，**不是**「从未有人做过」。

---

## 套上去的顺序（先做完再谈标题）

对每个还活着的候选，按同一工作负载记账，禁止把局部倍率相乘。

| 阶 | 做什么 | 过门才进下一阶 |
|---|---|---|
| S0 身份 | 一句话 + 算法面/电路面 + 对照先验 | 能写两句 pitch |
| S1 CPU/捕获 | 用 ep34 已有 descriptor 或 overlay 前向，数加法/访存/可跳过 lane | 有可复核收据 |
| S2 子集 AEE | 10 帧或单序列；若改图则短训 | ΔAEE 方向清楚，不要求 valid825 |
| S3 全量 | A800 valid825 | AEE ≤ 1.259 且优于 1.5848 |
| S4 岛 RTL | Verilator 功能；需要时才 DC/PT | 同资源时间线相对「抄全先验」不差 |

换注意力/神经元：从 ep34 微调。换骨干/窗/T：可从 SDformerFlow 或随机初始化。

---

## 簇 A — 旧机制重测（以前杀掉，条件已变）

算法可动、端口不锁 1RW、允许重训。先前失败可能不再成立。

| ID | 一句话 | 算法面 | 电路面 | 先前失败 | 这次套什么 |
|---|---|---|---|---|---|
| A1 | 把 Prosperity 完整链抄到 H67，φ=θW | 无，或 PAFT 训稀疏 | C1 父林 + 1RW/双口 父缓存 | 判为「别人做过」；完整抄全未当标题 | 完整层 3000×6912×768 对照官方 run_fc；双口是否抬吞吐 |
| A2 | 父值寄存器原位提升 | 无 | C1 增量 | +1.58% 加法，已停 | 换布局/双口后再数，不换公式 |
| A3 | Phi PAFT：Hamming 正则换更高稀疏 | 训法 | C1 pattern 命中 | valid825 ΔAEE=+0.029 超旧 0.02 门 | 新精度门是 1.259；重跑 running-BN |
| A4 | TSBG 权重行广播 | 可训成更多同行复用 | C2 前端 | 创新不足；B8 未当标题 | 新稀疏结构下 B8 周期是否 ≥1.15× |
| A5 | 有损 S2 块跳过 | 训/ε | C2/patch 前端 | 多数层上限 <1.15× | 只对 FC1/patch 重测上限，不过门停 |
| A6 | C2 时间共享部分和 | θ 幅值保留 | 有界槽 RTL 已有 | 6.5/10，加法减量 20.8%/42.2% 非加速比 | 接 bank 返回 + 持久 Y，同资源 vs 直接 FTP |
| A7 | C2 每 bank 一种 10-bit 模式 | 校准位置冻结模式 | 无字典归约 | 加法 9.3%/15.1%；bank 映射不是捕获地址 | 先改捕获布局或改映射，再谈周期 |
| A8 | 晚知 BN 分界的 θ 确认包 | 无 | encoder/verifier/group-commit RTL 已有 | 6/10；缺真实 BN 区间 | 接真实 BN/PSN 数，看恢复率 |
| A9 | 运动残差默认状态 | 要新表示、要训 | 无 RTL | 仅有理数例子 | 先 DSEC 小训，AEE 不动再谈硬件 |

---

## 簇 B — 注意力岛（脉冲 QK 锁死，公式可换）

旧周期信封里注意力约 0.6%。组件级 TCAS-II 可以只讲岛；系统摘要仍要重测份额。

| ID | 一句话 | 算法面 | 电路面 | 近年/对照 | 套法 |
|---|---|---|---|---|---|
| B1 | Motion-XOR 三 popcount 打分 ALU | 保持 H67，或把 α 做成可学 | 组合 pop + Q7 | FireFly-T 只做 QK AND-PopCount；α-XNOR 无时间对端 | ep34 重测 dirty/K-zero；RTL 叶 vs 软件 Q7 |
| B2 | 时间对端 dirty-lane：静默则复用 t0 分数 | 无损跳过，或训「何时可跳」 | 与 B1 同岛 | DLSS/DeltaCNN 是特征图跳过，不是分数 lane | 必须 ep34 行级 dirty，防 97% token 相等被整行分母拖死 |
| B3 | K=0 ⇒ 跳打分且不取 V（K 当 V） | 无损 | 与 B1 同岛 | 不是 empty-tile skip | ep34 逐 token K 零率 |
| B4 | 把 SDSA/QKFormer 线性 QK 换进 H67 窗 | 换注意力，从 ep34 微调 | 线性 QK 引擎，可比 Motion-XOR 更简 | QKFormer 2024；SDformerFlow 公开是 SDSA 不是 Motion-XOR | overlay 换核，valid 子集 AEE |
| B5 | STSA：时空联合脉冲注意力 | 换核 | 时间维与窗维同一 pop/加阵列 | Spiking ST-former 2025；STAtten | T_w=2 是否太短；可试 T_w>2 重训 |
| B6 | SLI 邻域脉冲交互 + SSA 互补融合 | 加轻量 depthwise 通路 | 小卷积引擎旁路注意力 | arXiv:2608.19238，挂在 QKFormer 上涨点 | 当协同：SSA 稀疏时 SLI 补局部运动 |
| B7 | SQKFormer：通道增强 QK + 膜电位自适应 BN | 换 BN/注意力 | 膜电位 BN 硬件与 C3 交界 | Neurocomputing 2026 | 与现有 PSN/ATLIF BN 冲突要先对齐 |
| B8 | LRF-SSA 神经动力学自注意力 | 换 SSA 内部动力学 | 仍无 softmax 的脉冲注意力 | ICLR 2026，挂 Spikformer/QKFormer/SDT-v3 | 插件式，适合 ep34 微调 |
| B9 | FireFly-T 二值引擎搬到光流窗注意力 | 量化到 4-bit 脉冲 | LUT/ASIC 的 AND-PopCount + 稀疏解码 | FireFly-T TC 2026，KV260/Zynq | 分数是 Motion-XOR 不是 QK^T，要对齐三项 pop |
| B10 | 双稀疏注意力加速器（Q 与 K 都是 spike） | 保持双 spike | 双输入 spike 编码跳零 | Spike-driven Transformer 加速器 2025 | 现网 Q/K 已是 ATLIF 发放，可直接套 |

SpikePool 用 max-pool 换 SSA：和「必须脉冲驱动注意力」冲突，**不进标题池**，可作消融对照。

---

## 簇 C — 神经元 / T / 前端

| ID | 一句话 | 算法面 | 电路面 | 套法 |
|---|---|---|---|---|
| C1 | 混合视野 ATLIF：膜留在岛内，对外 1-bit；T=10 与 T=2 两套调度 | 推理合同，可微调 | 神经元岛升级 C3 | 不宣称 first mixed-T ASIC |
| C2 | 学出来的 T 压缩 / 动态 T | 训动态 T | 可变 T 控制 | STISA TSC；光流边界可能怕短 T |
| C3 | Spiking Patches 代替体素再进 Swin | 前端 tokenization | 异步 token 队列 | arXiv:2510.26614；改输入身份，要从 SDformerFlow 起重训 |
| C4 | 把 ATLIF θ 折进 W vs 显式连续载荷 | 数值合同 | 下游只加减 vs 带宽换精度 | 冻结捕获是 z=θg；改合同必须 AEE 过门 |

ST-FlowNet（ConvGRU）和 Spike-GRU 光流不是 Transformer，**不进标题池**，可当 DSEC 精度对照。

---

## 簇 D — 卷积 / FFN / 解码（份额大）

历史信封：Conv≈42%，FC1+FC2≈26%，ATLIF≈21%，注意力≈0.6%，解码 ConvTranspose≈22%。系统加速更可能在这里，标题创新压力也更大。

| ID | 一句话 | 算法面 | 电路面 | 套法 |
|---|---|---|---|---|
| D1 | LoAS 完整 T 并行 inner-join，T=2 与 T=10 分纤维 | 发放打包 | C2 执行织物 | 官方工件无完整 ASIC RTL；抄模型再改混合 T |
| D2 | RSR++ 平坦/递归共享归约 | 无或轻 | C2 | 已有 FC2 机会；接真实 Y/BN2 |
| D3 | 解码器 ConvTranspose 精确稀疏岛 | 无 | 新岛 | 旧诊断 dense→K8 有倍率，缺完整 decoder 表 |
| D4 | 结构化权重稀疏 + 硬件跳过（FireFly-S 双端稀疏） | 剪枝+4bit 训 | 双端稀疏引擎 | 旧 N:M 审计 FP32 无精确零块，必须重训 |
| D5 | 解码/Conv 与 C1 共用同一套父林引擎 | 无 | 单一 product/add 织物多适配器 | 协同故事：一个对象服务多算子 |

---

## 簇 E — 近年网络替换（仍是 SNN Transformer 光流）

| ID | 一句话 | 起步权重 | 注意 |
|---|---|---|---|
| E1 | 公开 SDSA 换回 / 与 Motion-XOR 并列消融 | ep34 微调或 SDformerFlow | 硬件变简单，AEE 可能回退 |
| E2 | QKFormer 块替换 stage2（6 块最深） | ep34 微调 | 只换最贵 stage，省训力 |
| E3 | Spike-driven Transformer v3 块 | 视改动量 | 保持无乘法注意力 |
| E4 | 窗 15×15 / T_w=2 改成硬件更整齐的 8×8 或 T_w=4 | 换窗/T → 可从 SDformerFlow 重训 | 直接改几何，A800 全量才有资格引用 |

---

## 建议的第一批套法（不是赢家，是最便宜的信息）

不做标题预选。按 **信息/机时** 排序：

1. **B3 + B2 + B1 统计**：只读 ep34 捕获，不训、不 RTL。输出 K 零率、dirty 行率、三项 pop 直方图。  
2. **A6 / A8**：已有 RTL，接真实 BN 或 Y 端口，看恢复率和同资源周期。  
3. **A3 PAFT**：按新 AEE 门 1.259 重评 running-BN valid825（若收据还在，先复算再决定是否重训）。  
4. **B4 最小 overlay**：stage2 一块 SDSA 或线性 QK vs Motion-XOR，子集 AEE。  
5. **D1/D2 CPU**：完整 T FTP vs 现有共享和，同一 FC2 捕获。  
6. **E2/E4**：只有 1–5 显示系统份额或 AEE 有空间时才占 A800。

---

## 评价准则（打分辅助，不自动决出）

| 准则 | 方向 | 证据 |
|---|---|---|
| 映射到脉冲 QK 光流 | 必须 | 结构是否改 softmax |
| 精度 | AEE ≤ 1.259 且 < 1.5848 | valid825 |
| 同资源性能 | 不差于抄全最强先验 | 完整时间线，含排空反压 |
| 可套性 | 越快越好 | S1 能否本周做完 |
| 两面是否同对象 | 算法改动能被同一电路吃掉 | 不是两篇订在一起 |

文献状态一律 `search-incomplete`：2024–2026 arXiv/IEEE 抽样，未穷尽。

本轮检索触及：ST-FlowNet 2503.10195；SpikePool 2510.12102；FireFly-T 2505.12771 / IEEE TC 2026；SQKformer 2026；SLI+ACF 2608.19238；LRF-SSA ICLR 2026 2603.19290；Spiking Patches 2510.26614；Spike-driven Transformer 稀疏加速器 2501.07825；ASTER PIM 2511.06770（模拟 CIM，不当本课题 PPA 主线）。

---

## 引用

筛查流程使用了 Scientific Agent Skills 的 scientific-brainstorming 程序：

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
