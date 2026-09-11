# 下一任 Agent 接手提示词（中文，目标仅 TCAS-II）

把「--- 以下复制 ---」到文末整段贴进**新开**的会话。不要 `codex exec resume` 旧会话 `01a01043-f192-78e0-91bf-1bae75124f18`（jsonl 约 554 MB，remote compact 反复断流）。不要往那条 Codex 线程里塞指令。

--- 以下复制 ---

你接手的是华中科技大学硕士课题的**电路短文**线。现在**只投 IEEE Transactions on Circuits and Systems II: Express Briefs（TCAS-II）**，**不再投 ISCAS 2027**。IEEE 声明「has not been submitted elsewhere」，ISCAS 稿即使写好了也**禁止并行投**。不要再把 ISCAS 四页双盲当作目标格式，不要提 DATE 2027 作为备选会。

先把本提示词读完，再读磁盘上的 idea 目录，再动任何文件。数字以仓库已封存证据为准；没有 VCS+DC/PT+Formality 同工作负载闭环的，不得写成 RTL 加速比。禁止发明流片、整网 FPS、把组件倍率相乘。

# 0. 你必须先读的路径（按顺序）

1. 本接手说明（若在仓库里）：`/home/zhumd/work/sdformer_codex/ideafromai/HANDOFF_NEXT_AGENT_20260905.md`
2. 多 AI 想法总入口：`/home/zhumd/work/sdformer_codex/ideafromai/README.md`、`INDEX.json`、`MANIFEST.txt`
3. Grok 4.6 新机制调研（**冻结身份是二值 ATLIF**）：  
   `/home/zhumd/work/sdformer_codex/ideafromai/research/grok46_20260905/`  
   必读：`00_READ_THIS_FIRST.md`、`01_kill_list.md`、`02_ranked_mechanisms.md`、`03_algorithm_native_motionxor_atlif.md`、`06_two_workflow_conflict.md`、`07_next_stats_rtl_gates.md`
4. Codex 独立假说（**研究记录，未授权开实验**）：  
   `/home/zhumd/work/sdformer_codex/ideafromai/codex_independent_20260905/`
5. Grok Bot C1*/C2* 包（**假设 int8 ATLIF 载荷，与冻结捕获冲突**）：  
   `/home/zhumd/work/sdformer_codex/ideafromai/research/04_SYNTHESIS_C1_C2_REMAKE.md`  
   `codex_cards/CARD_A_OP_STW.md`、`CARD_B_HBG_RP.md`  
   `contracts/ATLIF_contract_r1_grokbot.md`
6. 现有 TCAS-II 稿（读，且默认**故事已过时**：仍在卖 C1+C2，用户已判定创新不足）：  
   `/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/paper/tcasii/main.tex`  
   同目录 `README.md`、`COVER.md`
7. 算法/硬件叙事（当前最完整中文）：`/home/zhumd/work/KAITI_report/技术工程报告_脉冲Transformer软硬件协同_20260903.md`  
8. 算法源码索引（**数字过期，只用来找文件**）：`sdformer_codex/SDformer/docs/H67_MOTION_ALGORITHM_FILE_INDEX.md`

想法目录的**唯一规范路径**是 `/home/zhumd/work/sdformer_codex/ideafromai/`。  
过期副本：`/home/zhumd/work/hw_autoresearch_nts07/ideasfromai/` 以及 git 树里 `SDformer/hw_autoresearch_nts07/ideasfromai/`（README 已写 MOVED）。  
Grok Bot 隔离 RTL 骨架：`/home/zhumd/work/sdformer_c1c2star_grokbot/`，用户没点头不要合并进主硬件树。

# 1. 工作目录与仓库

| 用途 | 路径 |
|---|---|
| Git 主仓 | `/home/zhumd/work/sdformer_codex` |
| 分支 | `autoresearch/neuron-ops-20260507`，跟踪 origin |
| 硬件 | `SDformer/hw_autoresearch_nts07/` |
| 算法/H67 训练 | `SDformer/neuron_experiments/H9_bipolar_self_attention/` |
| 额外硬件工作副本 | `/home/zhumd/work/hw_autoresearch_nts07/` |
| Synopsys/轨迹/DATE 史料 | `/home/zhumd/work/synopsys_date_dual/` |
| 开题/组会 | `/home/zhumd/work/KAITI_report/` |
| 多 AI 想法 | `/home/zhumd/work/sdformer_codex/ideafromai/` |

接手时 HEAD：**`7e3d3030`**  
说明：`Measure C1 dispatch opportunities and compression tradeoffs; close cofill mapped checks`  
当时工作区可能脏一份：`reviews/tcasii_accelerator_story_and_next_ideas_20260905.md`。先 `git status`。

Python 必须 3.12：`/usr/bin/python3.12` 或 `/opt/anaconda3/bin/python3.12`。系统 `python3` 是 3.6，会炸。  
License：`27030@ic.ismd-nemo`。EDA 互斥锁：`/tmp/date_dual_synopsys_same_uid_eda_queue.lock`。  
禁止把 Yosys/OpenROAD/Nangate45 当 ASIC PPA。没有 SRAM/RF `.db` 不得 `PPA_ADMISSION=1`。  
**禁止改 `docs/359`**（SHA `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`）。禁止 H81 RTL。LFSR 冻结 `16'h1d3f`。

# 2. 目标刊物：只投 TCAS-II

- 全称：IEEE Transactions on Circuits and Systems II: Express Briefs  
- 门户：https://ieee.atyponrex.com/journal/TCAS2  
- 作者指南：https://ieee-cas.org/publication/TCAS-II/guidelines-author  
- **严格 5 页**：正文最多 4.5 页，**最后 0.5 栏只能是参考文献**。超页会桌退。  
- **单盲**：投稿前要换成真实姓名、IEEE 会员、邮箱、ORCID、基金。现在稿里是占位。  
- **二值录用/拒稿**，几乎不修回。  
- 投稿信必须声明 **未投其他刊物/会议**。因此 **ISCAS 2027 不再投、也不能同时投**。  
- Circuits 编辑部常见桌退：看不出相对先验的 **性能优势**（他们习惯看实测）。纯仿真短文风险高；C1 九宏有 P&R+SPEF 的空间，C2 hold 需要匹配 P&R，不是 `set_fix_hold`。  
- 2026 Hybrid OA APC 约 $2800；传统页费约 $110/页。  
- 老师曾希望 9.20 前交正文、赶 ISCAS 10.13 前拿到一审——**时间线上做不到**（按近年一审约 6 周会拖到 11 月）。用户后来说 9.20 不再是硬门。当前策略：**把 TCAS-II 当唯一电路出口**，不赌双会。  
- 短文口味：一个机制、一张因果图、组件级指标可以成立（不必整网 FPS），但必须诚实标签 model/VCS/DC/PT。不要实验室流水号、不要 M 编号进正文。

现有 `paper/tcasii/main.tex` 标题仍是 *Single-Port Product Capture and Context-Safe Weight Broadcast*，摘要仍报 C1 1.6945×（周期模型）和 TSBG 1.8345×（VCS）。**用户已经判定这条故事没有创新、C1/C2 都要重做或降级。** 接手后默认：**旧稿是基线实现的写法，不是最终贡献句。** 未获用户点头不要把旧摘要当必须保住的数字去包装。

# 3. 算法架构（冻结身份 = Motion C12 / H67 / ep34）

## 3.1 任务与检查点

- 任务：事件相机二维光流，主评 DSEC（本地 valid825：825 帧 / 18 序列 / 48,152,523 有效像素）。硬件账本常用 `zurich_city_09_a`。
- 输入：异步事件 \(e=(x,y,t,p)\)，\(p\in\{-1,+1\}\)，先聚成体素（时间 bin 与 PSN 步 \(T_{\mathrm{snn}}=10\)，极性正负分开），再进网络。
- 软件主干：**不改** `third_party/SDformerFlow`。顶层类 `MS_SpikingformerFlowNet_en4`（多尺度 U-Net + 3D Swin 脉冲编码器）。
- H67 用 **overlay** 挂到 `sys.path` 前部，替换 `models.STSwinNet_SNN.*`。真正改动在：  
  `sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/`
- 冻结检查点：**Motion C12 ep34**，SHA 前缀 `4bbaf7fc`。valid825 AEE **1.199514**，发放 **5.6709%**。  
  `hardware_quant_enabled=false`。Motion-XOR **α=0.125**。**K 当 V**。
- **不要**和 ep35 / `docs/359` 混用（那是另一份历史冻结，AEE 约 1.33）。  
  `docs/H67_MOTION_ALGORITHM_FILE_INDEX.md` 仍写着 ep35、α=0.25、窗 9×9——**过期**。当前硬件主线是 **全分辨率 \(T_w=2\)、空间窗 \(15\times 15\)、\(N_{\mathrm{tok}}=450\)**。

## 3.2 网络拓扑（逻辑数据路径，不是已测的单一 RTL 顶层）

```
Event voxel (T=10 bins, polarity)
  → PatchEmbed / head conv
  → 瓶颈 / 残差 Conv3×3（96-lane）          ← 现有 C1 挂这里
  → 4-stage Swin 编码器，共 12 个注意力块
        stage0: 2 块, C=96,  约 240×320
        stage1: 2 块, C=192, 约 120×160
        stage2: 6 块, C=384, 约 60×80     ← 深度最大
        stage3: 2 块, C=768, 约 30×40
        每块：Linear Q/K → ATLIF(sn_q/sn_k, T=2)
              → Motion-XOR 分数 →（部署）Shiftmax 门控
              → attn = gate ⊙ K（无独立 V 投影）
              → proj + MLP（FC1/FC2）        ← 现有 C2/TSBG 挂这里
              → ATLIF T=2 或 T=10
  → 2 × ResBlock
  → 4 × ConvTranspose 解码 + skip（U-Net，不是 RAFT 相关体积）
  → 4 个 flow head，时间维求和，插值到全分辨率
```

几何（冻结工作负载，不是短测 yaml）：

| 参数 | 值 |
|---|---|
| \(T_{\mathrm{snn}}\) | 10（PSN/神经元展开） |
| 注意力时间窗 \(T_w\) | 2 |
| 空间窗 | \(15\times 15\) |
| 窗内令牌 | 450（=2×15×15） |
| 瓶颈通道 | 96 |
| 注意力 | 12 块，全线统一 H60/H67 路径，不是公开 SDSA 的 stage 混用 |

注意力在**旧**周期信封里大约 **0.6%**。所以旧组件稿不把注意力当性能贡献；瓶颈卷积和前馈才是现有 C1/C2 对象。用户现在要创新，可能反而要把注意力叶做成新岛——**必须先在 ep34 上重测份额**，0.6% 不是新测的系统加速许可。

解码器 8700-shard 全量回放未完成，**没有整网 Table A**，没有合法整机 FPS。

## 3.3 注意力算术（身份 vs 部署）

冻结软件路径（训练/评估身份）：

- Motion-XOR 打分，α=0.125
- K 复用为 V：`attn = gate ⊙ K`
- 量化关

部署候选（另开评价，M2045 一类，**不是**冻结训练算术）：

- Q7 分数、next-power-of-two 分母的 Shiftmax、Q1.7 门控
- 4 个瓶颈 Conv3×3 与 4 个 ConvTranspose 的逐通道 dyadic-INT8 QDQ

**禁止**写「分数是 2 的整数幂所以只移位」。门控仍有 `gate × weight`。Q1.7 约 28 种码。

Motion-XOR 硬件叶源码：  
`hw_autoresearch_nts07/rtl_h67/h67_motionxor_score_q7.sv`  
（synopsys 树也有 `rtl_h67/h67_motionxor_score_q7.sv`）

冻结部署注释公式：

`round_even(128 * (overlap + same_zero/64 + motion_xor/4) / 32)`  
- overlap = `popcount(Q AND K)`  
- motion_xor = `popcount(K XOR K_peer)`（时间对端 K）  
- same_zero = 共静默  

overlay 核心实现：`.../overlay/models/STSwinNet_SNN/bsa_attention.py`（H60/H67 分数、Shiftmax、gated-K）。  
公开 SDformerFlow 注意力是 SDSA/线性 QK，**不是**这套公式。新岛若做注意力，故事是「H67 Motion-XOR 的第一份数字映射」。

## 3.4 算法源码地图（只读这些写方法节）

仓库根：`/home/zhumd/work/sdformer_codex/SDformer/`

| 路径 | 角色 |
|---|---|
| `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_STSwinNet.py` | 顶层 en4 |
| `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py` | 3D Swin 载体（含已注释的 `sn2_q`/`attn_sn` 原路径） |
| `third_party/SDformerFlow/models/STSwinNet/PatchEmbed.py` | patch/卷积前端 |
| `third_party/SDformerFlow/models/unet.py` | 多分辨率外壳 |
| `third_party/SDformerFlow/DSEC_dataloader/` | DSEC |
| `neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py` | **H67 注意力核心** |
| `.../atlif_ternary_psn/atlif_ternary_psn.py` | PSN+ATLIF，binary `{0,θ}` |
| `.../atlif_ternary_psn/installer.py` | 按 yaml 安装（Q/K 与 `all_non_qk`） |
| `.../atlif_ternary_psn/training.py` | 阈值训练/冻结 |
| `.../h9_losses.py`、`h9_load_audit.py` | 损失与 210-key 加载 |
| `neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py` | 训练入口 |
| `src/` 下 sdformer/sparse_ops | **另一套骨架，不是 H67 主线** |

# 4. ATLIF 计数（不要写 85）

# 4. ATLIF 计数（不要写 85）

**105 是安装数，仍然正确。** 12 个 Swin 注意力块：stage0/1/3 各 2 块，**stage2 有 6 块（0–5）**，总共 12，不是 8。  
105 = 12×5（每块 `sn_q, sn_k, sn2_q, attn_sn, proj_sn`）+ 12×2 MLP + 3 downsample + 4 decoder + 8 resblock SN + 2 patch + 4 pred。

两批 12 **禁止合成一句话**：

| 层 | 数量 | 含义 |
|---|---:|---|
| 安装 | **105** | checkpoint / 模块树里都在 |
| `sn2_q` | **12** | H67 用 `gate⊙K` 换掉「Q 求和再发放」。前向**从未调用**，不在 capture CSV。计算上关了，参数还在。不是已经做了电源门控。 |
| capture 调用 | **93** | 证据：ep35 `atlif_activity.csv` 93 个 unique name，`calls=1`。48 个 T=2 + 45 个 T=10。出口全是 `{0,θ}` 二值。 |
| `attn_sn` | **12** | 仍调用：`attn=self.attn_sn(x); x=self.proj(x)` 用的是 x。CSV 里 12 条全 `deployment_dead_result=True`。软件算了，功能图可删。 |
| 图上活着 | **81** | 93−12。36 个 T=2 + 45 个 T=10。固定推理真正有消费者。 |

**85 不是合法库存**（典型误算：93−8，把注意力块当成 8）。正文分母用 81 图上活着；不要写「只剩 85」。

ATLIF 训练（自适应阈值、homeostatic、发放正则）把网训成稀疏 `{0,θ}`。推理 θ 是 checkpoint 静态（`homeostatic_freeze_after_step` 1224）。官方 AT-LIF 允许把 θ 折进下一层 W。下游是加/减，不是新的幅值 PE。  
**禁止**把 Grok Bot 的 `{g, int8 p}` 当成 ep34。C3 只覆盖 45 个 T=10 的位级状态与提交（约 17 周期/tile，63756 µm²），**不是第三条加速**。

带符号 `load_source_sign` 是极性/校正协议，不是 ATLIF 还在输出模拟张量。

# 5. 现有硬件架构（实现还在，创新故事要换）

用户明确：看过 C1 和 C2/TSBG 后认为**没有创新**，就是别人做过的，换了乘法顺序。C1 要重做，C2 也要重做。现有岛可当执行底座和对照，**不能当 TCAS-II 标题贡献**，除非用户点头保住其中之一当「实现」而非「新机制」。

硬件**没有**已测的单一芯片顶层。三座 **execution island** 共享类型化源协议，从未作为 system accelerator 闭环。工艺意图：数字 **28 nm**，foundry **TS1N28 1RW SRAM 宏**，不是模拟 CIM，不是 FPGA 一作。

## 5.0 执行合同（三岛共用）

软件不把稠密张量直接喂给岛。公共描述符至少含：行/令牌下标、符号（`sign=1` 表示 −1）、目的上下文、last/terminal、有效位。空描述符不得进 live 位图。

- C2 自然二值源：`load_source_active` 且 `sign=0`（+1）。负号是附加桥：进 Acc24 前用九位补码取负，避免 INT8 −128 回绕。
- 精确回退：C1 无合法父行 → 空父行、发原始源；C2 权重行身份不一致 → 分开取，绝不合并不同符号/目的/Acc24。
- Acc24 有符号 24 位。观测累加约 \([-29680,27619]\)，缩位不进本轮。
- 有损剪枝/N:M 与这套无损回退分界。

## 5.1 逻辑挂载（不是已测 SoC）

```
事件体素
  → 卷积/瓶颈 96-lane 稀疏乘加     → C1 乘积捕获岛（1RW 父行）
  → 12 块注意力（Motion-XOR 叶在 rtl_h67；旧稿不当贡献）
  → Swin MLP 的 FC1/FC2            → C2 K8 织物 + 内部 TSBG
  → ATLIF T=10 状态机              → C3 覆盖瓦
  → ConvTranspose 解码             → 无第三加速岛
```

## 5.2 生产 RTL 目录（主仓 `hw_autoresearch_nts07/`）

树里有几百个 `rtl_m*` 实验目录，**不要把编号最大的当主线**。当前组件主线：

| 岛 | 目录 / 叶 | 说明 |
|---|---|---|
| C1 父存储 | `rtl_m528_dw1rw/` | 9×128×128 1RW 父行暂存；死写抑制 |
| C1 匹配管线 | `rtl_m935_c1_match_pipeline/`、`m935_m912_three_stage_exact_parent_match_product_capture_island` | 精确子集父行匹配 |
| C2 八 bank / K8 | `rtl_m803/`、`m803_fc2_*k8*` | 类型化 K8 vs K1×8 |
| C2 TSBG | `rtl_m2018/`，`m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv` | SHA `96fb3557…`，SCHEDULE_MODE 0/1 |
| Motion-XOR / Shiftmax 叶 | `rtl_h67/h67_motionxor_score_q7.sv` 及 `h67_temporal_*shiftmax*` | 身份叶，不是 C1/C2 周期贡献 |
| 更早 H67/Local5/QFIT | `rtl_h67/`、`rtl_h68/`、`rtl_local5/`、大量 `qfit_*` | 旁路/史料，不要当投稿微结构 |

综合/仿真：`dc_handoff/`（filelist、tcl、one-shot runner）、`contracts/`（json 合同+sha）、`reviews/`（hammer 收据）、`results/`、`tb_*`、`verif_*`、`system_simulator/scripts/`。

## 5.3 C1：瓶颈 Conv3×3 有限容量乘积捕获

- 64 行 × 16 源 × 96 通道精确乘积；精确子集森林（max-popcount 父行，原下标打破平局）。  
  数学：`P_r = P_p + Σ_{s ∈ M_r \ M_p} p_s`。  
- **这就是 Prosperity HPCA’25 的 product sparsity**，特化到 foundry TS1N28 **1RW**（9 个 128×128 宏），面积 166514.312 µm²。同账本 10 个 zurich_city_09_a、5184 万行：相对 strongest-zero **1.694510×**（−40.99%）；相对 bit-skip 约 1.689×。1.6945× **包含继承的乘积稀疏**，不能全算电路增量。  
- `rtl_m935` 审计后：**不是** grant 时刻最早活父行；**没有**任务内驱逐；未用父行写抑制 ≠ 最后消费者回收。稿里已删这些话，接手不要写回去。  
- 1RW 是工艺密度合同，不是数学必然。1R1W 可接近约 1.90×，但端口故事消失。  
- 「卷积复用」= 挂在瓶颈 Conv3×3 上，不是 Eyeriss 滑窗 ifmap 复用。  
- 禁止引用 M1665 的 152898.626。周期比是同账本 **模型**，不是全账本 RTL。9 宏 ≠ 105 宏 214912 B 全账本面积模型（约 0.988 mm²）。

## C2：类型化有符号 K8 + TSBG（在 C2 内部）

- 等带宽 K8 vs K1×8：周期 1913 vs 1945 = **1.016728×**。面积 131086 vs 585479 µm²（−77.61%）。定向吞吐/逻辑面积 **4.5411×**。两边都加同一 288 KiB SRAM 模型后约 1.687×。  
- **禁止**写 C2 相对单条 K1 的 4.76×。  
- TSBG **不是第三座岛**：同一份 RTL SHA `96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21`，`SCHEDULE_MODE` 0=token-major，1=group-major。在 `mem_req` **之前**决定。先验：ELSA Gustavson、Eyeriss、SpikeX、FireFly-T。剩下的差是类型化有符号、4 上下文、8 bank 的预请求抑制，很薄。  
- M2057 1920 点 RTL：2.4438× 周期，请求 −64.25%。门控后面积 **+0.496%**。hold 刚正但 **不是 post-CTS**。门控 PTPX 候选：低复用 **−11.53%** 能量、中位 +26.15%、高 +62.12%。摘要不得藏低复用变差。  
- 不要把 1.8345× 写成「一次 RTL 战役」；M2112 1.7657× 覆盖 12 个 FC2 身份。全 token 2.0874× 是 CPU 模型。  
- `row_live` / 空源跳过是普通基线，不是 TSBG。

## C3：Fixed-T10 精确服务

保证 C1/C2 数字不是截断网络。没有公平加速比，不进摘要倍率。

## 评价纪律

- 「RTL 加速比」= 同一工作负载上 **VCS + DC/PT + Formality** 都闭。  
- 预宏、prelayout 必须打标签。禁止发表 0.0118 W；旧 0.0118% 面积只属于未门控。  
- 有损数字不得与无损已准入数字混表。  
- 99.4% 是非注意力份额，**不是**跳过率。  
- M472 的 2.46× 是官方 Prosperity product-vs-bit，**不是我们的 RTL**。  
- 禁止 ep44 AEE 1.2819、禁止 1.770× 当编码器。  
- 禁止 C1×TSBG 相乘当系统加速。

# 5.4 技术文档目录（先读哪、别读哪）

硬件树 `docs/` 有 **500+** 篇流水账，按编号不是「越大越对」。**不要从 docs/00 顺序读到 525。** 先读下面「必读」，其余当史料。

基路径 A = `/home/zhumd/work/sdformer_codex/SDformer/`  
基路径 B = `A/hw_autoresearch_nts07/`

### 必读（当前身份）

| 文件 | 内容 |
|---|---|
| `/home/zhumd/work/KAITI_report/技术工程报告_脉冲Transformer软硬件协同_20260903.md` | **当前最完整的中文技术叙事**（C1/C2/C3/TSBG、数字纪律、ATLIF 两层 12） |
| `/home/zhumd/work/KAITI_report/组会_C1C2硬件线_20260905.md` | 组会用的 C1/C2 说明 |
| `/home/zhumd/work/KAITI_report/硕士开题报告_面向事件视觉光流估计的脉冲Transformer软硬件协同优化研究.md` | 开题（论文目标已改为只投 TCAS-II，开题里若仍写 ISCAS 以本提示词为准） |
| `B/paper/tcasii/main.tex` + `README.md` + `COVER.md` | 现行短文稿（故事过时，格式仍是 TCAS-II 5 页） |
| `B/docs/1741_ISCAS2027_C1C2C3统一叙事与准入表_20260901.md` | 三岛准入表（文件名带 ISCAS，内容仍有用） |
| `B/docs/525_公开机制迁移包装与24小时同资源门_20260827.md` | 公开机制迁移 / 同资源门 |
| `A/docs/H67_MOTION_ALGORITHM_FILE_INDEX.md` | 算法文件索引（**身份数字过期**：仍写 ep35/α=0.25；用它找文件，数字以 ep34 为准） |
| `A/neuron_autoresearch/H67_PAPER_IDENTITY_CONTRACT_20260813.md` | 算法身份合同（同样可能偏旧，对照 ep34） |
| `A/BASELINE_MODEL_WALKTHROUGH_ZH.md`、`BASELINE_FLOWCHART_ZH.md` | 上游 SDformer 骨架 |
| `A/FULL_STACK_TECHNICAL_GUIDE_ZH.md` | 全栈导读 |

### 冻结/禁改

| 文件 | 注意 |
|---|---|
| `B/docs/359_DATE终局冻结_20260813.md` | **禁止修改**。SHA `dedde7ce…`。ep35 历史冻结，**不要**用来改 ep34 数字。 |
| `B/docs/228_CICC2026光流芯片借鉴与FAED本土化设计_20260801.md` | CICC 2026 光流芯片笔记；空间局部性已被 ASNA-Flow 占，DLSS 只能借时间相似。 |

### 按目录逛（不要通读）

| 目录 | 里面是什么 |
|---|---|
| `B/docs/` | 500+ 篇实验备忘、DATE 审稿轮、架构迭代。早期 `00–09`、`01_nts07b_architecture_profile.md` 可能是 **9×9 短测/旧 H60**，与冻结 15×15 冲突时以工程报告为准。 |
| `B/reviews/` | 每个 M 号的 hammer/收据/fail-closed。查某次 DC/VCS 是否可引用时来这里。 |
| `B/contracts/` | json 合同 + sha256 + seal。开 EDA 前对合同，不要手改数字。 |
| `B/results/` | 原始结果；隔离目录带 `FAILED_DO_NOT_CITE` 的禁止引用。 |
| `B/paper/tcasii/` | **唯一投稿目标稿** |
| `B/paper/iscas2027/`、`B/paper/date2027/` | **停投**，只当历史稿，禁止再投、禁止双投。 |
| `B/dc_handoff/` | 综合/仿真 filelist 与 runner |
| `KAITI_report/ISCAS2027_submission/` | 停投的会议编译目录 |
| `A/neuron_experiments/H9_bipolar_self_attention/docs/`、`entrypoints/`、`configs/` | 训练与配置 |
| `A/neuron_autoresearch/` | 算法身份、AAE、注意力设计空间 |
| `A/paper_artifacts/`、`A/docs/` | 更早的论文草稿/矩阵 |
| `/home/zhumd/work/sdformer_codex/ideafromai/` | 多 AI 创新调研（第 8 节） |
| `/home/zhumd/work/synopsys_date_dual/` | 另一份 EDA/轨迹工作区；ep35 profile CSV 等 |

### 查证顺序建议

1. 工程报告（KAITI 20260903）定身份与禁止用语。  
2. `paper/tcasii/main.tex` 看现有五页写了什么。  
3. `ideafromai` 看创新方向冲突。  
4. 需要对某倍率时，再按工程报告里的标签去 `reviews/` + `results/`，不要从 `docs/100` 开始读历史 DATE 轮。

# 6. 硬件完成度（对 TCAS-II 桌退风险）

**相对能写进短文的（须标签）：**  
C1 九宏映射岛 DC/PT/FM，3 ns 建/hold 在该岛上过；同账本周期是模型。C2 K8 逻辑面积故事。TSBG 在选定 FC 身份上 VCS 强。C3 覆盖瓦片。

**未闭、不能装成已测优势：**  
C2 门控能量不干净（低复用变差）。C2 hold 非 post-CTS。ICC2 卡在 Library Manager。R9 FC2 续跑 **fail-closed**（slot 163 挂死，隔离 `FAILED_DO_NOT_CITE_NO_RETRY`，禁止自动重试）。若干 SAIF/PTPX（M2119 窗口、M2127 sdf 路径、M2235 filelist 等）已消耗，不得复活。解码器 ConvTranspose 96 通道已经铺满 96 lane，GANAX 式插零跳过 ≤1.0×，不是第三岛。

**2026-09-05 最后一批提交仍是打磨旧 C1/C2**（co-fill、消费者寿命、Prosperity 边界措辞、压缩/dispatch 探针 m2260–m2268），**不是新机制 RTL**。

旧 Codex 会话因 **remote compact** 失败（`stream disconnected` / `error decoding response body`，代理 `127.0.0.1:7897`）。请**新开会话**，不要 resume 那条。

# 7. 创新困境（用户为什么要你接手）

用户原话大意：C1、C2/TSBG 看起来全是别人做过的，没有包装就直接设计出来，没有 idea taste。C1 要重做；C2 不过是换乘法顺序，也要重做。要求从 ANN 加速器、SNN 加速器、光流加速器、只有算法没有硬件的顶会顶刊/arxiv、稀疏跳过，以及**针对我们改过的算法**（ATLIF 出口为阈值/二值而不是 T=10 模拟张量；Motion-XOR）找**新机制**。idea 路径只有两条：自己想；大量论文里借机制再改。

独立评审同意：C1 相对 Prosperity 要回答「有限存储+单口下别人没解决什么」；C2 相对广播/Gustavson 很近；typed-signed 在自然非零上多为 +1，负号靠定向测试，撑不起「必须全新有符号架构」。

TCAS-II 只有四页半：不能堆四个组件。要么 **一个新机制 + 组件级实测**，要么承认现有 C1 是 1RW 特化、创新分很低、桌退风险大。用户当前倾向前者。

# 8. 磁盘上三套 idea（禁止合成一座岛）

全部在 `/home/zhumd/work/sdformer_codex/ideafromai/`。

**A. Grok Bot（iscas_ssh）**  
`research/00–04、07–14`，Card A OP-STW（光流方向预测唤醒 tile），Card B HBG-RP（`{门控, int8 载荷}`）。假设 ATLIF 幅值**不能**折进 W。  
**与冻结 ep34 冲突。** 除非重训并开 AEE Pareto，否则不要当投稿机制。

**B. Codex 独立包** `codex_independent_20260905/`  
标记 `RESEARCH_HYPOTHESIS_ONLY`。同意 C1/C2 当不了主创新。假说：按目的 token 掩码**选择性构造共享部分和再广播**（近邻 Mailman / CSE，不能宣称新代数）。未授权 EDA。

**C. Grok 4.6** `research/grok46_20260905/` + `codex_cards/CARD_C_MX3P_DIRTY_SCORE.md`  
冻结身份 = **二值 ATLIF**。排序：Motion-XOR **三 popcount** + `K_peer` 影子 + 脏 lane/时间相似门，作为**一座**注意力岛；T=10 膜留在神经元岛、岛界只出 1-bit。  
α-XNOR（CVPR 2025）已有共静默项——只借项，α 他们用 0.3/0.5，我们冻结是 0.125。Chen 28 nm 已做过 T=4/2/1 mux 展开——混 T 必须收窄成「T=10 神经元 vs T=2 窗口」。历史周期信封里注意力可能只有约 0.59%（无限加速注意力 ≈1.006× 系统）——**先在 ep34 上重测份额，再决定能不能当短文主电路**。Card C 是草稿：用户点头 + `07` 里 T0–T5 统计之前不要做生产 RTL。

**禁止把 Card A+B+C 合成一个加速器。**

# 9. 硬禁令

- 不 resume `01a01043-…`，不 websocket 注入旧 Codex。  
- 不改 docs/359，不动 H81。  
- **只投 TCAS-II，不投 ISCAS，不双投。**  
- 不模拟 CIM；不用 Nangate45 冒充 PPA；不发 0.0118 W；不写 1.770× 编码器；不写 C2 vs 单 K1 的 4.76×；C3 不当加速；不把倍率相乘。  
- 隔离/未锤的数字不当论文。  
- 空 tile 跳过、bit-skip vs strongest-zero（约 1.003×）、ASNA-Flow 空间局部性当「我们的原语」、FireFly-T LUT6 AND-PopCount 当 CMOS 创新、Comperity 的 XOR-GeMM 复用冒充 Motion-XOR——全部杀掉。

# 10. 接手后先做什么

1. `git status` / `git log -1`，确认没有 resume 554 MB 会话。  
2. 通读 `/home/zhumd/work/sdformer_codex/ideafromai/`（第 0 节清单）。  
3. ATLIF 只使用 105 / 12 sn2_q / 93 / 12 attn_sn / **81**，永远不要 85。  
4. 默认**不要**继续给旧 C1/C2 综合出「更亮的 1.69×」。用户要的是新机制。  
5. 等用户在下列里点一项再开生产 RTL：  
   - （推荐先做统计）冻结 ep34 的注意力份额 + 行/lane 脏率（见 grok46 的 `07_next_stats_rtl_gates.md`）；  
   - 若份额够、脏率低：Motion-XOR 三 popcount + 脏门（Card C）；  
   - 若用户改口保住实现短文：C1 九宏 + 诚实 Prosperity 边界 + 等 C2 能量/hold，五页写组件不写整网；  
   - 若坚持实值 ATLIF：必须重训，不能偷 ep34。  

未点名不要改 `paper/tcasii/main.tex` 的贡献句，也不要新开 C1* 目录进主仓。

用中文与用户沟通。英文只用于标识符、论文专名和必须引用的数字。

--- 复制到此结束 ---
