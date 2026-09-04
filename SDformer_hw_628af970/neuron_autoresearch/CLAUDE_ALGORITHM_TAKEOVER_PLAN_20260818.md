# CLAUDE 算法侧接管计划

Date: 2026-08-18. 继 Codex（`019ec76b-…`）与 Grok（`DATE_ALGORITHM_GROK_TAKEOVER_20260813.md`）之后，
由 Claude 接管 DATE 论文算法侧推进。本文档为接管计划 + 当日决策记录。

## 0. 接管时点快照（2026-08-18 ~13:04 UTC）

- H82 Class-Major TTX ft15（从 H81 ep29 续训 15 epochs）训练完成（status.log `ALL COMPLETE`）。
- `checkpoint_epoch4/9/14.pth` 在 `results/dsec_fullres_w15_H82_class_major_ttx_ft15_20260817/`。
- 2026-08-18 12:47 UTC 启动 H82 ep14 standard_valid825 评测，**13:04 UTC 已完成（exit_code=0）**，
  结果见 §1.1。GPU 已释放。
- 磁盘 overlay `bsa_attention.py` 为 H83–H86 叠代版（SHA `66d0a339…`）；
  H82 冻结算子 SHA `807a50e0…`（训练进程 RAM 中运行的就是冻结版；评测 launcher 已记录 frozen vs disk
  的逐行 diff 结论：仅 1 行不可达代码差异 + h83-h86 纯增量，h82 数值路径与冻结版一致）。
- 硬件侧（hw_autoresearch_nts07/docs/441、442）在 DATE 尺度否决 H86 的 4.0 声称与 H83–H86 的
  递进声称；DATE 主模型仍为 H67 Motion-TTX ep35。`CLAIM_4_0 = NO`。
- 红线：不启动 GPU 任务、不动 GPU 进程、不改硬件仓、不删文件、不改冻结合同、
  不改 `bsa_attention.py`。新配置 seed 0，不混 MDR/MVSEC/transfer 表。

## 1. H82 ep14 valid825 结果与后续选项（问题 a）

### 1.1 结果（2026-08-18 13:04 落地，`standard_valid825/epoch14/spike_profile.json`）

| 指标 | H82 ep14 | 对比：四线 rank-1 |
|---|---|---|
| AEE | **1.281708** | Local5 1.281893（−0.014%）、H67 1.3297（−3.61%）、H81 1.3306（−3.67%）、NB0 1.4454（−11.32%） |
| AAE-2D（legacy） | **5.783146** | 历史最佳（Local5 5.8498、H67 5.9004、H81 5.9692） |
| AE-3D（Barron） | **5.496274** | 历史最佳（Local5 5.5087） |
| DSEC Fl-all (%) | **5.7727** | 历史最佳（Local5 6.0210） |
| Spikes (G) | 84.5805 | Local5 85.24（−0.77%）、H67 82.11（+3.0%）、H81 80.90（+4.5%） |
| energy (uJ) | 74735 | — |
| sparsity | 93.42% | — |
| 加载审计 | overlay 210/210、missing=0、unexpected=0 | 同四线协议 |
| module counts | ATLIFTernaryPSN=105、ShiftmaxAttention=12 | 同四线 |

结论：**H82 ep14 在绝对预算 45（H81 ep29 + 15 epochs）上与 Local5 ep44 同预算直接对比，
AEE 与 spikes 双赢（AEE 1.281708 < 1.281893，spikes 84.58G < 85.24G），成为 DSEC valid825 全线新 rank-1。**
加载审计 clean；checkpoint SHA `74a0756c…`、config SHA `12b5d274…` 与冻结合同一致（§6 已复核）。

预期 vs 实际：接管时我按 val-loss 轨迹（H82 终值 0.9752，明显低于 H81 ep29 的 val 1.1097，
即 −12.1%；H81 的 val 曲线本身噪声大：ep26–31 依次 1.005/1.021/1.110/1.050/1.266/1.028）
预估 AEE 落于 1.28–1.32、最可能 1.29–1.31。实际 1.281708，落在预估区间下沿之外（更好），与
“class-major 分区在训练后段收敛更优、且轨迹远比 H81 平稳”一致。

### 1.2 预算口径（重要，等预算对照免费）

H82 是 H81 ep29 的续训，epoch 记号为 0 基（`checkpoint_epochN` = 1 基第 N+1 个 ft epoch 结束时保存）：

| H82 checkpoint | = H81 ep29 + | 绝对 epoch | 四线“预算”口径 | 四线同预算参考（AEE） |
|---|---|---|---|---|
| ep4 | +5 | 34 | 35 | H81 1.3475 / H67 1.3297 / Local5 1.3355 / NB0 1.4584 |
| ep9 | +10 | 39 | 40 | H81 1.3438 / H67 1.3434 / Local5 1.3153 / NB0 1.4549 |
| ep14 | +15 | 44 | **45** | **Local5 1.2819（其 rank-1）** |

**等预算对照不需要任何新训练**：H82 ep14 已直接对上 Local5 的 rank-1 预算（45），并且赢了。
剩余只差 ep4/ep9 两个 standard_valid825 评测（各约 16 分钟 GPU）来补齐预算 35/40 两行。

### 1.3 决策树（触发条件用 AEE/spikes 阈值，参考四线值）

**A. 是否续训（ft40 / ft25）——默认不续训。**
依据：
- 四线先例：四条线都在预算 40–45 附近过最优点（last−best 0.7–1.3%）；Local5 ep49（预算 50）
  AEE 1.2982 反而比 ep44 差。
- H82 val loss 已在 1 基 ep11 见顶（0.9719），终值 0.9752（last−best +0.34%，比四线都平）。
- ep14 已是全线 rank-1，续训 28 小时换取的边际收益与过拟合风险不成比例。

触发条件（**需同时满足**才重议续训，且续训只追加 10 epochs（ft25），不跑 ft40）：
1. `AEE(ep9) − AEE(ep14) ≥ 0.02`（即最后 5 个 epoch 仍贡献 ≥1.5% AEE，说明曲线未到顶）；且
2. `AEE(ep14) ≤ 1.285` 且 `Spikes(ep14) ≤ 86G`（续训以不牺牲 spike 故事为前提）。
否则：停在 ft15，把 ep14 作为 H82 唯一主推。

**B. 是否做 equal-budget 对照——做（免费，强烈推荐）。**
- 无条件做：这是回答“H82 是不是多训出来的”的唯一证据，四条线均有此表。
- 步骤：H82 ep4、ep9 各跑一次 standard_valid825（复用 `run_h82_ep14_standard_valid825_20260818.py`
  模式，改 checkpoint 路径），输出进 `standard_valid825/epoch4/、epoch9/`。
- 目标表格：§1.2 的三行（预算 35/40/45），与四线表同协议（batch_size=1、valid825、BN no_running）。
- 预算 35 行特别关键：若 H82 ep4 AEE ≤ 1.3297（H67 rank-1），则“H82 即使同预算也赢 H67”成立，
  可直接回答审稿人。

**C. 主推哪个 checkpoint——ep14（唯一主推）。**
- ep14：DSEC rank-1（1.2817 / 84.58G），与 Local5 同预算双赢。论文候选主行。
- ep9/ep4：只进等预算表与机制消融（“class-major 增益何时出现”），不主推。
- 无 ep11 checkpoint（force_save 只有 4/9/14；val-loss best 在 ep11 未存盘）——不必补训。

**D. H82 的身份边界（必须遵守）**
- H82 尚无任何 RTL provenance；硬件侧创新分 2.6/2.7（docs/437、439），不是 DATE 主模型。
- H82 数字不得与 H81 G0 百分比、H67 RTL、Local5 RTL 混绑；spike 能量仍是代理不是芯片实测。
- H82 的 DATE 归属：DSEC 主表可新增 H82 行（同协议），但“主模型”裁决仍归 H67，除非硬件侧另行复议。

## 2. H86 冻结标准与开训前置条件（问题 b）

### 2.1 对照 H82 冻结时做了什么（已冻结，2026-08-17T15:21Z）

| 项 | H82 冻结内容 | 文件/SHA |
|---|---|---|
| 合同 JSON | schema、status=`OPERATOR_FROZEN_TRAINING`、时间戳、C8 标签、禁止项清单、父本、artifacts | `H82_CLASS_MAJOR_TTX_OPERATOR_CONTRACT_20260817.json`（JSON SHA `8770d985…`） |
| 算子 SHA | `bsa_attention.py` → `807a50e0…`（冻结后文件被续写，RAM 里仍是冻结版） | 合同 artifacts.operator_py |
| 配置 SHA | `dsec_fullres_w15_H82_class_major_ttx_ft15.yml` → `12b5d274…` | 合同 artifacts.config |
| 父本 SHA | H81 ep29 → `8825c933…` | 合同 parent |
| 合同 md SHA | `ccad95f0…` | 合同 artifacts.contract_md |
| 单元测试 | `tests/test_h82_class_major_ttx.py`（3 例 CPU 通过） | hw docs/436 复核 |
| 状态账本 | status.log 记录冻结 + 启动命令 | `results/…/status.log` |
| 禁止项 | Motion-XOR、Local5 stencil、multiplicity 等价改写、C1–C7 粘贴 | 合同 forbidden |

### 2.2 H86 现状盘点（差什么）

已存在（算法侧 2026-08-18 00:55Z 已写）：
- `H86_MEMBER_DELTA_CLASS_FILE_CONTRACT_20260818.{json,md}`：status=`OPERATOR_FROZEN_WAITING_FOR_H82_GPU`，
  claim_4_0=true、review 4.0（算法侧自评）。
- 算子 SHA `66d0a339…`（= 当前磁盘叠代版，已复核一致）、配置 SHA `359e17dc…`（已复核一致）、
  父本 H81 ep29 SHA `8825c933…`、测试 `tests/test_h86_member_delta_class_file.py` 6/6 OK。

缺口（对照 H82 冻结清单逐项核）：
1. **算子 SHA 不是“冻结快照”而是“当前状态”**：H82 的 `807a50e0…` 是冻结后才被续写，diff 可证；
   H86 的 `66d0a339…` 就是现网文件本身，任何人再改文件即失效。必须在启动 launcher 里
   记录 frozen==disk 且断言相等（同 `run_h82_ep14_standard_valid825_20260818.py` 的 SHA 记账模式）。
2. **无冻结状态账本条目**：status 停在 `WAITING_FOR_H82_GPU`，缺 `OPERATOR_FROZEN_TRAINING` 的
   status.log 式冻结记录。
3. **无 checkpoint SHA**：未训练，任何冻结都缺训练后闭环（rank-1 val-loss + AEE + ckpt SHA）。
4. **硬件侧未放行**：hw docs/439 明言“H86 还没有冻 SHA，不能开训”；docs/441、442 在 DATE 尺度
   否决 4.0 声称（无 directory RTL、expand 仍 `K_i*gate_c`、padded-15 不是测得的 41%）。
5. **配置语义待确认**：H86 yml `preserve_mean: false`（H82 为 true）——member-delta 展开不保均值，
   需确认为有意设计而非笔误；`class_stability_regularization_weight: 0.01` 需确认绑定的是
   C8.1 member Jaccard 而非 H82 的旧 C8.1。
6. **父本选择**：合同写 H81 ep29（非 H82 ep14）——开训命令必须用 `--prev_runid` 指 H81 ep29，
   不得混 H82 身份。

### 2.3 H86 冻结 checklist（开训前逐项打勾）

- [ ] 1. 算法侧重写合同 status 为 `OPERATOR_FROZEN_TRAINING`（新 JSON 文件，不改已冻结的
  `H86_MEMBER_DELTA_CLASS_FILE_CONTRACT_20260818.json`），记录新算子 SHA、新 JSON SHA。
- [ ] 2. 在 `results/` 状态账本（status.log 风格）写冻结条目：算子 SHA==`66d0a339…`、
  配置 SHA==`359e17dc…`、父本 SHA==`8825c933…`、时间戳。
- [ ] 3. 复跑 `tests/test_h86_member_delta_class_file.py` 6/6，并把结果 SHA 绑定进合同。
- [ ] 4. 启动 launcher（仿 `run_h82_*`）：启动时断言 frozen operator SHA == disk SHA，否则拒绝启动。
- [ ] 5. 开训命令核对：`--config dsec_fullres_w15_H86_member_delta_class_file_ft15.yml`、
  `--prev_runid …/H81_nomotion_bb1e4_ft40_20260811/checkpoint_epoch29.pth --finetune 1`、
  force_save 4/9/14、seed 0、无 MDR/MVSEC/transfer 混合。
- [ ] 6. 禁止项自检：无 Motion-XOR、无 Local5、无 `codes/token_gate/member_mask/member_idx/513-bin`
  作为 expand 操作数、无 class-name reuse_set loss。
- [ ] 7. 训练后闭环（照 H82 模式）：force_save 的每个 checkpoint 记 SHA；用 rank-1 val-loss 选点；
  standard_valid825 + load audit（105/12/210，missing=0/unexpected=0）。
- [ ] 8. 声称边界：H86 训练产出只能作为 C8 算子/对象草稿证据；DATE 主模型仍是 H67；
  不写 41% 目录、不写 C8.2、不把 padded-15 当实测 CSR 收益。

**结论：H86 现在不能开训**——不是缺算子 SHA（66d0a339 已存在），而是缺“冻结状态账本 + 启动断言 +
硬件侧放行 + 训练后闭环”这四样。硬件侧 441/442 已否决 4.0 声称，开训前需先与硬件侧对齐
“H86 训练仅为算法侧 C8 草稿证据、不参与 DATE 主模型竞争”的边界。

## 3. MVSEC 策略（问题 c）

### 3.1 主线 MVSEC 现状（已做完，不需要重跑）

- 协议：`MVSEC_TRAIN_TEST_PROTOCOL_20260717.md`（2026-08-01 按 CICC2026 全文修正）——
  train=outdoor_day2 仅此一个序列（day2 scratch）、test=indoor_flying1/2/3 + outdoor_day1、
  固定 800 输入/序列（CICC 口径）+ full-sequence 分开报告、事件掩码 AEE 为主指标。
- 已完成的同协议 scratch 运行（`results/mvsec_cicc_*`，均有 load audit）：
  NB0 ep11（macro 1.8273）、H67 ep12（**1.7671，唯一四序列全过门**）、H81 ep12（1.7926，IF1 不过）、
  Local5 scratch ep12（1.8011，IF1 不过）、Local5-FT（1.6686，transfer 行）。
- 结论（DATE_FOUR_LINE_PAPER_FIT §4）：MVSEC 门是 H67 当 DATE 主线的支柱之一；H81/Local5 scratch
  都在 IF1 翻车。

### 3.2 H82 是否做 MVSEC（训练 + 推理）

先决门：按协议“at most one new DSEC winner”做最小矩阵。H82 现为 DSEC rank-1（1.2817），
已满足候选门。但：
- H82 无 RTL、硬件侧创新分低；MVSEC 数字对“DATE 主模型”没有分量，只可能是算法侧的
  “class-major 跨数据集泛化”证据行。
- 成本：MVSEC-NB0 seed0（~2 天）+ H82-from-MVSEC-NB0（~2 天）GPU。
- 触发条件（建议）：H82 ep4/ep9 等预算行出齐、且 ep4 AEE ≤ 1.3297（同预算即赢 H67）之后，
  若论文需要 class-major 的跨数据集证据，再做 day2-scratch MVSEC（从 MVSEC-NB0 初始化）。
- **禁止 DSEC→MVSEC 直接 FT**：几何不同（480×640 vs 256×256），会重蹈 Local5-FT 的
  PE-drop 身份问题（见 §3.4）。H82 若要 MVSEC，必须 day2-scratch，与 H67/H81 同协议可比。

### 3.3 CICC2026 参考做法（已提炼，PDF 在 hw_autoresearch_nts07/docs/）

Tao Zhang 等，CICC 2026，“A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation,
Bit-Width-Aware Compression and Similarity Detection”：Spike-FlowNet 衍生 Hybrid U-Net、INT8、
三机制（BWAC 71.4% EMA 降、Dense-Channel-First Speculation 73.8%、DLSS 深度跳层）；
800 输入/序列评估，固定 800 的 AEE（0.84/1.32/1.14/0.52，均值 0.96）。
- 我们的 MVSEC 流程已按该论文的**实验组织**（固定 800 manifest、outdoor_day2 训练、四序列评估）
  落地，但**绝对值不可横向比**：模型、INT8、掩码口径都不同（0.96 vs 我们 full 1.77+）。
- CICC 的机制（BWAC/speculation/DLSS）只作为方法学引用，不作为 TTX 的机制归属（TTB 是本项目
  自己的机制，协议文档已声明）。

### 3.4 Local5 救援身份问题（PE=12 dropped keys、same_checkpoint_identity=false）

事实（`MVSEC_H81_LOCAL5_RESCUE_20260816.md` + hw docs/435 回执）：
- Local5-FT 是 DSEC ep44 预训练 + day2 15-epoch FT；加载时因 DSEC（480×640）与 MVSEC（256×256）
  几何不匹配，**重初始化了 12 个 positional-encoding tensor**（PE=12 dropped keys）。
- 其 checkpoint SHA `fe774db3…` ≠ DSEC ep44 ckpt SHA `19820bec…`，即
  **same_checkpoint_identity=false**——MVSEC 数字不能与 ep44 的 DSEC 硬件身份绑定。
- 435 回执明说：Local5-FT 只能作为 transfer/救援表，不继承 RTL/PPA 身份；若要用它当论文模型，
  需要整条新硬件 profile/replay/活动身份链。
- 对项目的影响：Local5-FT 1.6686 永远只能放 transfer 附表（四线 paper fit §5 已如此处理），
  且它的存在不改变四条 scratch 线的裁决。**对 H82 的教训**：任何 DSEC→MVSEC FT 都会踩同样的坑，
  H82 的 MVSEC 只能走 day2-scratch。

## 4. AAE 差距（问题 d）

### 4.1 审计结论（已冻结，不再翻案）

- `AAE_BASELINE_DIAGNOSTIC_20260717.md` + `AAE_METRIC_TEST_RECEIPT_20260805`（8/8 PASS）：
  本地 legacy AAE 是 (u,v) 二维方向角；DSEC benchmark AE 是 Barron (u,v,1) 归一化角。
  公式已修正并回归（AE-3D 为 benchmark-facing 指标）。
- `NB0_AAE_GAP_CLOSURE_20260812`：valid825（本地 18 段留出）与官方 hidden test（7 条独立序列）
  是不同 population/协议；本地三种聚合（frame-equal/pixel-global/sequence-balanced）均高于官方
  4.871，聚合方式本身不能闭合差距；`4.871` 只能由官方提交复现，不能把本地训练拉到该数。
- H82 现在的 AAE-2D 5.7831 / AE-3D 5.4963 是本地历史最佳，但**仍不与 4.871 同口径**。

### 4.2 还有没有可做的

1. **官方 DSEC test 提交**（唯一能真正对齐 4.871 的路径）：需要写 `mode=test` 的提交 writer
  （`eval_DSEC_flow_SNN.py` 目前只实现 `mode=valid`），消费官方 416 test samples、按 DSEC 格式
  出 flow PNG；提交前做 ATLIF/attention/overlay/BN policy 审计（`AAE_OFFICIAL_TEST_AND_CICC2026_PROTOCOL` §DSEC Closure）。
  这是代码任务 + 一次推理（GPU），不是训练。若 DATE 需要“官方 AE 数字”，这是唯一剩余项。
2. **本地聚合/调参：不再做**。三条聚合已证不能闭合差距，继续调只有过拟合 test 之嫌。
3. 论文写法：本地列与官方列分开；只比内部相对差（本地 AEE/AAE-2D/AE-3D 均优于四线）。

## 5. 磁盘 / 队列清理清单（问题 e，只列不动）

磁盘现状：`/root/private_data` 851G 已用（86%），可用约 150G。

### 5.1 必须保留（被账本/合同/配置引用或活体）

- 四线账本引用：`dsec_fullres_w15_NB0_equal_plus10_ep40_20260805`（2.1G）、
  `dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811`（2.6G，兼 H82/H86 父本）、
  `dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805`（3.0G）、
  `dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805`（3.5G）、
  `dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812`（2.0G）。
- MVSEC 引用：`mvsec_cicc_nb0_w8_seed0_v4_20260811`（5.6G）、`mvsec_cicc_h67_motion_w8_seed0_v4_20260811`（3.1G）、
  `mvsec_cicc_h81_nomotion_w8_seed0_20260816`（2.5G）、`mvsec_cicc_local5_w8_seed0_v4_20260811`（2.4G）、
  `mvsec_cicc_local5_dsec_ep44_ft15_20260816`（3.6G）。
- 身份链引用：`h67_allbinary_all12_motionxor_ttx_w025_…_20260711_setsid`（H67 crop 父本 ep19）、
  `h81_allbinary_all12_h60_nomotion_equalbudget_…_20260717_setsid`（H82/H86 配置的 source_crop_checkpoint）、
  `dsec_fullres_w15_H67_crop_bb1e4_resume10/15/30_*`（H67 lineage 锚点）、
  `dsec_fullres_paper_w15_*`（paper 口径运行）、`dsec_fullres_w15_rescue_*`（08-01 收据引用）、
  H82 活体目录（2.1G，正在产评测）。

### 5.2 清理候选（无任何账本/合同/配置引用；已按 08-05 prune audit 政策复核）

| 目录 | 大小 | 理由 |
|---|---:|---|
| `nts11aw_hw_h60_s23_sn2qbin_w720_stdlr_bs8_20260613_133609_setsid` | 2.8G | 早期 NTS11 crop 阶段 profiling，非 fullres 主线 |
| `nts11u_hw_h60_s23_scope_ffn_all_ternary_scope_full30_bs8_20260612_130819_setsid` | 711M | 同上（早期 scope 消融） |
| `nts11aa_hw_h60_s23_scope_downsample_ternary_scope_full30_bs8_20260612_065413_setsid` | 710M | 同上 |
| `nts11bd_u12_ds_w720_fastlr_full30_20260613_223042_bs8_20260613_223042_setsid` | 711M | 同上 |
| `nts11bd_u12_dsffn2_w720_fastlr_full30_20260613_212628_bs8_20260613_212628_setsid` | 710M | 同上 |
| `nts07_ffn_floor_hw_short_20260608_034549` | 1012M | 更早的 FFN floor 摸底 |
| `ttx_mdr_full_cupy_w6_cpuvoxel_resume_ep10_20260702_161956` | 1.5G | MDR 线负结果（redesign plan 明确不再训 MDR） |
| `ttx_mdr_ep20_calib_lr025_ep21_25_20260708_163326` | 1.3G | 同上 |
| `mdr_valid_resume_seed_epoch22` | 1.2G | 同上 |
| `h66a_allbinary_all12_axnor_matrix_shiftmax_…_20260712_setsid` | 1.5G | 被 h66d 取代的候选 |
| `h66b_allbinary_all12_hamming_linear_…_20260712_setsid` | 1.1G | 同上（hamming 候选被否） |
| `h66d_allbinary_all12_lr_ttx_…_20260712_setsid` | 1.1G | 待复核：若 Local5 线 crop 父本不是它则可清（见 §5.3 注） |
| `h68_allbinary_all12_castling_ttx_aux050_…_20260711_setsid` | 711M | castling 被否 |
| `h69_allbinary_all12_dyadic_temperature_ttx_x8_…_20260711_setsid` | 710M | dyadic 被否 |
| `h70_allbinary_all12_event_selective_ttx_maxshift3_…_20260711_setsid` | 710M | event-selective 被否 |
| `h71_allbinary_all12_window_context_ttx_…_20260711_setsid` | 711M | window-context 被否 |
| `h73_allbinary_all12_de9_match_code_…_20260720_setsid` | 887M | match-code 变体（cf10/de9/dn9），非主线 |
| `h79_allbinary_all12_cf10_match_code_…_20260720_setsid` | 887M | 同上 |
| `h80_allbinary_all12_dn9_match_code_…_20260720_setsid` | 1.8G | 同上 |
| `local5_ep44_gatecard_qat20_sweep_20260815` | 2.3G | ep44 QAT 扫掠，8-15 已重绑 component RTL，未被引用 |
| `local5_ep44_gatecard_qat200_dual_20260815` | 1.2G | 同上 |
| `local5_ep44_gatecard_tailgap_qat200_dual_20260815` | 1.2G | 同上 |

合计候选约 26G。**未执行任何删除**（红线）。清理前需再过一遍 §5.3 的复核项。

### 5.3 复核项（清理前必查，本次未查完的）

- `h66d_allbinary_all12_lr_ttx` 是否被 Local5 线的 crop 父本链引用（查 `dsec_fullres_paper_w15_h66d_local5_ep29_ft30_20260728` 的 source 记录）。
- 08-05 prune audit 已保留各 `h68–h71` 的 ep19/ep29 checkpoint 作身份锚点——本次清单是全目录删除候选，
  若执行需先确认锚点已被替代。
- QAT 扫掠目录是否被任何 gatecard/component RTL 收据引用（hw 侧 423/424/425 周边）。
- 清理动作须在 H82 等预算评测（ep4/ep9）之后，避免误删正在使用的 loader 缓存。

## 6. 已完成的零 GPU 准备工作（2026-08-18）

1. **H82 ep14 valid825 评测已自然完成**（非我启动；13:04 UTC exit_code=0），结果见 §1.1；
   我读取并核对了 `spike_profile.json`：AEE/AAE/AE-3D/Fl/spikes/energy/sparsity/module_counts/加载审计全部字段。
2. **H82 checkpoint 一致性检查（纯 CPU）**：
   - `checkpoint_epoch14.pth` 顶层为完整模型对象，加载需 `register_shiftmax_pickle_compat()`（与 harness 一致）。
   - H82 ep14 state dict 与父本 H81 ep29 键集**完全一致**（921 键，0 差异）→ 架构零改动，纯算法分支切换。
   - 用 H82 冻结配置重建模型：ATLIFTernaryPSN=105、ShiftmaxAttention=12；
     `load_checkpoint_with_h9_audit` 结果 overlay 210/210、missing=0、unexpected=0 → **与冻结合同一致**。
   - 三 checkpoint SHA：ep4=`c61313b7…`（与 hw docs/444 一致）、ep9=`fc219082…`、ep14=`74a0756c…`。
3. **算子/配置 SHA 复核**：磁盘 `bsa_attention.py`=`66d0a339…`（H83–H86 叠代版）、
   H82 配置=`12b5d274…`、H86 配置=`359e17dc…`，全部与各自合同记录一致。
4. **训练曲线核对**：H82 ft15 共 15 epochs（1 基），val loss 1.0285(ep1)→1.1266(ep4 峰值)→1.0045(ep5)
   →0.9719(ep11 best)→0.9752(ep15 终值)；H81 ep29 的 val 为 1.1097（且 ep26–31 曲线噪声大），
   H82 终值低 12.1% 且轨迹平稳。
5. **未生成 H82 ft40 配置**：按 §1.3 决策，默认不续训。若触发条件满足（ep9−ep14≥0.02 等），
   届时用 §1.2 口径手写/生成 `dsec_fullres_w15_H82_class_major_ttx_ft25.yml`（seed 0，
   parent=H82 ep14，force_save 每 5 epochs，无 MDR/MVSEC/transfer 混表），本计划为其预留模板说明。
6. **磁盘清单**：见 §5（只列未动）。

## 7. 队列与下一步（GPU 状态：H82 ep14 评测结束后，~13:12 UTC 已有一个新的 ep9
   standard_valid825 评测被启动（PID 2873300，非本会话启动），随后应接 ep4）

1. [H82 等预算] `standard_valid825` 评测 ep4、ep9（各 ~16 分钟，免费）——补预算 35/40 行。
   复用 `run_h82_ep14_standard_valid825_20260818.py` 改 checkpoint 路径。ep9 已在跑。
2. [决策] 依据 §1.3 阈值判是否重议续训（默认不续训）。
3. [H86] 若要与硬件侧对齐“算法侧 C8 草稿证据”边界，按 §2.3 checklist 推进冻结；在边界对齐前不开训。
4. [MVSEC] 视 §3.2 触发条件（需要 H82 ep4 ≤ 1.3297 先落地）决定是否跑 H82 day2-scratch MVSEC。
5. [AAE] 唯一剩余项是官方 test 提交（写 mode=test writer），按 DATE 需要再立项。

## 8. 红线与证据分档

- 红线遵守：本会话未启动任何 GPU 任务（评测为接管前已运行、本次仅读取其结果）、未动 GPU 进程、
  未改硬件仓、未删文件、未改冻结合同、未改 `bsa_attention.py`。
- 证据分档：
  - `[rtl]`：H67 ep35 同 ckpt component RTL、Local5 ep44 重绑（hw 侧证据，本计划只引用）。
  - `[prof]`：spike_profile.json（AEE/AAE/AE-3D/Fl/spikes/energy/sparsity/加载审计）。
  - `[模型]`：H82 ep14 AEE 1.281708 属标准 valid825 同协议评测（§1.1）；checkpoint/配置/算子 SHA 已核。
  - `[待验证]`：H82 ep4/ep9 等预算行（未评测）、H86 冻结闭环、H82 MVSEC（未跑）、
    official DSEC test 提交（未实现 mode=test）。
