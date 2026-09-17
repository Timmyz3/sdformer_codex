# Claude 侧全量工作交接：融合 + 精度（2026-09-15，给 Codex）

> 目的：用户要求把 Claude 侧从开始到现在的全部融合工作与精度工作汇总，
> 标明产出文件位置，供 Codex 主对话交叉引用。
> 根目录：`ideafromai/research/hardware_innovation_20260908/`（下文相对路径均以此为根）。

## 0. 归属声明（防止冒领/漏认）

- **Claude 侧自有产出**（本报告覆盖）：`survey_ab_fusion_20260910/`（后期筛选加工）、
  `exploration_tcasii_20260911/`、`a_plus_x_screening_20260914/`、
  `claude_fusion_trials_20260914/`（T1–T12 全部）。
  初始 A+B 文献清单来自 ChatGPT web 交接（`survey_ab_fusion_20260910/CHATGPT_WEB_HANDOFF_FULL.md`
  / `sources_read.md`，按原稿归档）。
- **Codex 侧目录，只引用不冒领**：`open_fusion_execution/`（catalog 775 实体/698 论文、
  精度政策与 NB0 基准）、`fusion_ten_trials_20260914/`、`r8_consumer_fusion_20260914/`、
  `r0_stream_fusion_20260914/`、`r0_source_retirement_20260913/`、`r0_execution_trials_20260913/`、
  `pro_fusion_trials_20260913/`、`deep_target_research_20260913/`、
  `representation_transfer_20260914/`（质量报告）、`shared_execution_20260915/`（含对
  Claude T5/T6 的独立审计）、`paper_mechanism_transfer_20260915/`、`algorithm/ psn/ bn_state/`。
- 生产树 `hw_autoresearch_nts07` 与主稿：只读，未动。全部 Claude 代码在隔离树，
  未复用/修改其他 AI 的代码。

## 1. 时间线总表

| 阶段 | 日期 | 目录 | 一句话结论 |
|---|---|---|---|
| A 文献盘点+P0 精读 | 09-10~13 | `survey_ab_fusion_20260910/` | 300+→732 合并、245 张 P0 idea 卡、三轮 killcard、RTL 融合筛 13 候选全杀/BASE_ONLY |
| B TCAS-II 独立探索 | 09-11 | `exploration_tcasii_20260911/` | AT-LIF {0,θ} 身份定案；F1 双完成 last-use 为唯一对准已测洞的方向；四轮（independent/absorb/gap/wide） |
| C A+X 六卡筛选 | 09-14 | `a_plus_x_screening_20260914/` | 六张一页卡+三杀实验：粒度是决定性变量；C1 分级条件完成晋级 |
| D T1–T10 照抄试验 | 09-14~15 | `claude_fusion_trials_20260914/` | 四杀一留：**C1 存活形态定案（组级 BFP+位平面证书，17.3–17.7%）** |
| E T11 全库过筛 | 09-15 | `claude_fusion_trials_20260914/t11_triage/` | 698/698 五桶：APPLICABLE_NEW=0 |
| F T12 标杆精读 | 09-15 | `claude_fusion_trials_20260914/results/T12_BENCHMARK_READING_20260915.md` | C1 拍数下界解析证明；移植空间双通道穷尽 |

## 2. 融合工作明细（按阶段，含文件路径）

### 阶段 A：文献盘点与 P0 精读（`survey_ab_fusion_20260910/`）

- 文献合并与分层：`literature_merged_300plus.{md,csv,json}`、
  `literature_precision_732.{md,csv,json}`（732 篇精度分层，P0=246）、
  `literature_list_300plus.md`、`literature_new_this_round.md`。
- P0 全文/方法级精读：**245 张 idea 卡**（`idea_cards/`，含 Phi.md、unresolved_Bishop.md、
  unresolved_COMPASS.md 等）+ 数据行 `idea_extract_per_paper.csv`（252 unique uid）、
  进度表 `p0_deepread_progress.md`；全文缓存 `p0_txts/`（本 session 起点）、`p0_pdfs/`、
  摘要批次 `p0_excerpt_batches*/`。
- 候选综合与优先级：`ab_fusion_candidates.md`、`ab_fusion_priority.md`、
  `idea_synthesis.md`、`novelty_near_neighbors_x1_x2.md`、`tech_arch_snippets_from_survey.md`、
  `TECH_ARCH_SOFT_HARD_FULL.md`、`understanding_and_gaps.md`。
- 三轮 killcard：`fusion_killcards_round1/2/3.md`（短杀门纪律：same-port/背压公平、
  BASE_ONLY 不得冒充 X）。
- RTL 融合筛（iverilog 12.0 + yosys 0.52 techmap，13 候选）：
  `rtl_fusion_ROUND1/2/3_REPORT.md`、`rtl_fusion_SCOREBOARD.md`、`FUSION_RTL_ROUND2/3.md`、
  `OVERNIGHT_RTL_SUMMARY.md`、`rtl_microprobes_for_box.md`。
  结论：KEEP_PROBE 无；BASE_ONLY=cand1c/cand8/驻留R24；KILL=cand1b、cand2–7、cand9–13
  （弹性直供/基底+例外/gate-summary×NRV 等全 KILL_LAYOUT）。
- 论文设计草稿：`f1_f2_paper_design.md`、`f3_f7_paper_design.md`。

### 阶段 B：TCAS-II 独立探索（`exploration_tcasii_20260911/`）

- 身份定案：`IDENTITY_ATLIF.md`（AT-LIF 二电平 {0,θ}，θ 推理折入 W——后续所有
  融合论证的前提）；问题冻结 `SCOPE.md`/`PROBLEM.md`。
- 四轮探索：`independent/`（六路独立构想）、`round2_absorb/`（FireFly/Bishop/Phi 映射，
  含 `literature/R2L3_firefly_bishop_phi.md`）、`round3_gap/`（含
  `literature/R3L1_firefly_t_iand.md`）、`round4_wide/`；融合候选
  `fusion/FUSION_CANDIDATES.md`；对抗审阅 `adversarial/`；可证伪记录 `hypotheses/`；
  取舍日志 `DECISION_LOG.md`；文献精读 `literature/`（Prosperity 一页卡
  `L1_prosperity_exspike.md` 等）；会刊扫描 `web/W1_2025_2026_venues.md`。
- 结论：F1（双完成 last-use：连续 θg 双消费者占用并集）是当时唯一对准已测洞的方向；
  交接稿 `CODEX_HANDOFF.md`、总报告 `REPORT.md`。

### 阶段 C：A+X 六卡筛选（`a_plus_x_screening_20260914/`）

- 入口 `README.md`；六张一页卡：`C1_consumer_graded_completion.md`、
  `C2_rsparse_two_path.md`、`C3_rounding_margin_cert.md`、`C4_motion_aligned_delta.md`、
  `C5_scannow_window.md`、`C6_dcom_runtime_decomp.md`。
- 三杀实验：`kill_experiments/run_kill_abc.py` + `kill_{a,b,c}_raw.json` +
  `RESULT_{A,B,C}.md` + 综合 `kill_experiments/SYNTHESIS.md`。
  关键数字：lifting |q|<Q/2=45%；逐通道接受 97.65% 但块级 78.5% 必有失败通道；
  lane8 联合接受 85.64%、门占用 25.1%（anchor=25% 时并集节省 56.2%）；b=4 即锁 95%。
- 未覆盖文献检索：`literature_search/UNCOVERED_WORKS.md`（MINT/ANNS-AMP/Sparsity Tax/
  APEX/M-HySMap/SupraSNN/ASTER，后均入 T11 判读）。

### 阶段 D：T1–T10 照抄试验（`claude_fusion_trials_20260914/`，全部自有代码）

总入口 `README.md`；逐项报告+数据在 `results/`（同名 T*_REPORT.md + t*.json）：

| 试验 | 照抄对象 | 结果 | 报告 | 脚本/RTL |
|---|---|---|---|---|
| T1 | 本地 K=4 条件完成+位平面 | 静态 12–14bit/57.9%，弱 | `results/T1_REPORT.md` | `t1_lane_bitdepth.py` |
| T1b | BitFair 式逐判决终止 | 2.22bit 系口径错误→救回 T5 | `results/T1B_REPORT.md` | `t1b_dynamic_depth.py` |
| T2 | MotionDeltaCNN 运动对齐 | 仅降 4.8%，**杀** | `results/T2_REPORT.md` | `t2_motion_toggle.py` |
| T3 | R-Sparse 幅值两路 | anchor=精确 25%；分流需 keep≥50%，空间有限 | `results/T3_REPORT.md` | `t3_anchor_rsparse.py` |
| T4/C3 | lifting40 RNE 跳过证书 | 命中率 0.04% vs 门 15%，**杀** | `results/T4_C3_REPORT.md` | `t4_c3_rounding_cert.py` |
| T5 | C1 证书门核最小 RTL | **17.3–17.7%，零差+过审计，晋级定案** | `results/T5_REPORT.md` | `t5_c1_cert_core_model.py`、`t5_rtl/`、`t5_neg_gamma_regression.py` |
| T6 | ptau 可实现版供数 | 16.5–17.7% 全程稳定 | `results/T6_REPORT.md` | `t6_ptau_supply.py` |
| T7 | ConvReflex 静态锁深 | 误 fire 37–38%，**杀** | `results/T7_REPORT.md` | `t7_convreflex_static.py` |
| T8 | AO-BFP 离群打包 | 元数据税杀（11–12% 需旁路通道） | `results/T8_REPORT.md` | `t8_aobfp_outlier.py`、`t8b_static_msb.py` |
| T9 | EITCE 通道序+ECHO 符号 | 59–61% / 2.3–7.1%，**杀** | `results/T9_REPORT.md` | `t9_channel_order.py` |
| T10 | BitL LUT 数据通路 | 乘法全消+LUT 化，三重零差，**保留** | `results/T10_REPORT.md` | `results/t10_rtl/`（`cert_gate_bitl.sv`、`tb_bitl.cpp`、`verify_t10.py`） |

给 Codex 的滚动建议：`CODEX_SUGGESTIONS.md`（含 T1b 口径修正、粒度定律、
f14 精度边界口径、anchor=25% 结构常数等）。

### 阶段 E：T11 全库过筛（`claude_fusion_trials_20260914/t11_triage/`）

- 脚本：`classify_b0.py`–`classify_b6.py`（7 批逐篇人工判读 dict）、`make_tsv.py`、
  `merge_t11a.py`、`fetch_abstracts.sh`（arXiv 摘要补拉，代理限流下 2/10 成功）。
- 数据：`input/batch_*.json`、`output/batch_*.jsonl`、合并 `t11a_merged.json` +
  `t11a_summary.json`。
- 结果：698/698 五桶 = INCORPORATED 50 / FAMILY_COVERED 154 / APPLICABLE_BLOCKED 78 /
  NOT_APPLICABLE 416 / **APPLICABLE_NEW 0**。
- 总汇报：`results/TOTAL_REPORT_20260915.md`（融合/调研/性能三部分 + 78 篇被挡三级清单）。

### 阶段 F：T12 标杆精读（`claude_fusion_trials_20260914/results/T12_BENCHMARK_READING_20260915.md`）

Bishop/Phi/FireFly-S/T/GustavSNN/Prosperity 全文 + COMPASS/C-Transformer 题名级。
产出：C1 供数协议拍数下界解析证明（sop 拍免费打包 h+sign+e，sign 即最高位平面，
4.2 拍/组为同端口下界）；非 SNN→SNN 移植双通道穷尽判定表；GustavSNN 式解析评估
模型与 Bishop ECP 式叙事两个零成本移植；COMPASS（推测+恢复）入索取清单（11→12 篇）。

## 3. 精度工作明细（Claude 侧产的是"判决层"精度证据，不产整网 AEE）

| 项 | 结论 | 文件 |
|---|---|---|
| f14 精度边界 | RNE 到 14 分数位仍有 122/73.7M 门判决翻转（1.7ppm）；"无损"只能以部署整数链为基准 | `results/T1_REPORT.md`、`CODEX_SUGGESTIONS.md` 第三节 |
| T1b 口径修正 | "2.22bit/9.24%"系相对精度位数口径错误（ceil(log2(L1/margin))/24），已公开修正 | `results/T1B_REPORT.md`、`CODEX_SUGGESTIONS.md` 头部 |
| T5 零差验证 | 4 traces×4 模式 320 万判决与全深度整数模型零差、cert 拍数逐组一致（Verilator 4.028） | `results/T5_REPORT.md`、`results/t5_rtl_verify.json`、`t5_rtl/` |
| T5 审计修 bug | 模型侧负 gamma 方向双折 + 49bit thr 合同 bug（Codex 审计发现），已修+合成回归通过；真实数据判决不变 | `t5_neg_gamma_regression.py`、`results/T5_REPORT.md` |
| T6 扰动鲁棒性 | thr 扰动（ptau−终态 tau 差 0.5–4×，相对 3–7%）拍比 16.5–17.7% 全程稳定 | `results/T6_REPORT.md`、`results/t6_ptau_supply.json` |
| T4/C3 舍入证书 | 真实输入 40 写命中率 0.04%（1516/3.93M）vs 门 15%，连续稠密状态结构性无效 | `results/T4_C3_REPORT.md` |
| T10 三重零差 | vs 模型判决+拍数、vs T5 RTL，均零差 | `results/T10_REPORT.md`、`results/t10_rtl_verify.json` |
| a_plus_x 静态杀实验 | 接受率/占用/位深锁定的样本内静态证据 | `a_plus_x_screening_20260914/kill_experiments/SYNTHESIS.md` |

**AEE 口径（引用 Codex 侧，非 Claude 产出）**：质量门=优于同环境原 NB0
（valid825 1.447937，另一口径 1.445353）；最新网络质量见
`representation_transfer_20260914/quality/QUALITY_REPORT.md`（自由U 1.258343 等）。
C1 的 AEE 主张是"门路判决不改（证书=精确终止，非近似）"，**尚未做真实 consumer
净服务与整网 AEE 联测**——这是收尾第一优先（见 §6）。
Codex 审计（`shared_execution_20260915/claude_review/README.md`）已指出：
T6 五点扰动仅证明样点敏感性，不证明部署阈值/动态 BN 矩为训练期冻结常量；
17% 成立范围限于"预生成输入+已知阈值的局部门核协议"。这两条 Claude 侧接受，
net-service 实验即为回应。

## 4. 性能总表（C1 当前最强候选）

| 指标 | 数值 | 出处 |
|---|---|---|
| 供数拍比（vs 诚实 FX 基线 24 拍/组） | **17.3–17.7%**（≈3.2 数据平面+1 sop 拍/组） | `results/T5_REPORT.md`、`results/T6_REPORT.md` |
| thr 扰动下 | 16.5–17.7% 稳定 | `results/T6_REPORT.md` |
| Codex cert_transport 迁移复测 | 相对同电路 BF full −22.66~23.88% 周期（背压 −12.35~13.36%），外部字节不减 | `shared_execution_20260915/cert_transport/RESULTS.md` |
| 数据通路 | dot10→2×5bit 子集和 LUT（12.8kb）；20×48bit 乘法/平面→移加 | `results/T10_REPORT.md` |
| 拍数下界 | 4.2 拍/组为同端口协议下界（解析证明） | `results/T12_BENCHMARK_READING_20260915.md` §4.1 |
| RTL 状态 | 仅功能验证（Verilator），无综合/PPA/时序 | — |

## 5. 结构性定律（杀出来的，供 Codex 设计候选时直接引用）

1. **粒度定律**（三实例）：位深类收益可达性=f(供数粒度)；块级机制被 384 通道求并吃掉
   （78.5% 块必有失败通道）；接受单元锁 lane8。
2. **元数据税定律**（四实例）：~3 拍/组预算下 ≥1 拍逐词元数据吃掉 ≥33%；
   T8 的 11–12% 需旁路通道兑现。
3. **锁深是样本性质**（T7）：j* 跨序列 σ=2.30–2.42，静态表误 fire 37–38%——
   运行时证书必要性获定量论证。
4. **operand 序必须匹配 margin 精化结构**（T9）：位平面序每拍同步精化全部词+margin；
   词序需 oracle k=8.3–9/10。
5.（T12 补充）**证书界已紧**：未知 bit 为独立未知量，符号分裂（P/N）已捕获唯一可静态
   利用的结构；仿射算术/值预测/CSD 无残余空间。

## 6. 下一步推进计划（Claude 侧）

1. **C1 真实 consumer 净服务实验**（第一优先）：证书判定+元数据费用同端口计费，
   接真实 Y/τ 供数链（对齐 Codex cert_transport 的接口批评：外部字节不减的问题
   必须在净服务口径里显式回答）；目标 ≥15% 净服务准入门。
2. **t10_rtl 综合/PPA**：`results/t10_rtl/cert_gate_bitl.sv` 进综合，补面积/频率/功耗，
   与 FX 基线同口径对比。
3. **解析评估模型**（T12 §4.2，零 RTL）：拍比=f(margin 分布, ρ) 曲线 + 置信区间，
   强化 TCAS-II 评估节；顺带回应 Codex "T6 不证部署阈值"——用部署分布拟合替代
   五点扰动外推。
4. **T6 阈值来源补证**：与 Codex 协同确认动态 BN 矩/τ 是否训练期冻结常量
   （Claude 侧原脚本从当前完整 Y 计算动态矩，Codex 审计已点名）。
5. **12 篇全文索取**（用户行动，Claude 无剩余筛选工作量）：
   PredLM(W0751)/CompRRAE(W0366,W0108)/MFPSN(W0171)/Comperity(W0007)/CICC25 剪枝
   (W0703)/量化 spike Transformer(W0339)/ADET(W0113)/Broad-Spectrum(W0749)/
   ScanNow/AEU26/EITCE26/**+COMPASS(W0056，T12 新增)**。
6. F1 双消费者 last-use 维持未来工作（TCAS-II 5 页只容一个机制，C1 优先）。

## 7. Codex 侧交叉验证现状（引用）

- `shared_execution_20260915/claude_review/README.md`：T5 合法域+存档回放零差通过；
  口径批评两条（见 §3）Claude 接受并已列入待办。
- `shared_execution_20260915/cert_transport/RESULTS.md`：证书在同电路 BF full 对照下
  −22.66~23.88% 周期——与 T5 的 17% 属不同分母口径，两数并存不冲突
  （T5 分母=24 拍诚实 FX；cert_transport 分母=同电路 BF full）。
