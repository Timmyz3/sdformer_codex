# ideafromai — AI idea drop zone (Codex reads here)

**2026-09-12 统一公开idea入口并实际试做：** [全部工作/仓库/候选](research/hardware_innovation_20260908/open_fusion_execution/catalog/README.md) · [四家族融合、训练与RTL结果](research/hardware_innovation_20260908/open_fusion_execution/README.md) · [后续接口](research/hardware_innovation_20260908/open_fusion_execution/EXECUTION_QUEUE.md)。相位H8已做64步成对恢复及实际U/F消费者重执行；NRV列流精确父匹配RTL已用真实捕获验证；连续PED位面和完整动态BN四臂均已执行。公共源驻RF、code0压缩/默认传播有效，新增X仍未形成强接收证据。CICC遗漏补表已纳入，共16项并保留阅读层级。当前授权是适配组合先试原型，历史“先等某门才实验”不再限制早期尝试；负结果停具体布局，生产与主稿尚未重构。

**2026-09-11 Pro/Grok新调研已审阅并测量：** [逐路线复审、采用/修订理由及后续顺序](research/hardware_innovation_20260908/PRO_GROK_REVIEW_20260911.md)。源核8088确认来自指定FIFO背压；整数双消费者的原字最后读与输出完成已分开测，通用门提前对照暂无净服务增量；Pro空间P2舍入证书真实数据零新增命中。lifting主家族与物理源事务闭包候选保留，完整Stage B及相对AEE门仍未过。

**2026-09-11 用户最新更正：AT-LIF = {0,θ} 二电平；推理固定θ可折入下一层W。** 硬件发放接口可只传g，不能据θ是实数就要求逐事件幅度MAC。PSN发放前和连续残差仍另计。[当前推理合同与候选影响](ATLIF_INFERENCE_CONTRACT.md)。本条优先于下方及外部包的旧“不可折权/任意实值发放”前提。

**2026-09-11 目录迁移：本目录现由主仓直接管理。** 唯一实际位置为 /home/zhumd/work/sdformer_codex/ideafromai/；旧 /home/zhumd/work/ideafromai 是兼容链接。整个研究树已搬入，不再维护筛选同步副本。[Git与本地数据说明](README_SYNC.md)。

**Canonical path:** `/home/zhumd/work/sdformer_codex/ideafromai/`

**下一任 Agent（中文，只投 TCAS-II）：** `HANDOFF_NEXT_AGENT_20260905.md`

**2026-09-09 文献总表与去留复审：** [按会议/期刊查工作](research/literature_audit_20260909/VENUE_INDEX.md) · [逐篇题名、深度、停止/纳入理由与未试项](research/literature_audit_20260909/LITERATURE_INVENTORY.md) · [可筛选CSV](research/literature_audit_20260909/literature_inventory.csv) · [后续推进计划](research/literature_audit_20260909/assessment_and_plan.md)。仅停止有明确负结果的具体版本；完整迁移缺口、前提改变和未尝试方向分别列出。三代理分域复核，已有上下文，非盲法独立发散；本轮无新训练/RTL/EDA。

**2026-09-09 用户规定的挖点方法：** 每个候选只维护一页，按 **B（本网有证据的缺口）→最强对照→A（完整先验及尚缺实现）→X（真实增量）→固定杀门** 的顺序。先写B和对照再实验；不能先实现再找故事。数值门不过即停，不扫参续命；评分不等于录用概率。今天完成[支持码零响应](research/hardware_innovation_20260908/bn_state/response_zero_one_page.md)与[完整PSN判决](research/hardware_innovation_20260908/psn/psn_decision_one_page.md)两份一页审查：前者未超出约束块剪枝＋稀疏L；后者的“门确定就停”已有完整判决先验，尚缺同学生完整常矩阵编译，不能把MAC分母当最强A。均停在当前X的创新门，本轮无新增训练、仿真或RTL。[工程最新状态](research/hardware_innovation_20260908/README.md)。

**2026-09-07 当前：已完成现有 ep34 QK 普查与同账本 M0 MAC 工作量表。** [M0–M2 结果与机制判断](research/ep34_attention_measurement_20260907/README.md)。40样本×12块；S0/S1仅前缀窗口，不能称全帧普查。K零只直接允许跳输出乘积，不能删归一化贡献；行内配对复用仍成立，clean条件下打分次数零开销上限35.24%。分数叶工作量proxy偏小，真实系统周期份额未闭。C1融合/C2静态共享不重开，生产与主稿只读。按用户要求，本轮不做哈希封存或报告附件链。

**前一轮：C1 完整算子融合未胜出；C2 已补真正八 bank/T10 并行，新增共享仍无时延净收益。** [一页净收益表](research/same_workload_c1c2_20260907/net_benefit.pdf) · [完整结果与五项思想复审](research/same_workload_c1c2_20260907/report.html)。C1五轴、四档缓存、两种端口共40点CPU模型中，当前融合均未超过同资源Prosperity。C2共享比同结构直接并行轴多215周期（+0.23454%），包含最终归约的加法少8.9552%；生产与候选的内部资源不同，不能把大周期差归为新增共享。五个思想恢复研究，失败布局不等于思想被证伪。BN2/shortcut、完整生产/压缩、全帧与物理时序未闭合；两条线仍未完成生产重构，无强接收结论。nts07与主稿保持只读。

**前一轮：C1/C2 均未完成生产重构；完整迁移边界与单完整算子对照已补齐。** [完整迁移、实际结果与独立晋级判断](research/complete_transfer_20260907/report.html) · [执行合同](research/complete_transfer_20260907/complete_path_contract.md)。C1 已完成一个3000×6912×768算子的官方外层模型适配和独立复算；Phi根构造存在每K父值与跨K完成边界冲突。C2既有单源RTL叶仍非完整流水；新选定通道拆分布局的单模式筛查加法减少9.34%/15.11%，但捕获bank映射不同且没有bank专属模式选择的额外收益证据。整行等价类显式字典净状态收益薄，停止该版本。无新RTL速度、PPA、AEE或强接收结论。

**前一轮：C2 时间共享部分和已完成第一份协议 RTL；C1 两版新共享候选的增量仍薄。** [机制解释、强先验与参考结果](research/fusion_delivery_20260907/report.html) · [C2 原型及三配置功能收据](research/c2_temporal_shared_protocol_20260907/README.md) · [C1 保留父值提升的停止结论](research/c1_retained_parent_promotion_20260907/README.md)。C2 保留连续值阈值幅值，8 槽源顺序参考有20.78%/42.21%加法减量；单源在途原型通过逐段、多消费者背压及部分确认后复位检查。它仍是6.5/10候选，未建立强接收或 RTL 加速结论。C1联合图改写省1.17%、寄存器内提升省1.58%加法，均不升级标题贡献。完整 Prosperity、完整T直接FTP及RSR++继续作为强对照；不改主稿、生产RTL或已封存旧证据。

**前一轮：用户明确恢复 C1 的 Prosperity 融合重构。** [完整底座、双图主候选与首次参考结果](research/prosperity_fusion_20260906/README.md)。主候选为“静态基模式图＋动态纯残差图”，借 Phi/Prosperity/TA 改变共享对象与有限物化；独立概念新颖性7/10，参考已完成，当前整体优势未成立：共同缓存挤占，所选版本比同批原Prosperity多34.07%局部加法。仅联合选父的增量约1.2%；C2逐签名跨BN保存和SVD残差确认两版未过当前样本门。各结果仅CPU机会/诊断，不是稳accept。完整θ幅值、生产网络、主稿与EDA边界保持。此前C1降级记录不是永久禁令。

**前一轮：用户已授权按创新点推进硬件。** [第一版 RTL、机制说明与独立取舍](research/hardware_mechanisms_20260906/README.md)。晚知分界候选已实现区间编码、确认及完整 T 组提交三个模块，五配置功能验证通过；另保留“运动残差＋动态实值默认状态”新模型候选。两条新颖余量独立评分均为 6/10，不是稳 accept。旧 C1/C2 保留强对照。θ 幅值不丢；生产网络、主稿和 EDA 未改。

**前一轮：先解释构想与 ATLIF 幅值。** [ATLIF 幅值与 idea 概念重评](research/concept_reassessment_20260906/report.html)。ATLIF 按“发放标志＋连续值阈值幅值”解释，不能把输出数值直接当成 1；是否可以折权按路径判断，也不预设每事件任意 int8 载荷。ExSpike 恢复为 C1 重构底稿，运动解释后的计算选择恢复讨论。旧综合 T 分含实现成熟度，不作为概念新颖性淘汰分；该轮暂停设计的阶段已被本轮推进授权取代。

**前一轮：G/H 复核与全新机制筛查。** [报告与交互读图](research/mechanism_rebuild_gh_20260906/report.html) · [完整研究说明](research/mechanism_rebuild_gh_20260906/report-source.md)。复核G/H七叶及并发新增的Card I五叶与集成关系；主攻“晚知阈值下的二值提交”，二值源统计电路列第二，二维运动见证列备选。预声明sample0两个FC1层共82,944,000输出位与封存捕获一致，仅属NumPy float64机会筛查。独立潜力评分5–6/5/5，没有形成两至三条已证明净收益的强机制；没有稳accept、RTL速度或PPA准入。

**前一轮：创新优先逐条复审。** [可检索审阅台账](research/innovation_first_audit_20260906/C1C2创新逐条复审.html) · [研究说明](research/innovation_first_audit_20260906/README.md)。原始42/42文件、620条定位记录已审（含重复、来源与工程史料），旧候选与评分保留；当前优先级以上述新报告为准。

**上一轮包降为实现探索：** [父槽重算与原始权重暂存](research/codex_deep_rebuild_20260906/README.md) 的 CPU 结果原样保留，按用户创新优先要求，不再作为主创新推荐。

下列两个外部包均已逐条复审；先看最新复审结论，再读各包原始假说。冻结 ep34 身份优先。

| Pack | Author | What it is |
|---|---|---|
| Grok Bot (`iscas_ssh`) | 2026-09-05 | C1\*/C2\* remake: OP-STW, HBG-RP **int8 ATLIF payload**, pyramid residual, occlusion |
| Grok 4.6 | 2026-09-05 | New-mechanism survey: **Motion-XOR triple-popcount + dirty-lane score memo + mixed-T binary ATLIF**. C1/C2 demoted. |

**冻结数值语义（Motion C12 ep34）：** 93 个调用出口为 `{0,θ}` 二电平，`z = θ × g`。固定θ可吸入下游线性层权重，以 `W'g` 保持数值；当前 checkpoint 的 θ 为逐模块实数标量。Grok Bot HBG-RP 的任意逐事件 `{g, int8 p}` 属于另一份共设提案。

## Codex read order

1. This file  
2. `research/grok46_20260905/00_READ_THIS_FIRST.md`  
3. `research/grok46_20260905/01_kill_list.md`  
4. `research/grok46_20260905/02_ranked_mechanisms.md`  
5. Optional: Grok Bot `research/04_SYNTHESIS_C1_C2_REMAKE.md` as an **alternate** C1\*/C2\* story (real-valued ATLIF). Do not merge it with Card C without an AEE Pareto.  
6. Stats gates: `research/grok46_20260905/07_next_stats_rtl_gates.md`  
7. Cards: `codex_cards/` — **do not run Card C until the user picks Rank 1+4 and T0–T5 exist**

## Layout

```
ideafromai/
  README.md                          ← start here
  MANIFEST.txt
  INDEX.json
  research/
    00–04, 07–08, m2067_*            ← Grok Bot ANN/SNN/OF/attention/video + C1*C2* synthesis
    grok46_20260905/                 ← Grok 4.6 survey (this session)
  microarch/                         ← Grok Bot C1*/C2* sketches
  plans/                             ← Grok Bot R1+R2 plan
  contracts/                         ← Grok Bot ATLIF int8 draft
  codex_cards/
    CARD_A_OP_STW.md                 ← Grok Bot C1*
    CARD_B_HBG_RP.md                 ← Grok Bot C2* (int8 payload)
    CARD_C_MX3P_DIRTY_SCORE.md       ← Grok 4.6 (binary Motion-XOR island)
```

## Hard rules

- Does **not** modify Codex `hw_autoresearch_nts07` production RTL by itself.
- Does **not** authorize `codex exec resume`, `docs/359` edits, or H81 RTL.
- Do not treat misplaced copies under `hw_autoresearch_nts07/ideasfromai/` as canonical.

## Related isolated RTL (Grok Bot only)

`/home/zhumd/work/sdformer_c1c2star_grokbot/`
