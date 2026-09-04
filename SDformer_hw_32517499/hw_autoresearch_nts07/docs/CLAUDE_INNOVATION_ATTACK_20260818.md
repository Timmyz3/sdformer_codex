# CLAUDE 硬件创新性攻击：389 活门边界枚举 / H82 决策树 / H86 重开条件

日期：2026-08-18。本文为**只读攻击文档**：不改任何 RTL、脚本、文档；不碰
`docs/359`、194436Z 拷包、封存表；不抢 H82 GPU（valid825 ep14 评测正在跑，
本机只读文件，未启动任何进程）。分析基于 docs/389、433、436、437、440、442、
445、439、407、432、411、412、208、262、264、270、272 与
`scripts/h82_multiplicity_free_quotient_model.py` 及
`results/h82_multiplicity_free_quotient_model_20260818/model.json`，
并用纯 CPU 脚本（/tmp，见附录）独立复算了所有闸门数字。

证据分档：[rtl] 冻结 RTL/仿真证据；[prof] 真实 trace 统计；[模型] 分析模型/代理；
[待验证] 尚无证据。本攻击中一切创新分估计均为硬件侧独立尺度、宁低勿高。

---

## 1. 总裁决表

| # | 候选 | 合同/对象变更 | 预闸门 | 创新分（硬件侧） | 裁决 |
|---|---|---|---|---|---|
| M1 | RQTB quotient 跨窗/tile 保持（滚动 quotient band） | 新存储：位置索引 quotient band（240-bit 移窗）；新执行：列移入 15 新 flag | 类别封禁（389/445 已封 score-front CSE） | 2.8–3.0 | **NO-GO**（唯一未攻击过的边界，见 §2.1，理由+可证伪条件写明） |
| M2 | RQTB 投影边界 quotient 保持（K-pair 多播/合并读） | 无 | 20260809 对偶多播 [prof] 命令降幅 ~3% < 15% 线 | — | 已关死，不复开 |
| M3 | RQTB quotient → Class File + 成员差分（Motion 版 ST_PATCH） | 新存储：窗级 Class File + patch RAM；新执行：ST_PATCH 成员 insert/delete | member Jaccard ≥ 0.60（T450 扫描行） | 2.7→3.3 上限 | 与 H86 同一扇门（439 §3），并入 §4 |
| L1 | Local5 FCSR/TCFM5 换对象重写（时间 stencil 等） | 需新算法算子合同（T>2 关系） | 与 Motion T=2 身份冲突；432/439 已断言无新 exact 物化对象 | N/A（无合同）；假设成立 3.0–3.3 | **关死**，硬件等算法侧新合同，不主动立项 |
| H82-A | multiplicity-free quotient 侧车（quotient-gate / denom-only） | 新存储：513-bit occupied bitmap + quotient descriptor(+denom cert)；新执行：ST_CLASSIFY→ST_SHIFTMAX→ST_EXPAND | profile 三闸门：p95 C≤192 且 D/T≤0.60 且 相对 pair-gather 状态≥20%（联合面见 §2.4） | 2.6（437）→过门+sidecar exact 3.0–3.2 →SAIF/PPA 3.5–3.7 →4.0 NO | **CONDITIONAL**，随 AEE 情形分级推进（§3） |
| H86-B | member-delta directory（41% 存储故事） | 新存储：差分 directory（patch RAM）；新执行：ST_PATCH | member J≥0.60 且 edit/rebuild≤0.35 且 class_retention 非退化（§4） | 2.7（442）→过门 3.3 上限 | **重开门**（条件见 §4，当前实测 0.003–0.03，关着） |

结论一句话：**433 合同下 Motion/Local5 硬件侧已无可推的 4.0 候选**；唯一活门
（389）的物理落点是 H82/H86 的 quotient-descriptor 侧车 + member-delta directory，
两者都由 rank-1 的 [R,450] 分数行 dump 一票裁决，与 AEE 情形正交。

---

## 2. (a) Motion/Local5 在 433 合同下的剩余候选攻击

### 2.1 Motion：RQTB quotient 跨 tile/跨 window 的新边界清单

RQTB quotient（score0==score1 → 单 slot {score, tmask}）生于分数前端，死于
gated-K 展开边界（docs/253：quotient 在归一化域取商、分母保留 multiplicity、
输出端重新展开 K0/K1）。逐边界攻击：

| 边界 | 含义 | 状态 |
|---|---|---|
| FIFO→SCS 边界 | quotient 以 tmask 多重度进入 histogram | C7 已保持（`*multiplicity`），无新对象 |
| SCS→目录边界 | 450 带分名单 | H86 TLM 的目标（M3/H86），见 §4 |
| 目录→Shiftmax 边界 | denom 证书 | docs/389/407 已封（leaf-only 注脚） |
| Shiftmax→gated-K 边界 | K0/K1 双 bank 恢复 | 对偶多播已否决（M2，~3%） |
| **窗/行→窗/行边界（跨 tile）** | 同一空间位置 (x,y) 的 quotient 对相邻滑窗不变 | **唯一未被 NO-GO 覆盖的边界**（docs/264 §9 曾预告"exact 跨窗重叠候选"但从未攻击） |

**M1 攻击结论**：480×640、15×15 滑窗下窗口数 466×626≈291,716；每个位置进入
至多 225 个窗口；相等测试朴素逐窗 65.6M 次，滚动 quotient band（240-bit 移位
窗，每列滑入 15 个新 flag）只需 4.4M 次（降 93.3%）——但：
1. 该对象只省分数前端的相等测试，不改 slot FIFO 内容、不改目录、不改投影；
2. H81 实测 equal-pair 98.1%（all）/94.0%（nonempty），band 内容近全 1，是近常数
   存储；docs/262 明确 RQTB 主目标是 descriptor/SCS/K-store 流量，不是 score
   峰值吞吐；
3. 389 §5 与 445 已把 score-front CSE 类别整体封死（MSSB5 只当 RQTB 支撑），
   433 合同要求"新算子合同 + 改硬件存储/执行对象"两条腿，M1 无新算子。

预闸门（可证伪，若将来要重开）：真实 T450 ordered trace 中 score-front
（相等测试+packetize）占 head-row 周期/能量 ≥10%，且 band 方案在相同 FIFO/端口
下净赢 ≥15% 能量——按 docs/262 现有 [prof] 该门不可能过。**裁决 NO-GO，创新
2.8–3.0，不得作为 4.0 门**。

### 2.2 Motion：tile 边界（streaming 帧序列）

下一帧对 (t+1,t+2) 的 quotient 与当前帧对 (t,t+1) 无关（比较对象不同），帧间只
共享 score 值本身，不共享 quotient；"帧间 quotient 保持"不存在，唯一可复用是
score 值（仍是 score-front CSE）。关死。

### 2.3 Local5：FCSR/TCFM5 换对象角度

攻击结论与 docs/432 一致并补充一条：

1. FCSR-RX 已吃掉"rolling inverse-stencil 消除 T450 relation"（docs/208，
   439 §1 明示不许再造一轮）；
2. 拓扑上能 exact 改的四个对象（Q==0 不打分、逆模板编译、相等 gate 合并、
   五色 1RW 无冲突写）全部在用；
3. 唯一未尝试的换对象是**时间 stencil**（T>2 帧间 relation，FCSR ring 变双平面、
   last-consumer 闭式加时间维）——但这要求新算法算子合同，且与 Motion 的
   T=2 quotient 身份直接冲突，硬件侧不得主动立项；
4. TCFM5 双目的地提交（PPDI 型）已在 docs/144-149/431 关死；同端口系数融合
   433 §3 关死。

裁决：Local5 硬件侧在 433 合同下无候选，创新封顶 3.1，后续资源应转完整度
（Amdahl 边界、目标库 DC/SAIF），不转创新。

### 2.4 H82 侧车闸门的联合敏感度（独立复算）

用模型公式独立复算（/tmp 脚本，与 model.json 全一致）：pair-gather 基线
`= 4788 + 9·C` bit（513 occupancy + 9C compact gate + 225×19 pair record）；
quotient-gate `= 513 + 9C + 12·desc`；denom-only `= 513 + 14 + 12·desc`，
desc = 450 − eq。docs/445 锚点 C=128/eq=212 → 5940/4521/3383 复核一致。

闸门是**三腿联合面**，不是三个独立门槛：

| 腿 | 阈值 | 换算 | 松紧 |
|---|---|---|---|
| D/T | ≤0.60 | ⟺ eq≥180/225=0.80 | 松（H81 QF7 非空 equal 0.94） |
| occupied p95 | ≤192 | — | 中（CPU 代理 raw≈195/平滑≈62，恰在线上） |
| state win | ≥20% | 等价 4275−12·desc ≥ 0.2·(4788+9C)；eq=0.94 时 C≤256，eq=0.90 时 C≤189，eq=0.84 时 C≤103；**C=192 时需 eq≥0.91** | **最紧** |

关键风险：**state 腿在 C=192 处要求 pair-equal 率 ≥0.91**；H81 QF7 非空 equal
0.94 只留 3 个点的余量（eq=202/225 时 q_win=19.9% 恰被拒，eq=212/225 时
21.8% 过）。若 H82 513-bin 相等率掉到 0.90，即使 C 与 D/T 都过，侧车仍 DENY。
rank-1 必须同时报告 (occupied 分布, pair-equal 率) 两点，不能只报 occupied。

另外：模型自身周期下界显示侧车与 pair-gather 基线的 cycle 是**平手或略差**
（450+C+desc=880 vs 450+C+PAIRS=867），因此侧车唯一可辩护的生产收益腿是
**能量腿**（目录读路径 read-bit 约 8550→4998，-41.5%），需 SAIF 同端口证明，
不能先写周期主张。

### 2.5 K-active descriptor：侧车周期腿的潜在翻盘点（攻击新增）

模型要求"K-active descriptor 计数"但现网无法测。T450 token K-zero=83.96%、
pair both-K-zero=75.99%（docs/262 [prof]），若 K-active descriptor 占比≈24%
则 desc_eff≈57：展开阶段可跳过双零描述符，cycle 下界 450+192+57=699 对基线 867
（-19.4%），侧车将第一次拿到周期腿。但跳过会破坏 pair-order 隐式寻址（K 地址由
流序隐含），需补 K-active 位图（~450 bit）或 valid-gate，有状态-周期权衡，且
score 相等与 K-zero 相关，**必须实测不能外推**。rank-1 dump 需带 k_mask 键。

---

## 3. (b) H82 硬件决策树

### 3.1 情形划分（valid825 float AEE；锚点：H81=1.3306 [433]，Local5 ep44=1.2819 [425]）

| 情形 | AEE 区间 | 身份 | 硬件推不推 | 推哪个对象 |
|---|---|---|---|---|
| S1 | >1.3306（或 ≤1.3306 但差 <1% 且 selector 未定） | 不敌父本 | **不推**；仅跑一次 CPU profile 闭环归档 | 无 |
| S2 | 1.2819 < AEE ≤ 1.3306 | 优于 H81、劣于 Local5 | **推侧车**（profile 过门才写 sidecar RTL） | quotient-gate 侧车；directory 只测不推 |
| S3 | AEE ≤ 1.2819 | 新 SOTA | **推侧车 + directory** | 侧车优先；member J≥0.60 才动 directory |

关键性质：**侧车的 profile 三闸门与 AEE 完全正交**（bit-exact 构造，只依赖
rank-1 分数行统计）。AEE 只决定"值不值得推"，不决定"闸门过不过"。S1 里即使
profile 全过也不推——无部署身份的算子不能锚定 DATE 主张。

### 3.2 每种情形的推进明细

**S1（不推）**：不写任何 sidecar RTL；创新保持 2.6（437 口径）；H82 降级为
负结果存档（H81 回退线 docs/412 已备）。唯一动作：若算法侧给了 [R,450] dump，
CPU 跑一次 profile 把 513 档统计留档（证明"对象可构建、模型没有骗人"），
不进论文。

**S2（推侧车）**：
1. rank-1 后等算法侧吐 `scores[row,450]` npz（硬件扫描序：同一 head 连续滑窗行；
   Q7 量化按 H82 配置 [-2,2]/step 1/128）；
2. 跑 `scripts/profile_h82_class_file_from_scores.py`（现成，CPU）→ occupied
   分布、pair class equal、class-set Jaccard、member Jaccard；
3. 补一个 CPU 扩展（`profile_h82_quotient_rank1.py`，硬件 agent 写，规格见
   §3.3），吐 p50/p95/max occupied、all/nonempty pair-equal、desc、D/T、
   K-active desc、class_retention、member Jaccard（带幸存集守卫）、
   row_max/denom_shift 分布；
4. 三腿联合面过门 → 写 default-off sidecar RTL（bit-exact vs fused token-major，
   随机反压 0 mismatch）→ 然后才谈生产门：cycle≥10% **或** energy≥15%
   （建议主打能量腿，§2.4）、面积≤10%、Fmax 损失≤5%；
5. directory（ST_PATCH）同一 dump 上 member J≥0.60 才动——按现网代理
   0.003–0.03 基本必死，S2 下直接写 NO-GO 留档，不写 RTL；
6. 创新：profile+sidecar exact → **3.0–3.2**；+SAIF/PPA 同宏过门 → 3.5–3.7；
   4.0 NO。

**S3（推侧车 + directory）**：在 S2 基础上，member J≥0.60 且
edit/rebuild≤0.35（§4）才开 ST_PATCH，41% 存储故事才可碰；4.0 门仍需
"Class File/quotient 由 score 流原生在线构建（单遍 pair-order 流、无重排、
occupancy bitmap 在 ST_SHIFTMAX 前完整、expand 不物化 token-gate）"的 TLM 证明
+ 端到端 EDP 收益不能被融合/CSR/RQTB 解释（445 尺度）。在线构建与 RQTB 的
身份碰撞风险：H82 的 quotient descriptor {class_id,k_mask,pair_last} 与 H67
RQTB packet {score,tmask} 同形，侧车论文必须锚在 **513-bin one-vote 归一化
操作数 + denom certificate**，不许写成"RQTB v2"。

### 3.3 rank-1 统计规格（可复用/需新建）

| 指标 | 来源 | 复用 | 门 |
|---|---|---|---|
| occupied p50/p95/max（513 档） | `profile_h82_class_file_from_scores.py`（需加分布输出） | 现脚本只有 mean/min/max，**需小改** | p95≤192 |
| pair class equal（all / nonempty） | 同上（现脚本给 mean，够用） | 复用 | D/T=1−eq≤0.60；state 腿 eq≥0.91@C=192 |
| desc / D/T | 由 eq 换算 | 复用 | 同上 |
| K-active desc | **需 dump 带 `k_mask` 或 `k_active` 键** | 新建 | 无硬门，用于 §2.5 周期腿 DSE |
| class_retention / member Jaccard（幸存集） | 同上脚本 member_jaccard_surviving | 复用 | J≥0.60 且 retention≥0.5（§4） |
| row_max / denom_shift 分布 | 需从 scores 计算 | 新建 | denom-certificate 容量 |

所有统计 CPU-only，H82 评测结束释放 GPU 后即可跑，无冲突。

---

## 4. (c) H86 member-delta directory 重开条件（可证伪）

现状：442/445 否决（轴是窗内 15 列不是 T450 扫描名单；合成邻窗 446 insert/
446 delete/4 stay；代理 member Jaccard 0.003–0.03 远低于冻结包 0.30）。

重开必须**同一份 rank-1 dump 同时满足四条**（全部可证伪）：

- **J1：surviving member Jaccard ≥ 0.60**，且必须同时报告
  `n_surviving / class_retention ≥ 0.5`。陷阱：参考实现
  `member_jaccard_surviving` 在幸存 class 集为空时返回 1.0，`jaccard` 空集也
  返回 1.0——类名全换时该指标会伪高分，必须用 retention 守卫，否则 442 的
  "轴不对"会以退化形式复活。
- **J2：mean member_edits / 450 ≤ 0.35**（直接数 insert+delete，不靠 J 外推）。
  闭合式（幸存集全覆盖、class 大小均匀时）edits/450 = (1−J)/(1+J)：
  J=0.30→54%、J=0.60→25%、J=0.90→5%。**0.60 门的经济含义是 patch 只写重建
  的 25%**；考虑 patch 控制/地址开销，edit/rebuild≥0.35 时补丁不如重建，
  这是 J≥0.60 之外的独立硬门（现网实测 edit/rebuild≈1.9–2.0，差 5 倍）。
- **J3：轴合同**——delta 必须定义在**硬件扫描相邻 T450 行**（连续滑窗、同
  head），禁止拿窗内 15 列行差充当（442 否决点）。
- **J4：AEE 情形 S2/S3**（H82 部署身份成立；S1 下即使 J 过门也不推）。

四条中任意一条不满足 → 维持 NO-GO（2.7），ST_PATCH 不得写 RTL、不得碰 41%。
四条件全部满足 → 重开为 "directory 后继"，创新上限 3.3（437/440 尺度），
仍非 4.0。

---

## 5. 审计中发现的度量/身份风险清单

1. **侧车 state 腿最紧**（eq≥0.91@C=192），rank-1 报告必须给联合面
   (occupied, eq)，不给单指标（§2.4）。
2. **member Jaccard 空集退化**（§4 J1 守卫）。
3. **H82 磁盘 SHA 漂移**：watcher 显示 frozen=807a50e0（RAM 训练收据有效）、
   disk=66d0a339（437 记录 e22b06bd 之后又被 H83-H86 改写）。rank-1 统计必须
   挂冻结收据，H83-H86 禁止并入 H82 身份（437 §3）。
4. **侧车 vs RQTB 身份碰撞**：descriptor 与 RQTB packet 同形，主张必须锚在
   one-vote + 513-bin + denom certificate（§3.2 S3 末）。
5. **denom-only 的 Fmax 风险**：238 次 exp2 再生在展开路径上，可能加流水级；
   denom-only 是存储冠军（48.1% 降幅）但时序风险最高，RTL 决策须优先
   quotient-gate 版本。
6. **周期腿被模型自己封顶**：侧车 cycle 下界与基线平手，生产主张只能走能量腿
   （§2.4），除非 K-active descriptor 翻盘（§2.5，需实测）。

## 6. 附：本攻击验证过的脚本路径

- `/tmp/attack_verify_20260818.py`（本攻击自建，纯 CPU，不在仓库）：
  - 复核 445 锚点 5940/4521/3383 与 model.json 全一致；
  - 闸门三腿联合面（C/eq 扫描，state 腿最紧，C=192 需 eq≥0.91；扫描表为准，
    推导式上界经文档修正为 eq=0.94→C≤256、eq=0.90→C≤189、eq=0.84→C≤103）；
  - H86 闭合式 edits/450=(1−J)/(1+J) 与阈值换算；
  - Motion 跨窗 band 摊销（65.6M→4.4M 测试，93.3%）与近常数性；
  - D/T≤0.60 ⟺ eq≥0.80 换算；侧车 vs C7 目录 59.5–73.2% 参考账（非强基线）。
- 仓库现成可复用：`scripts/h82_multiplicity_free_quotient_model.py`（[模型]，
  未改）、`scripts/profile_h82_class_file_from_scores.py`（[prof] 钩子，现成，
  需小扩展 p95/分布/K-active）、`scripts/h82_c81_member_tv_contract.py`（[模型]
  门 0.60 依据）、`scripts/h86_window_member_delta_reference.py`（TLM，未改）。
- 本攻击未运行任何仓库脚本（只读文件 + /tmp 独立复算），未动 GPU、未改仓库。
