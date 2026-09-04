# 四线全实验对比：谁更适合写 DATE 论文

Date: 2026-08-17. Seed0 only. DSEC = local valid825, not official hidden test.
Energy = spike proxy. H67 vs H81 is a recipe-level control, not step-paired.

**结论先说：** DATE 这篇该写 **H67 Motion-TTX ep35**。DSEC 主增益在 NB0→H81（统一 TTX），Motion 是小而有用的 indoor 先验，Local5 是精度扩展不能当主模型。Local5-FT 只能单独开 transfer 行。

## 1. 四条线是什么

| 线 | 神经元 | Attention | 时间先验 | 空间候选 | DSEC rank-1 | 同协议 MVSEC | 同 ckpt RTL |
|---|---|---|---|---|---|---|---|
| NB0 | PSN | 原 SDSA | 无 | 原窗口 | ep29 | scratch day2 | 无 |
| H81 | binary ATLIF | all12 TTX | 无 | self | ep29 | scratch day2 | 无 |
| H67 | binary ATLIF | all12 TTX | Motion-XOR α=0.25 | self | ep35 | scratch day2 | 有，绑 ep35 |
| Local5 | binary ATLIF | all12 Local-TTX | 无 | self+4 axial | ep44 | scratch day2 | ep44 component RTL 已重绑（2026-08-15） |
| Local5-FT | 同 Local5 | 同 Local5 | 无 | self+4 | DSEC ep44 | **DSEC 预训练 + day2 FT** | 无 |

不要混：Motion+Local5、MDR/day2/transfer 表、把 Local5-FT 写进 scratch 表。

## 2. DSEC valid825（主表）

| Method | Epoch | AEE | AAE-2D | AE-3D | Fl-all (%) | Spikes (G) | vs NB0 AEE | vs NB0 spikes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NB0 | 29 | 1.4454 | 6.5128 | 6.1803 | 7.9323 | 126.12 | ref | ref |
| H81 TTX | 29 | 1.3306 | 5.9692 | 5.6726 | 6.4310 | **80.90** | −7.94% | **−35.85%** |
| H67 Motion | 35 | 1.3297 | 5.9004 | 5.6509 | 6.4279 | 82.11 | −8.00% | −34.89% |
| Local5 | **44** | **1.2819** | **5.8498** | **5.5087** | **6.0210** | 85.24 | **−11.31%** | −32.41% |

拆开看：

- **几乎全部 DSEC 增益是 TTX（NB0→H81）**，不是 Motion。
- H67 vs H81：AEE 只好 **0.069%**，spikes 反而多 1.49%。Motion 在 DSEC 上不是主贡献。
- Local5 比 H81 再降 **3.66% AEE**，但 spikes 回到 85.2G，而且多训到 ep44。

## 3. 等预算（防审稿人说“你多训了”）

| Method | ~30 | ~35 | ~40 | ~50 | rank-1 | last−best |
|---|---:|---:|---:|---:|---:|---:|
| NB0 | **1.4454** (29) | 1.4584 (34) | 1.4549 (39) | — | ep29 | +0.66% |
| H81 | **1.3306** (29) | 1.3475 (34) | 1.3438 (39) | — | ep29 | +0.99% |
| H67 | 1.3387 (30) | **1.3297** (35) | 1.3434 (40) | — | ep35 | +1.03% |
| Local5 | 1.3286 (29) | 1.3355 (34) | 1.3153 (39) | 1.2982 (49) | **ep44 / 1.2819** | +1.27% |

等预算 40：Local5 1.3153 仍好于 H81/H67 ~1.34。所以 Local5 的 DSEC 优势 **不完全是多训出来的**。ep44 的 component RTL 已在 8 月 15 日重绑，不再是“数字和芯片不是同一个 ckpt”。

四条线在 40 附近都已过最优点（last−best 0.7–1.3%），没有一条还在明显爬坡。

## 4. MVSEC day2-only scratch（次数据集，同协议）

Full-sequence AEE：

| Method | OD1 | IF1 | IF2 | IF3 | Macro | Spikes (G) | 四序列都 < NB0 |
|---|---:|---:|---:|---:|---:|---:|---|
| NB0 | 0.8450 | 1.5998 | 2.7536 | 2.1106 | 1.8273 | 251.5 | reference |
| H67 | **0.8201** | **1.5868** | **2.6258** | **2.0357** | **1.7671** | **140.7** | **yes** |
| H81 | 0.8205 | **1.6248** | 2.6670 | 2.0581 | 1.7926 | 141.1 | **no（IF1）** |
| Local5 scratch | 0.8414 | **1.6282** | 2.6679 | 2.0669 | 1.8011 | 141.4 | **no（IF1）** |

H67 是唯一过预注册门的 scratch 模型。Motion 在 DSEC 上几乎没贡献，但在 MVSEC indoor_flying1 上是 **H81/Local5 翻车、H67 过门** 的那一点。

Fixed800 与 full 同序，不改结论。

## 5. Local5-FT：好看但不能当主线

| | Macro AEE | OD1 | IF1 | IF2 | IF3 | Spikes (G) | 协议 |
|---|---:|---:|---:|---:|---:|---:|---|
| Local5-FT | **1.6686** | **0.8070** | **1.4811** | **2.4704** | **1.9159** | 200.4 | DSEC ep44 + day2 FT |

四序列都打赢 NB0，但：

- 不是 scratch，不能进 H67/H81/Local5 同协议表；
- spikes 比 H67 高 **+42%**；
- 没有 H67 ep35 或 Local5 ep29 的 RTL 继承权。

写论文时最多作为 transfer 附表。

## 6. 机制消融（最小因果）

| ID | Binary ATLIF | all12 TTX | Motion-XOR | Local5 拓扑 | DSEC AEE | MVSEC 四序列门 |
|---|---|---|---|---|---:|---|
| A0 NB0 | no | no | no | no | 1.4454 | reference |
| A1 H81 | yes | yes | no | no | 1.3306 | fail IF1 |
| A2 H67 | yes | yes | yes | no | 1.3297 | **pass** |
| A3 Local5 | yes | yes | no | yes | **1.2819** | fail IF1 |

审稿人会问的三句话，现在都能答：

1. **主增益是什么？** 统一 TTX（H81），不是 Motion。
2. **那为什么主线带 Motion？** 因为同协议 MVSEC 上只有它过四序列门，而且 RTL 绑在这个 ckpt。
3. **Local5 不是更准吗？** 是。同 ckpt component RTL 也已经有了。它仍然过不了同协议 MVSEC 四序列门，端到端相对 H67 也还没有 10% 能量/EDP 赢面。

## 7. 加载审计（附录已出）

| ID | ATLIF | Shiftmax | Overlay | Missing / unexpected | Ckpt SHA | RTL |
|---|---:|---:|---:|---|---|---|
| NB0 ep29 | 0 | 0 | 0 | 0 / 0 | `7e8d524e0784` | none |
| H81 ep29 | 105 | 12 | 210 | 0 / 0 | `8825c933e491` | none |
| H67 ep35 | 105 | 12 | 210 | 0 / 0 | `4f33e086070b` | **同 ckpt component RTL** |
| Local5 ep44 | 105 | 12 | 210 | 0 / 0 | `19820bec07cc` | **同 ckpt component RTL（8-15 重绑 PASS）** |
| Local5 ep29 | 105 | 12 | 210 | 0 / 0 | `6e0e92a56229` | 旧锚点，已被 ep44 收据取代 |

H67 identity contract 的 SHA 检查仍 bind。三条 TTX 线加载都是 overlay=210 missing=0 unexpected=0。

## 8. 按 DATE 权重谁该上主模型

权重按这篇会怎么被审，而不是按“谁 AEE 最低”：

| 项 | 权重 | NB0 | H81 | H67 | Local5 |
|---|---:|---:|---:|---:|---:|
| DSEC AEE | 20 | 0 | 16.2 | 16.4 | **20** |
| DSEC spikes | 15 | 0 | **15** | 14.5 | 13 |
| 等预算 40 | 10 | 0 | 7.5 | 7.5 | **10** |
| 同协议 MVSEC | 25 | 6 | 10 | **25** | 10 |
| 同 ckpt RTL | 20 | 0 | 0 | **20** | 16 |
| 机制/审稿风险 | 10 | 6 | **10** | 8 | 5 |
| **合计** | 100 | 12.0 | 58.7 | **91.4** | 74.0 |

H67 拉开的不是 DSEC 精度，是 **MVSEC 门 + 硬件 provenance**。这正是 DATE 软硬件协同该比的两件事。

如果换会场，排名会变：

| 如果论文其实是… | 该写谁 | 为什么不行当 DATE 主线 |
|---|---|---|
| DATE 软硬件协同 | **H67** | — |
| 只比 DSEC 精度 | Local5 | RTL 不在 rank-1；scratch MVSEC 不过门 |
| 只讲机制干净 | H81 | 没有 RTL；IF1 失败；DSEC 上 Motion 差 0.07% 讲不圆“必须带 Motion”，但反过来 H81 也讲不圆“为什么室内泛化掉了” |
| 只追 MVSEC 绝对数 | Local5-FT | 协议不同、spikes +42%、无 RTL |

## 9. 审稿人最可能打的点，和现在怎么写

**会打、但挡得住**

- “Motion 在 DSEC 上没贡献。”
  承认。写成 TTX 主贡献 + Motion 作为 indoor 先验。H81 必须留在主消融表。
- “Local5 更准，为什么不是主模型。”
  等预算 40 也更准。用 MVSEC 四序列门 + rank-1/RTL 不一致挡。不要假装 Local5 没赢 DSEC。
- “H67 vs H81 不是 step-paired。”
  已经写进 fairness 收据。只称 recipe-level control。
- “valid825 不是 hidden test。”
  只比内部相对差，不拿 NB0 AE-3D 6.18 去对公开 4.871。

**现在不要写**

- 不要说 Motion 是 DSEC 的主要精度来源。
- 不要把 Local5-FT 1.6686 和 H67 1.7671 放同一张 scratch 表。
- 不要把 Local5 ep44 数字绑到 ep29 RTL。
- 不要从单 seed 伪造 std。
- 不要把 spike 能量代理写成芯片实测。

## 10. Table G：事件密度分层（已挂上）

冻结 voxel-L1 四分位，四线 rank-1 重评 AEE 与原 valid825 差 ~1e-9。

| Quartile | NB0 AEE / Fl | H81 | H67 | Local5 |
|---|---:|---:|---:|---:|
| Q1 低密度 (207) | 1.374 / 6.22 | 1.262 / 4.99 | 1.289 / 5.06 | **1.176 / 4.52** |
| Q2 (206) | 1.361 / 6.58 | 1.262 / 5.57 | 1.247 / 5.42 | **1.239 / 5.32** |
| Q3 (206) | 1.380 / 7.12 | 1.284 / 6.02 | 1.272 / 5.88 | **1.243 / 5.62** |
| Q4 高密度 (206) | 1.666 / 11.82 | 1.514 / 9.15 | 1.511 / 9.36 | **1.471 / 8.63** |

读法：

- 四条线都在 Q4 变差。高密度/大运动才是 DSEC 难点，不是低密度。
- Local5 **每个四分位都最好**。相对 H81 的额外增益主要在 Q1（−6.9% AEE），稀疏时五邻域拓扑最有用。
- H67 相对 H81：Q1 更差（+2.1%），Q2–Q4 略好。Motion 不是密度鲁棒的 DSEC 主增益，和主表 0.069% 的结论一致。
- 这不改变主线：Local5 仍赢 DSEC 分层，同 ckpt RTL 也已经有了；它仍输同协议 MVSEC 四序列门，也没有相对 H67 的系统能量赢面。

## 11. 已补完 / 故意没做

已经补上：

- Table G `DSEC_DENSITY_QUARTILE_TABLE_G_20260817.json`
- 加载审计 `DATE_LOAD_AUDIT_APPENDIX_20260817.md`
- 四线总账 `DATE_FOUR_LINE_LEDGER_20260817.json`
- 图：`figures/date_four_line_20260817/`
  Pareto、预算曲线、MVSEC 四序列、DATE 记分、Table G 柱图、4 张冻结密度帧误差图

故意不做：

- seed1/2 训练
- MDR 再训（已有 TTX-MDR ep20 比 baseline 差，且慢）
- Motion+Local5 混合
- 仅因 ep44 已重绑 RTL 就把 Local5 升成 DATE 主线（MVSEC 门和 vs-H67 系统账还没过）

**一句话：写 H67；用 H81 讲清楚 TTX 才是 DSEC 主增益；用 Local5 做精度/拓扑扩展；Local5-FT 单独放 transfer。**
