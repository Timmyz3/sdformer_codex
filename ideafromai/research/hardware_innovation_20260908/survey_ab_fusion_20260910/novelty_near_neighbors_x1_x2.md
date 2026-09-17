# Codex 加强创新：X1 / X2 近邻 + 杀门（调研页）

日期：2026-09-14  
配合：管理 `X1/X2/X3` 建议；玩具 RTL ROUND1–3 负结果；`fusion_killcards_*.md`。  
用途：贴进 Codex / 落 `CODEX_NOVELTY_PUSH` 附录。≠ Stage B 裁决；≠ 标题已成立。

---

## 共用纪律

- 借入 A ≠ 自称 X；底座增益记 A，禁止进标题句  
- 指标必须含**物理读/请求/并集 CR/job 周期**；禁止只报 MAC、W²、逻辑边  
- 同资源：same-port / 同状态 / 同背压；同训练预算给强对照  
- 不复活：位平面、分组 DA、列字典、阶段错位、并集满读删字、长背压打平旁路、免训 mask  

---

## X1 — 训练改变物理并集读（A 轴·真稀疏跳过）

### 待证 X（一句话）
在**门 + PED 双消费者**下，用训练直接压低「并集物理源字/请求」，且该压低不能被同预算普通 N:M / 逐 rank / HiNM 式结构剪枝复现。

### 最近邻（必须同页打过）

| 近邻 | 借入什么 (A) | 我们差分必须钉在哪 | 易假赢 |
|---|---|---|---|
| HiNM / Gyro / 分组 Gram 剪枝 | 结构组、恢复训练 | 优化对象是**并集物理读**，不是单层 W 块敏感度 | 只报参数↓ |
| VENOM / CRISP / DepGraph | 依赖图、通道/组耦合 | 消费者是非因果 T10×**双 PED+门**，不是单输出 CE | 换损失名 |
| SparseGPT / Wanda / 2:4·N:M | 同预算稀疏格式 | 对照必须**同宽同端口同训练步** | 格式本身当 X |
| 本仓整组剪枝 / 时间保持 | 有损候选 | Codex 已见对普通 rank 仅 ~1–2% | 对弱控制好看 |
| ROUND1 cand5 共同删字 | 负结果 | 并集满读 → 无事务差分 | 逻辑稀疏率 |

### 强对照清单
1. ordinary **逐 rank / hidden50** 同预算  
2. 固定 **2:4 或 3:4** 同宽同端口  
3. HiNM（或等价结构组）+ 同等恢复  
4. 同 lifting/源图 **无**并集目标的端点  
5. 免训活动 mask（应不胜或打平）

### 物理事务指标（必报）
并集请求字数、唯一物理源字、满读比例、双消费者各自到达后仍读次数、job 周期（若有模型）。

### 一票否决 → 停该布局
- 并集物理读不降，或只降 W²/逻辑边  
- AEE 不胜同预算强对照（NB0/相对门按当时合同）  
- 只对弱控制赢，对普通 rank / N:M 增量 <5% 且新颖性自评 ≤3  
- 为过门改同端口公平或拆双消费者义务  

### 与当前工作怎么融
叠在 **已编译源读集合之后**；先纸面/捕获证并集变稀，再 RTL。可与 lifting 后处理叠，但 **X 句不得依赖「lifting 二字」**——差分是并集训练目标。

---

## X2 — 稀疏保持型分解（B 轴·真分解）

### 待证 X（一句话）
分解后**第一段输出保持小整数/{0,±1,计数}（或可 bit-skip 的离散态）**，第二段禁止默认可乘满 dense；付清 gather/选择/状态后，周期仍优于**原二电平 bit-skip / 普通 2:4·3:4**，而不是优于稠密 MAC 童话。

### 最近邻（必须同页打过）

| 近邻 | 借入什么 (A) | 我们差分必须钉在哪 | 易假赢 |
|---|---|---|---|
| SVD / Tucker / CP 低秩卷积 | 低秩因式 | 中间量必须**稀疏可跳**；对标 bit-skip 不是 dense MAC | 只报 MAC↓ |
| PoT / DeepShift / 符号·移位编码 | 小整数基底 | 共享基底 + **按真实 K/双消费者选例外** | 贴 PoT 名 |
| 字典 / DA / 位平面（Codex 已慢） | 负结果边界 | 付清 walker/选择后更慢 → 布局已停 | 复活三刀 |
| 普通 2:4 / 3:4 / AAC bit-skip | 强执行对照 | 同端口同宽；分解链总服务 | 只比 dense |
| Prosperity / 公共子式 | 底座 A | 可作公共 walker，**禁止**写进标题 X | 底座当 X |

### 强对照清单
1. **原二电平 / bit-skip AAC**（主对照）  
2. ordinary **2:4 与 3:4** 同宽同端口  
3. dense-zero 上界（只作悲观参考，不当「已赢」）  
4. 同低秩但中间 **dense 可乘满**（应暴露连续中间量税）  
5. 纯 PoT 系数、无共享基底/例外调度  

### 物理事务指标（必报）
第一段非零/离散态率、第二段有效可乘密度、gather/index 字节、例外率与例外路径周期、总 job 周期 vs 2:4/3:4。

### 一票否决 → 停该布局
- 付清 gather/选择/状态后周期 ≥ 普通 2:4/3:4 或原 bit-skip  
- 质量不胜 NB0 同协议  
- 中间默认 dense 可乘满且未入账  
- 以 SVD/Tucker/Prosperity **名字**冒充 X  

### 与当前工作怎么融
优先挂 **未改/仍贵的 patch r0.conv***；小秩 INT8 完整执行必须带「稀疏保持」约束。Prosperity 只许进 A。整数基/H8-pair 已慢于 dense-zero/3:4 的布局 **不复活**，除非例外率与 walker 费用被显式重设计。

---

## X3（本页不展开，一句话锚）
双消费者非对称退休：只退休门与 PED **都**不再需要的物理字；串行 early-release 已杀。近邻：last-use RF、Avalanche 完成回收——借入≠X。

---

## 给 Codex 的粘贴摘要

```text
X1: train to cut PHYSICAL union reads (gate+PED). Beat same-budget N:M / per-rank / HiNM.
    Kill if union full, only W²↓, or <5% vs strong controls.
X2: sparsity-preserving factorization; stage-1 stays skippable discrete; beat bit-skip / 2:4–3:4,
    NOT dense MAC. Kill if gather/select tax ≥ savings or dense mid unchecked.
Do NOT revive: bitplane, group DA, col-dict, phase-skew, union-full delete, BP-flat bypass.
Score ≤3 and gain <5% → stop layout. Base A gains ≠ title X.
```

