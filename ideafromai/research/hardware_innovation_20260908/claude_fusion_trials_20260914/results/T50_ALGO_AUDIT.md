# T50 报告：算法侧代码审计 —— 两个 idea 各自出什么问题，外加目录整理

日期：2026-09-17　　state：**审计闭合（全部结论可复现）；三元重训 cell A 仍在跑**

一句话：**「ATLIF + motion 注意力」这两个 idea 的骨架都在，但 ep34 这个 checkpoint 里
它们各自的核心机制都没在工作** —— ATLIF 侧 **105 个模块里有 24 个是结构性死模块**
（12 个根本没被调用、12 个输出被丢弃），motion 注意力侧 **`h60` 的打分式在二值输入下
退化成 `1.02·overlap − 0.02·|K| + 0.125·Hamming`**，
文档里写的「同静默给 alpha 奖励、反极性给惩罚、SC 有符号共识」**在数学上一条都不成立**。

---

## 1. 目录整理（现状 + 建议）

### 1.1 现状

`claude_fusion_trials_20260914/`（顶层 42 个 `.py`、31 个子目录、总 1.2 GB）：

| 类别 | 内容 | 体积 | 判断 |
|---|---|---:|---|
| **RTL / PPA 工程** | `t24_fpga/` `t15_synth/` `t5_rtl/` `t10_rtl/` `t25_k4_gate/` `t26_width/` `t39_sglr_rtl/` `t47_fused/` `t47_stim/` `t41_fused/` `t40_arch/` | **1.06 GB（88%）** | 全是 Vivado 中间产物，只有 `*_REPORT.md` + `t*.json` 是结论 |
| **探测脚本（C1 线）** | `t1*` … `t37b*` 共 30 个目录 + 42 个顶层 `.py` | ~150 MB | 多数已被报告吸收，属于「可归档」 |
| **报告正文** | `results/*.md`（`T1`…`T49` 系列 + `results/t49_*`） | 112 MB 里的一小部分 | **只有这些是活文档** |
| **T49 新线** | `t49_*.py` + `results/t49_*` | 小 | 活 |

问题具体在哪：**结论（`.md`/`.json`）和工程产物（`*.dcp`/`*.jou`/`*.log`/`*_sim/`）
混在同一个 `results/` 里**，而 `results/` 自己有 129 个条目。

### 1.2 建议结构（**未执行**，等你点头）

```
claude_fusion_trials_20260914/
├── reports/            # 只放 T*_REPORT.md + 汇总 md，一个 flat 目录
├── data/               # 所有 t*_*.json / *.npz（被 report 引用的原始量）
├── probes/             # 所有 t*.py（按 T 编号自然排序）
├── rtl/                # t*_rtl / t*_fpga / t*_synth（.dcp/.jou/.log 一律 gitignore）
└── archive_20260917/   # 已关闭轴（T45c 判负）的工程产物
```

判断依据：**只有 `reports/*.md` 会被引用进论文**，`rtl/` 只需要留 Vivado 报告
（`*_report.txt`/`.rpt`）与 GDS 无关的 util 数字；`t24_fpga/` 那 989 MB 里绝大多数是
`*.dcp`/`*.runs` 中间态，删掉不损失任何可复现性（脚本能重跑）。
**要不要执行，你说一句就行。**

---

## 2. idea 盘点：确实只有两个吗

**是的，就两个**，而且从代码结构上可以精确枚举：

| # | idea | 代码落点 | ep34 是否在工作 |
|---|---|---|---|
| 1 | **ATLIF 自适应阈值神经元** | `overlay/.../atlif_ternary_psn/`（105 个模块，105 个 `thresh`） | ❌ 自适应关着 + **24/105 结构性死** |
| 2 | **motion 注意力（`h60`）** | `overlay/.../bsa_attention.py`（12 个 attention block） | ⚠️ 骨架在，**但打分式退化成 overlap + Hamming** |

**除了这两个，其余全是这两个 idea 的「外壳」**，不是独立 idea：
- 「signed consensus / bipolar / dual-rail / FAPS / class-file / match-code」→ 都是
  **idea 2 的不同打分实现分支**（`bsa_attention.py` 里 70+ 个 `mode`），同一个
  `_qk_shiftmax_gate_forward` 的 `elif` 链，**一次只走一条**；
- 「三值 / asymmetric / 量化 / importance / temporal-factor」→ 都是 **idea 1 的
  `threshold_mode` 变体**，ep34 一条都没用（见 §3.6）；
- 「polyphase / bit-plane / SGLR / C1」→ 是**硬件侧**映射与供给，不是算法 idea。

---

## 3. idea 1（ATLIF）：查出的问题

### 3.1 【新】105 个模块里 **24 个是结构性死模块** —— 三分法精确闭合

用两个独立口径交叉验证（`t49_audit_probe.py` + `spike_profile.json`）：

| 口径 | 数量 | 来源 |
|---|---:|---|
| 装上的 ATLIF 模块（checkpoint 里 `.thresh` 键） | **105** | `checkpoint_epoch34.pth` |
| 前向里被调用过的（profile 里有 firing rate） | **93** | `m2041/spike_profile.json` |
| 优化器里有 state（收到过梯度）的 | **81** | `t49_optimizer_probe.py` |
| `m2270` 独立探针 | `installed 105 / invoked 93 / result_consumed 81` | `m2270_atlif_semantics_probe` |

**105 − 93 = 12，93 − 81 = 12，正好凑出那 24 个缺席槽位，而且两组名字就是它们：**

| 组 | 数量 | 为什么死 |
|---|---:|---|
| `attn.sn2_q` | **12** | **h60 根本没调用它。** 基础模型 `Spiking_swin_transformer3D.py:693` 有 `att_token = self.sn2_q(att_token)`，但 overlay 的 `_qk_shiftmax_gate_forward` **整个替换了 forward**（`bsa_attention.py:6659` `module.forward = MethodType(...)`），里面**没有这一行**。config 的 `sn2q_binary` 组却把这 12 条路径逐条列了出来 ⇒ **配了个永远不生效的组**。 |
| `attn.attn_sn` | **12** | **输出被丢弃。** overlay 尾部和基础模型一样：`attn = self.attn_sn(x)` 之后 `x = self.proj(x)`——**proj 吃的是神经元之前的 `x`**，而 `attn_sn` 的输出作为第二个返回值 `attn` 返回；调用方 `SSA()` 只用了第一个返回值，第二个只在 `get_lst_block_attention_scores()` 这个**纯可视化方法**里才被取用。⇒ 12 个 block 的 `attn_sn` **spike 数精确为 0**（profile 实测，12/12），参数拿不到梯度。 |

**⚠ 这修正了 T49 §3.1 的归因**：报告当时把「24 个没 state」算进「代理梯度数值上死掉」。
真相是**两种死法叠加**：

- **24 个是结构性死**（12 没被调用 + 12 输出被丢）；
- 真正的**数值死**只发生在剩下的 **81** 个身上（`exp_avg ∈ [2.2e-17, 8.3e-12]`
  而 backbone 中位数 `5.8e-08`）——那部分结论不变。

### 3.2 【新】`attn_sn` 死掉的连带后果：**注意力输出路径根本没有脉冲稀疏化**

`proj(x)` 里 `x = attn.reshape(...)` = `k_orig.mul(gate)`，而 `gate = shiftmax(scores)`
是**连续实数**（∈ (0,1]）。所以：

- `attn_sn` 不是「被稀疏化后送 proj」，而是**整条旁路被绕过**；
- **`proj` 是一个作用在连续实数上的稠密 Linear** —— 不是二值 select、不是 0/θ 乘法。

⇒ 对 T47「二值操作数 ⇒ 乘法退化成 select ⇒ 0 DSP」这条结论，**必须限定作用域**：
它成立的前提是操作数 ∈ {0, θ}，而 attention 的 `proj` 不满足
（`attn` 支撑集 ⊆ `support(k_orig)` 仍然稀疏、zero-skipping 仍有效，但**乘数是实数**）。
**这一条要单独核一遍 T47/T48 的账本有没有把 `attn.proj` 算进去。**

### 3.3 `threshold_eta = 0`：自适应关着（T49 已证，此处只补出处）

`threshold_eta: 0.0` ⇒ `module.sp = 0.0` ⇒ `update_value ≡ 0` ⇒ `threshold_update()` no-op。
`activity_eta: 0.0`、`target_rate: null`、`target_rate_eta: 0.0` 同样为 0。
**产物出处**：`threshold_lr_scale: 50000.0` 这个名字破案了 ——
H3 的第一个 config 叫 **`diag_allqk_lrs50k_threshold_only_40.yml`**（"diagnostic"），
然后它一路**原样复制**进了 H4 / H6 / H8 / H9。

### 3.4 `threshold_lr_scale: 50000.0` 是个诊断残留，而且它是**第二个 LR 旋钮**

θ 有**两条互不相干的更新通路**，配置里却只有一个看起来像 LR 的项：

| 通路 | 公式 | ep34 实际步长 |
|---|---|---|
| 梯度（optimizer 组 `atlif_threshold`） | `AdamW(lr=5e-6)` | ~1e-11/步（§T49 3.1 已算） |
| 手动生长（`threshold_update`） | `update_tensor * (threshold_base_lr × threshold_lr_scale)` | `3e-6 × 50000 = 0.15` |

也就是说：**手动生长那条路的名义步长是梯度路径的 1.5e7 倍**。ep34 因为
`threshold_eta=0` ⇒ `update_tensor=0` 才没出事；**一旦有人把 `threshold_eta` 打开
而不动 `threshold_lr_scale`，θ 会以 0.15/step 的量级飞掉**。
这正好解释了为什么 H 系列里 `threshold_eta` 都配得极小（0.00065–0.001）——
是在**补偿**这个 5e4 的放大。

### 3.5 `threshold_freeze_after_step: 1224` 在源 run 里冻结了后 2/3

源 run `dsec_c12_alpha0125_ep29_resume5_20260830` 的 `train.log` 里最大 step = **3660**
（915 条 `[H9] step N update`）。`1224 / 3660 = 33%` ⇒ **后面 2/3 的训练里手动生长是冻结的**。
（次因：即使 `threshold_eta > 0`，这条通路也只在前 1/3 有效。）

### 3.6 三值 / asymmetric / quantile / importance / temporal-factor 全是**未使用的代码**

`atlif_ternary_psn.py` 里 `threshold_mode` 至少有
`official_atlif` / `asymmetric_scale` / `symmetric_bsa_tsn` / `symmetric_target_rate`，
外加 `center_mode` 的 center_bias / center_calibrated / quantile / importance /
temporal_factorized 分支。ep34 只走 `official_atlif` + `center_mode: zero`，
**其余分支一次都没被执行**（三元重训的 summary 实测：
`quantile_modules: 0`、`importance_modules: 0`、`temporal_factorized_modules: 0`、
`symmetric_bsa_tsn_modules: 0`、`official_atlif_modules: 105`）。

⇒ **模块名 `ATLIFTernaryPSN` 对 ep34 是误导性命名**：它输出的是
`(h ≥ θ)·θ ∈ {0, +θ}`，**严格非负二值**，没有任何 ternary。

---

## 4. idea 2（motion 注意力 / `h60`）：查出的问题

ep34 的实际配置：`mode: h60`、`alpha0: 0.02`、`mismatch_penalty: 0.0`、
`single_active_penalty: 0.0`、`consensus_score_norm: head_dim`、`score_scale: 1.0`、
`center_scores: true`、`preserve_mean: true`、`k_magnitude_alpha: 0.0`、
`bipolar_mu: 0.0`、`sc_mu_schedule_enabled: false`、`binary_motion_xor_alpha: 0.125`、
`event_temperature_enabled` 未设（默认 `False`）。

### 4.1 【已实测】打分式精确退化成三项

把 `_ternary_alpha_xnor_token_scores` 的表达式展开（`q,k` 来自 `sn_q`/`sn_k`，
即 ATLIF 输出 ∈ `{0, θ}`，θ=1）：

```
same_nonzero = |Q_act ∩ K_act|           (overlap)
same_zero    = D − |Q_act| − |K_act| + overlap
opposite     = ∅                          ← 空集
single_active= |Q_act| + |K_act| − 2·overlap

score = overlap + 0.02·same_zero − 0.0·opposite − 0.0·single_active
      = 1.02·overlap − 0.02·|K_act| + (0.02·D − 0.02·|Q_act|)
                                    └──────── 只与 Q 有关 ⇒ 整行常数 ────────┘
      + 0.125 · Hamming(K_t, K_{1−t})
```

**sd5ai 实测复核**（`t49_score_probe.py`，D=32, ρ=0.057）：

| 断言 | 实测 |
|---|---|
| 二值下 `opposite` 恒为空集 | `\|O\| mean = 0.000`（sign_ste 与 binary_event_ste 都是 0） |
| 三值下 `opposite` 才活过来 | `\|O\| mean = 0.062` |
| 上式与真实实现逐位相等 | `maxdev = 3.7e-9`（float32 噪声级） |

**⇒ 文档里承诺的三件事，一件都没发生：**

| 文档写的（`_ternary_alpha_xnor_token_scores` docstring） | 实际 |
|---|---|
| "same nonzero polarity is a strong match" | ✅ 但退化成**不带极性的 overlap** |
| "same silence gets the paper's small alpha reward" | ⚠️ **2/3 是惰性的**：`alpha0` 里 `+0.02·overlap` 与 `+0.02·D−0.02·\|Q\|` 一个被并进 overlap、一个在**行内是常数**（`center_scores` 与 shiftmax 都会精确消掉常数项）。真正存活的只有 **`−0.02·\|K_act\|`** |
| "opposite polarity is penalized because it is harmful for flow direction" | ❌ **双重失效**：定义域上没有反极性（`∅`），系数还是 0（`mismatch_penalty: 0.0`） |

`alpha0` 整体到底有多大用？实测：**`alpha0: 0.02 → 0` 只让 gate 变 rel 7.7e-4**。

### 4.2 `center_scores: true` 对 gate **零影响**（实测 0.0）

`center_scores` 做的事是 `scores − scores.mean(dim=2)`，其后紧跟的第一个算子
是 `shiftmax`，而 shiftmax 第一步就是 `scores − scores.amax(dim)` ⇒ **平移不变**。
中间的 `_event_selective_temperature`（默认关）与 `_apply_hardware_score_quant`
（`hardware_quant_enabled: false`）都是恒等 ⇒ **`center_scores` 在这份 config 里
精确地什么也没做**（实测 `shiftmax(centered) − shiftmax(uncentered) = 0.000e+00`）。

### 4.3 `sc_scores` 被乘了 **0** —— 整个 signed-consensus 分支是死算力

h60 的核心一行：`scores = tx_scores + mu * sc_scores`，其中
`mu = _scheduled_bipolar_mu(...)` = `bipolar_mu` = **0.0**（`sc_mu_schedule_enabled: false`）。

⇒ `_signed_consensus_token_scores()` 被完整计算（一次全 head_dim 的 popcount），
然后**乘以 0**。前向白算、梯度贡献精确为 0。
**「signed / bipolar / consensus」这套命名在这个 checkpoint 上没有任何作用。**

（顺带：即使 `mu > 0`，`_signed_consensus_token_scores` 在二值事件下的
`(q_event * k_event).sum(-1)` 也是**非负**的 ⇒ `_sc_agree_disagree_gate` 里的
`disagree = relu(−score)` 恒为 0 ⇒ 那条「signed gate」同样是无符号的。）

### 4.4 motion 项是**与 query 无关的偏置**，不是 Q–K 相似度

`_binary_temporal_k_xor_popcount` 返回的是 K 在两个时间片上的
**Hamming 距离**，`reshape` 成 `[B,H,T·N,1]` 后**逐 token 加到 score 上**。
只与 K 有关、与 Q 完全无关 ⇒ 它在 shiftmax 里等价于
**「给跨时间变化大的 key 加一个偏置」**，无法表达「attend 到与 query 的变化匹配的 key」。
这是 live 的、非标准的唯一一项，**语义是否符合光流的意图需要你确认**。

### 4.5 【真 bug】`_binary_event_ste` 被定义了**两次**，后者静默劫持前者

| 定义位置 | 实现 | 声明的梯度 |
|---|---|---|
| `bsa_attention.py:3143` | `hard = x.gt(0); return (hard − x).detach() + x` | **恒等**（docstring 未写，但这是 identity STE） |
| `bsa_attention.py:4800` | `hard = value.gt(0); proxy = value.clamp(0,1); return hard + proxy − proxy.detach()` | **clamp 代理**（`d/dx ∈ [0,1]`，区间外为 0） |

Python 在**调用时**查 globals ⇒ **4800 的那个赢**，于是 §4.4 的 motion 项
（调用点在 1807）实际用的是 clamp 代理，而不是它旁边那个恒等 STE。
**危害**：`x < 0` 时 `proxy = 0` ⇒ **梯度恒为 0**。一旦切三值
（`k_orig ∈ {−θ,0,θ}`），负极性那一半的梯度会被整个掐掉 —— 而 motion 项
本来就是在三值下才需要区分极性的那个（T49 §5）。

---

## 5. 对 T49 的两处修正（重要）

| # | T49 原文 | 修正 |
|---|---|---|
| 1 | §3.1「24 个 ATLIF 模块……一辈子没收到过一次梯度（代理梯度数值上死掉）」 | **24 个是结构性死**（12 `sn2_q` 未被调用 + 12 `attn_sn` 输出被丢弃）；**数值死只涉及剩下 81 个**。§3.2 的 surrogate 分析对那 81 个仍然成立。 |
| 2 | §4/§5「h60 无需代码改动就能跑三值，唯一极性泄漏是 `binary_motion_xor_alpha`」 | 方向对，但**低估了**：三值下真正需要改的还有 (a) **`_binary_event_ste` 双重定义**必须收敛成一个（否则负极性无梯度，§4.5），(b) `mismatch_penalty` 打开后 `opposite` 才非空，(c) `alpha0` 的 2/3 惰性说明**「同静默奖励」这条路本来就实现不出效果**，不要指望它。 |

---

## 6. 三元重训 cell A 的最新实测（顺带，2026-09-17 18:24）

`t49_ternary_ws4_ep5` 跑到 **epoch 1 / step 860（23%）**（远端 wall clock 16:24），无报错。
epoch 耗时 ≈ 85 min（3672 step × 1.39 s/it；epoch 0 收在 16:03，与 `checkpoint_epoch35.pth`
的 mtime 一致）⇒ **epoch 1 收在 ≈ 17:29（= T49 §7.1 的 gate 时间，原估计正确）；
5 epoch 收在 ≈ 21:43**（`epoch_offset: 35`，盘上文件 `checkpoint_epoch35..39.pth`）。

`[H9] step 780 update` 里已经能读出几件对 §3 有用的事：

| 量 | 值 | 读法 |
|---|---:|---|
| `threshold_mean` | **1.002033**（min 1.0 / max 1.003957） | 三值路径下 θ **真的在动**（T49 §7.1 结论维持） |
| `update_mean` / `raw_update_mean` | 0.0 | `threshold_eta: 0`，手动生长仍关着（刻意单变量） |
| `threshold_updates_frozen` | 1 | step = 3672×1 + 780 > 1224 ⇒ 冻结生效（与源 run 同机制，§3.5） |
| `activity_mean` | 0.0559 | FR 5.59%（scale=4） |
| `ternary_neg_mean` / `negative_trigger_mean` | 0.0126 / **0.3679** | 负脉冲实际 1.26%，负半轴原始密度 36.8% |
| `ternary_pos_neg_ratio` | 3.43 | 正/负比 |
| **`ternary_zero_pos_modules` / `ternary_zero_neg_modules`** | **25 / 27** | 105 個模块里 25 个正脉冲恒 0、27 个负脉冲恒 0 |
| `ternary_worst_pos_neg_ratio` | 4.4e7 | 最差模块几乎只有正脉冲 |
| `quantile_modules` / `importance_modules` / `temporal_factorized_modules` | 0 / 0 / 0 | **§3.6 的实测确认** |

> 对 25/27 的口径提醒：其中 **24 个就是 §3.1 的结构性死模块**
> （12 个没被调用 + 12 个输出被丢），它们恒为 0 是预期的；
> 即真正「活着但极性单边」的模块只有 **1 个（正）/ 3 个（负）**。

---

## 7. 结论：这个 checkpoint 该怎么描述

**不能写**（有实测反例）：
- ❌「ATLIF 自适应阈值」—— `threshold_eta=0`，θ 是手设常数 1.0；
- ❌「有符号三值脉冲」—— 输出 ∈ {0, θ}，`opposite` 恒空集；
- ❌「signed consensus / bipolar 打分」—— `mu = 0`，该项乘 0；
- ❌「反极性惩罚」—— 定义域空 + 系数 0；
- ❌「注意力输出经脉冲神经元稀疏化」—— `attn_sn` 输出被丢弃，`proj` 吃连续实数。

**可以写**（骨架真实存在、且是这两个 idea 的落点）：
- ✅「**motion 偏置**」：`+0.125 × Hamming(K_t, K_{t−1})` 是 live 的非标准项（需补实验证明有效）；
- ✅「**overlap 打分 + K 活跃度惩罚**」：实际打分式 `1.02·overlap − 0.02·|K|`；
- ✅「**固定阈值二值 IF 神经元**（ATLIF 命名残留，105 个模块中 93 个被调用）」；
- ✅ 硬件侧的 **θ≡1 ⇒ 无阈值存储、脉冲即 1-bit、0 DSP** 这套账不受影响。

**下一步建议（按性价比）**：

1. **先修 §4.5 的双重定义**（一行删除，零成本，且是三值重训的前置条件）；
2. **决定 §4.4 motion 项要不要改成 query-相关**（这是唯一 live 的非标准项，
   也是最可能成为论文 novelty 的地方）；
3. **清理 §3.1 的 24 个死模块**（config 里 `sn2q_binary` 组可直接删；`attn_sn`
   要用就得把 `proj` 改成吃神经元输出）；
4. `threshold_lr_scale` 收敛到一个旋钮（§3.4）；
5. 目录按 §1.2 归置（等你点头）。

---

## 8. 文件

| 文件 | 作用 |
|---|---|
| `t49_audit_probe.py` | 从 `checkpoint_epoch34.pth` 导出 105 个 ATLIF 模块路径（远端 `/root/`） |
| `t49_score_probe.py` | 数值复核 §4.1 的打分式退化（远端 `/root/`） |
| `results/t49_atlif_paths.txt` | 105 条路径清单（与 `spike_profile.json` 的 93 条 diff 出 12 个 `sn2_q`） |

**出处**：`hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs/`
（`checkpoint_epoch34.pth`、`spike_profile.json`、`.yml`）；
`third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py:620-833`（block 与 SSA）；
`neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py`
（`_ternary_alpha_xnor_token_scores:1736`、`_binary_temporal_k_xor_popcount:1787`、
`_binary_event_ste:3143/4800`、`_qk_shiftmax_gate_forward:5329`、h60 分支 `6083`、`install:6628`）。
