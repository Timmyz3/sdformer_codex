# T51：修 24 个死神经元 + overlay bug，以及「注意力太繁杂」的处置方案

日期：2026-09-17　　state：**三处修复已推 sd5ai 并冒烟通过（`num_modules: 81`）；LR 扫描完成（四臂严格单调）
并已证伪「加 LR」；自适应阈值标定完成（η=0.008 定档）且**两个 arm 的 1-epoch 长跑 + 评测已在跑**；
三值极性零样本矩阵（T53，6 arm）已接力到长跑之后**

用户指令（原文）：*"去修那24个死的神经元，把所有bug都修一下，然后就是注意力现在好像有点太繁杂了，
你看看目前有没有新的注意力可以用来替换"*；随后在「先救三值还是先开自适应」上选定
**"先做自适应，再去试三指"**。

---

## 0. 一句话

24 个「死神经元」**不是 24 个坏掉的神经元**，而是 24 个被 overlay 结构性地架空的 ATLIF 模块 ——
12 个（`attn.sn2_q`）**从不被调用**，12 个（`attn.attn_sn`）**输出被丢弃且阈值够不着**。
所以「修」的正确形态不是把它们救活，而是**不再往死路上装参数**（105 → 81）。
真正的代码 bug 只有一个（`_binary_event_ste` 双重定义），已修；
另外 motion 项一个极性泄漏，已修。**注意力那条：先别换 mode，先把 16 个实测惰性的旋钮删掉**——
「繁杂」的感觉主要来自这些不干活的参数，删它们是零风险的。

---

## 1. 24 个死模块：为什么修不了「活」，只能修「不再装」

### 1.1 `attn.sn2_q`（12 个）—— 从不被调用

基础模型用它造二值 Q 门：`att_token = q.sum(-1); att_token = self.sn2_q(att_token); attn = k.mul(att_token)`
（`Spiking_swin_transformer3D.py:693`）。overlay 把整个 forward 换成
`_qk_shiftmax_gate_forward`（`bsa_attention.py:6659`），里面**没有这一行**。
config 的 `sn2q_binary` 组却逐条列了这 12 条路径 ⇒ 配了个永不生效的组。

**处置**：installer 层拦截（见 §2.3），293 个 config 一个都不用改。

### 1.2 `attn.attn_sn`（12 个）—— 输出被丢弃 + 阈值够不着

两条独立的死因，各自都足够致命：

**(a) 输出被丢弃（硬结构事实）**。block 尾部是
```python
attn = self.attn_sn(x)
x = self.proj(x)          # ← 吃的是神经元之前的 x
...
return x, attn            # ← 神经元输出只是第二个返回值
```
调用方 `SSA()` 只用第一个返回值；只有纯可视化的 `get_lst_block_attention_scores()`
才取第二个。**这一行是从 SDformerFlow 上游继承来的**
（`third_party/SDformerFlow/.../Spiking_swin_transformer3D.py:707-709` 一模一样），
不是 overlay 引入的。

**(b) 阈值够不着（实测）**。`threshold_init: 1.0`，而进 `attn_sn` 的量是
`k_orig.mul(gate)`，`k_orig ∈ {0, ±θ}`、`gate = shiftmax(scores)`。
注意 ep34 配了 **`preserve_mean: true`**，所以 h60 分支里 `gate = gate * n_tokens`
（`bsa_attention.py:6126-6128`），**行均值被拉回 ~0.5–1** ⇒ 只有**高于行均值**的 key
才可能越过 θ=1.0 的门槛。实测口径（`spike_profile.json`）：**12/12 个模块在整个
valid825 上 spike 数精确为 0**。

> 我一开始写成「shiftmax 行和 ≤1 ⇒ 激活恒 ≤ θ/denom ⇒ 结构上不可能发放」，
> 那是**错的**——漏了 `preserve_mean` 的 ×n_tokens。正确说法是上面这条：
> **实测 0 发放**，机制是「阈值恰好等于门均值、只有高于均值的 key 才可能发」。

**处置**：同样在 installer 层不再安装。
**要真正救活它需要两件事同时做**：(i) `x = self.proj(attn)`（让 proj 吃脉冲输出），
(ii) 给它一个与注意力幅度匹配的阈值。这是**改精度的架构实验**，不是 bug 修复 ——
在实测发放率为 0 的前提下直接改 proj 会把注意力输出整条打成 0。

### 1.3 修完的效果

| | 修前 | 修后 |
|---|---:|---:|
| installed ATLIF 模块 | **105** | **81** |
| invoked | 93 | 81 |
| 有梯度 | 81 | 81 |
| 死模块 | **24** | **0** |

三分法（105/93/81，T50 §3.1）**收敛到 81**：装的 = 调的 = 有梯度的。
注意这**不改变任何前向数值**（24 个模块原本就对前向无贡献、对梯度无贡献），
所以**不需要重训就能生效**；唯一的副作用是旧 checkpoint 里那 24 组参数
在 warm start 时会报 `unexpected keys`（`strict=False`，非致命）。

**第三方独立佐证（修完在 sd5ai 实测）**：训练日志的 `ternary_zero_pos_modules` /
`ternary_zero_neg_modules` 从 **25 / 27 → 1 / 3**，**恰好少 24 个** ——
而 T50 §6 的判据正是「单边静默的模块恰好就是那批死模块」。两个独立口径
（结构调用链 vs 实测发放）给出同一个 24，这是本次修复最强的证据。

---

## 2. 代码 bug 修复

### 2.1 【真 bug】`_binary_event_ste` 双重定义 —— 已修

原状：`bsa_attention.py:3143`（恒等 STE）与 `:4800`（`clamp(0,1)` 有界代理）同名。
Python 调用时查 globals ⇒ **4800 赢**，于是**所有**调用点（包括 1807 的 motion 项）
拿到的是 clamp 代理，`x < 0` 时梯度恒 0。

修法（保留两边各自的意图）：

| 位置 | 改动 |
|---|---|
| `:4800` | 改名 `_bounded_binary_event_ste`，docstring 写明**不许再叫 `_binary_event_ste`**及其原因 |
| 它的 3 个调用点（`_cross_time_match_events` 两处、CF10/H79 的 `q_activity` 一处） | 跟着改名 ⇒ 继续用有界代理（它们本来就是 Match-Code 家族） |
| `:3146` 那个 | 恢复为**唯一**的 `_binary_event_ste`（恒等 STE），于是 1795–4799 之间的所有调用点恢复原本意图 |

### 2.2 【极性泄漏】motion 项对负事件失明 —— 已修

`_binary_temporal_k_xor_popcount` 原来 `k_event = _binary_event_ste(k_orig)` = `x.gt(0)`，
**负事件一律压成 0**。改为 `_ternary_sign_ste(k_orig)`：`(k_event - paired).abs()`
天然给出 0=同极性 / 1=一侧静默 / 2=反极性。

**前向等价性（关键，保证不破坏现有基线）**：ATLIF 二值输出 `k_orig ∈ {0, +θ}`，
`gt(0)` 与 `sign()` 在 `{0,+θ}` 上**逐位相同**。梯度上：原有效的是 clamp 代理
（`[0,1]` 内梯度 1），新的是恒等 STE（处处梯度 1），而 `k_orig ⊂ [0, 1.005]`
⇒ 差异只出现在 `x>1` 的测度零区域。**⇒ 对 ep34 的 binary/h60 配置，这次改动
前向逐位不变、梯度实质不变**；对三值配置则正好是想要的行为。

### 2.3 installer 层不再安装被架空的路径 —— 已加

`atlif_ternary_psn/installer.py`：

- 新增 `_is_shiftmax_gate_bypassed_path(path)`：捕获 `.sn2_q` / `.attn_sn`。
- `iter_non_qk_spiking_neuron_paths()` 跳过它们（挡住 `all_non_qk` 那条路）。
- `install_atlif_ternary_psn()` 的 **`target_paths` 与 group `paths` 两个显式列表**
  也过同一个过滤器（挡住 `sn2q_binary` 那条路），跳过的条数会打印一行告警。

**为什么改 installer 而不是改 293 个 config**：`sn2q_binary` 出现在
**293 个文件**里（configs/ + system_handoff/ 下游副本），逐份改既易漏又污染历史；
在安装点收口，一处生效、且有日志可查。

### 2.4 【连带必修】训练器**和评测器**的 checkpoint 审计守卫误报 —— 已修（两个文件）

§2.3 改完第一次冒烟**直接崩**：`RuntimeError: [H9] overlay checkpoint keys were not
registered before load`。原因不是新 bug，而是**旧守卫的反面**：
`entrypoints/train.py` 的 H9 审计原本假定「overlay 里的每个 ATLIF key 都应该被注册」，
少装 24 个模块后，旧 checkpoint 里那 24 组 key 在加载时全变成 `unexpected`，
守卫把它判成「load 之前漏注册」而报错。

**⚠ 关键：这条守卫在代码库里有两份，第一次只修了一份。**
`train.py` 走的是自己的 H9 守卫；而**评测路径** `eval_DSEC_flow_SNN.py` 走的是
`overlay/models/STSwinNet_SNN/h9_load_audit.py:124` 的**同名同语义**守卫
（`if overlay_unexpected: raise RuntimeError(...)`）。
只修 `train.py` 的后果是：**所有 pre-fix checkpoint（包括 ep34 锚点）的评测会全部报错**——
我自己的探针 checkpoint 因为是用修好的 installer 存的反而没事，所以这个回归
**在训练路径上完全看不出来**，只有去评 ep34 才会暴露。

修法（两处逻辑保持一致）：新增 `_is_dead_atlif_key`（train.py）/ `is_dead_atlif_key`
（h9_load_audit.py），判 `.spiking_neuron.` 之前的路径是否 `.sn2_q` / `.attn_sn`，
把 `overlay_checkpoint_keys` 与 `overlay_unexpected` 两个列表**同时**过滤。
对**晚于本过滤器的 installer** 这批 key 本就注册了，谓词永不命中 ⇒ 天然是 no-op。

**CPU 实测回归验证**（不需要 GPU，直接在 ep34 checkpoint 的 921 个 key 上跑）：

| | 值 |
|---|---:|
| checkpoint 里的 overlay key | 210 |
| 其中 `.sn2_q` / `.attn_sn` 的（= 死 key） | **48**（24 模块 × thresh/center） |
| **旧守卫会因此 raise 的条数** | **48** |
| **新守卫会 raise 的条数** | **0** ✅ |
| 过滤后 `overlay_checkpoint_keys` | 162 |

另外核对了两种判据的等价性：按路径分段（组件名 == `sn2_q` / `attn_sn`）那种写法
在 ep34 上选出的也是同样 48 个，**两边互无多余**。我最终采用 `.spiking_neuron.`
那一版，因为它与已经冒烟通过的 `train.py` 逻辑**逐字相同**，避免两份守卫出现语义漂移。

> ⚠ 踩坑记录：`train.py` 的 patch body **本身处在一个三引号模板字符串里**，
> 我第一版给新函数写了 `"""docstring"""`，直接**提前闭合外层字符串** ⇒
> `SyntaxError: invalid syntax`（line 55）。这里只能用 `#` 注释，不能用 docstring。
> `h9_load_audit.py` 是独立模块，没有这个限制。

---

## 3. 「注意力太繁杂」：先删惰性旋钮，而不是先换 mode

### 3.1 繁杂感主要来自 16 个**实测不干活**的旋钮

ep34 的 `bsa_attention` 配置块有 ~25 个键，其中**可证明无效**的：

| 旋钮 | ep34 值 | 实测结论（T50 §4） |
|---|---|---|
| `mismatch_penalty` | 0.0 | 系数为 0；且 `opposite` 在二值下**恒为空集** ⇒ 双重失效 |
| `single_active_penalty` | 0.0 | 系数为 0 |
| `center_scores` | true | 对 gate **精确零影响**（下游第一个算子是平移不变的 shiftmax），实测 dev `0.000e+00` |
| `bipolar_mu` + `sc_mu_schedule_enabled` | 0.0 / false | `scores = tx + mu·sc`，`mu=0` ⇒ 整个 signed-consensus 是**死算力** |
| `k_magnitude_alpha` | 0.0 | 整段分支不执行 |
| `hardware_quant_enabled` | false | `_apply_hardware_score_quant` 是恒等 |
| `event_temperature_enabled` | 未设(False) | `_event_selective_temperature` 是恒等 |
| `alpha0` | 0.02 | **2/3 惰性**：`alpha0: 0.02→0` 只让 gate 变 rel 7.7e-4，存活的只有 `−0.02·|K_act|` |
| `consensus_bias`/`consensus_score_norm`/`preserve_mean`/`eps`/`relu_k_floor` | — | 归一化/常数，不构成独立自由度 |

⇒ **有效打分式只有三项**：
```
gate = Shiftmax( (1.02·overlap(Q_act,K_act) − 0.02·|K_act| + 0.125·Hamming(K_t,K_{1−t})) / head_dim )
```
（T50 §4.1，`maxdev 3.7e-9` 与实现逐位相等）

**推荐动作 0（零风险，建议先做）**：把上表这些旋钮从 config 里删掉/置零并冻结，
这样**有效注意力立刻变成 3 项**，代码里的 ~20 个分支不再被配置激活，
「繁杂」是配置层的问题而不是算法层的。**AEE 不动**（因为改的都是 no-op）。

### 3.2 真要换 mode：repo 侧三个候选（有实测 AEE 的优先）

两条线都查过了：仓库内 178 个 mode 串 / ~55 个真实分支，以及 2023–2026 文献。

| # | 候选 | 性质 | AEE | 旋钮 | 判断 |
|---|---|---|---:|---|---|
| 1 | **`binary_axnor_local5_shiftmax`**（H66d / Local5） | **严格二值** `>0` | **1.2819**（valid825；hw-order 1.2804） | **3** | 唯一「全协议 valid825 有数 + 严格二值 + RTL 已闭环」三者齐备的 |
| 2 | `binary_axnor_temporal_pair_shiftmax`（H66/H67 TP-TTX） | 二值 + 时间同道 | 1.1656（**valid10**）/ H67 ep35 1.3297 | 少 | 加了时间 peer lane，只有 valid10 口径 |
| 3 | `binary_de9_match_code`（Match-Code h73–h80） | 二值 | 无全协议数 | 3–5 | 隐藏 18×D 码本，无 AEE |

锚点：**h60 当前 1.19951**（valid825）。
⇒ Local5 是 **+6.9% AEE 换掉 ~20 个旋钮 + 拿到严格二值操作数**
（顺带把 T50 §3.2 那个「`attn.proj` 是实数乘」的账本麻烦一并消掉）。
`h82–h86`（Class-File 家族）**更繁杂**、禁用 motion、无 AEE、硬件侧已被判 NO（2.4–2.6），
不是简化方向。

### 3.3 文献侧（2023–2026）—— 诚实结论：没有「更简单又更好」的

| # | 候选 | 说明 | 风险 |
|---|---|---|---|
| 1 | **SDSA**（Spike-driven Self-Attention）`SN(Q_s(K_s^T V_s))` | 二值、无 softmax/exp、O(ND)、归一化折进神经元阈值 —— 结构上最契合 | **是重写不是换开关**；验证场景是分类/分割，**在稠密回归（光流）上可能欠拟合** |
| 2 | **朴素 SSA 核 + 保留 Shiftmax 归一化** | 字面意义上的「h60 减 20 个旋钮」 | 与 §3.1 的「推荐动作 0」等价，但 BSA 自己的消融显示二值积 + Shiftmax 提升有限 |
| 3 | QKFormer 的 Q–K attention | 二值、无 scaling、方差最低 | 三者里最重 |

其他事实（对写论文有用）：本项目基座 **SDformerFlow 用的是朴素 windowed SSA**；
SNN 光流 SOTA **ST-FlowNet 完全不用注意力**；**BSA 是 NeurIPS 2025，不是
CVPR 2025** —— h60 的 docstring 写的是 *"Ternary extension of CVPR 2025 alpha-XNOR"*，
**出处表述建议核一遍**再写进论文。

### 3.4 推荐路径

1. **先做 §3.1 的「删惰性旋钮」**（零风险、零精度代价、立刻少 16 个键）；
2. 若论文需要「注意力很简单」的**硬件叙事** ⇒ 换 **Local5**，代价 +6.9% AEE，
   建议跑一格 `binary_axnor_local5_shiftmax` 在 valid825 上复现 1.2819 再定；
3. SDSA 只在愿意付「重写 + 重训」成本时考虑，且要先接受「分类上验证过的机制未必
   适配稠密回归」这个风险。

**没有换之前我不动 h60**：上面每一条都是可测量的取舍，需要你点头选路线。

---

## 4. 跑过的 / 待跑

### 4.1 已完成（sd5ai 实测）

| 实验 | 结果 |
|---|---|
| 推修复到 sd5ai + 3-step 冒烟 | ✅ `num_modules: 81`（原 105）；installer 跳过告警出现且点名 12 条 `.sn2_q`；H9 审计守卫通过；Validation 段正常跑完；无 traceback |
| `ternary_zero_pos/neg_modules` 复核 | ✅ **25 / 27 → 1 / 3**，恰少 24，与结构判据独立吻合（§1.3） |
| cell A 的 epoch-1 loss gate | ⚠ **原判「LR 小了约两个数量级」已撤回**（见 §4.2 第 3 条）。原证据：train 8.7222→8.7043（−0.2%），val 7.7205→**8.4124（+9%）**；我据此把原因归到「`2.5e-05` 只是源 run 两次衰减后的微调 LR」。但 §4.2 的三档扫描显示 **LR 越高越差**，且 2.5e-5 跑满一步后 val 7.7205 优于三档 600 步，⇒ 正解更可能是**过拟合/不稳**〈train 与 val 未同向〉。已杀进程、GPU 已腾空 |

### 4.2 已完成：短程 LR 扫描（用户已选）—— 结论与预期相反

`t51_lr_sweep.sh`：同一 ep34 warm start，三个 600-step 探针（≈11 min/arm，1.13 s/it，epoch=3672 step），
只差 backbone LR。判据是**同起点下的 loss 下降速度**，胜者起长跑（cell A′）。

| arm | LR | train（600 步均） | val（600 步终点，n_valid=1） | **valid825 AEE** | DSEC_Fl | FR% | spikes_G |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`t51_lrprobe_2p5e5`（同预算对照）** | **2.5e-5** | **8.9759** | **6.9601** | **9.3214** | **77.99** | 6.678 | 85.83 |
| `t51_lrprobe_1e4` | 1e-4 | 9.3095 | 8.1565 | 10.3057 | 79.28 | 6.497 | 83.51 |
| `t51_lrprobe_3e4` | 3e-4 | 9.6132 | 8.2008 | 12.1658 | 83.23 | 6.422 | 82.55 |
| `t51_lrprobe_1e3` | 1e-3 | 10.5619 | 9.1607 | 12.7404 | 84.49 | 6.957 | 89.42 |
| 参照：**零样本 S=4**（不训练，同 SRC_CFG） | — | — | — | **8.2631** | 74.96 | 6.880 | — |
| 参照：二值锚点 ep34 | — | — | — | **1.19951** | 5.31336 | 5.6709 | 72.89 |

**「LR 小了」的假设被证伪，而且方向完全反过来。**

1. **四个 arm 在 val 上严格单调：LR 越小越好。**
   val **6.9601**（2.5e-5）< 8.1565（1e-4）< 8.2008（3e-4）< 9.1607（1e-3）；
   train 均 **8.9759** < 9.3095 < 9.6132 < 10.5619，同序。
   同预算对照比 1e-4 好 **14.7%**（6.9601 vs 8.1565）——**不是噪声级差异**。
2. ⇒ **源配置的 2.5e-5 很可能本来就是最优的**，「把 LR 往上调」这条路线整体作废。
   长跑 cell A′ 与自适应格都应当用 **2.5e-5**。
3. epoch-1 gate 那组证据（train −0.2% / val **+9%**）的正解因此是
   **过拟合/不稳**，而不是「LR 太小」——LR 太小应当 train 与 val **一起**不动。
   §4.1 里「LR 小了约两个数量级」的判读**据此撤回**。
4. **仍存的分辨率限制**：2.5e-5 与 1e-4 之间的趋势虽单调，但这是 **n_valid=1** 的验证。
   2.5e-5 是否优于「更小」（如 1e-5）**没有测**，也不该在没有需求时测——
   当前证据只支持「不要往上调」。
   ⇒ 最终取舍仍以 valid825 的 AEE 为准（正在评）。
5. 一个尚未解释的现象：探针 **600 步** val 6.9601 < 源 run **3672 步** val 7.7205。
   若两者确可比（同起点同 LR），意味着**从 ep34 起再多训会把 val 训坏**
   （过拟合），这对「warm start 该训多久」是个独立信号。但两者
   LR 调度/预算都不同，**在 valid825 AEE 出来前不下结论**。

> ⚠ **两条口径纪律（都会直接改变结论读法）**：
>
> **(1) 不能跨预算比。** `Epoch loss` 是**整段 600 步的平均**，`Epoch loss (Validation)`
> 是 600 步后的终点，而源 run 的 8.7043 / 8.4124 是 **3672 步**之后的数。
> 拿 600 步的探针去比 3672 步的基线是**无效比较**（早退的探针天然吃亏/占便宜）。
> 所以另加 `t51_lrprobe_2p5e5`（**同样 600 步、同样 warm start、只差 LR=2.5e-5**）
> 作同预算对照 —— 有了它才敢说「2.5e-5 在这个预算下**更好**」。
> 四个 arm 之间同预算，**只有它们可以横向比**。
>
> **(2) 训练内 val 是 1 个序列。** 配置里 `test: n_valid: 1` ⇒ 训练时那个
> `Epoch loss (Validation)` 只取**单条序列**，噪声大。
> 因此**最终判据改用 valid825 的 AEE**（`eval_DSEC_flow_SNN.py --mode valid`，
> 走 `DSECDatasetLite(file_list="valid")` = 全 valid split，与锚点
> AEE 1.19951 / DSEC_Fl 5.31336 同口径），对每个探针存下的
> `checkpoint_epoch35.pth` 各评一次。

收敛目标参照源 run 的 0.93 / 0.88。

> ⚠ **读 cell A′ 时必须记住的一条**：探针实测 `update_mean: 0.0` /
> `target_rate_control_modules: 0`，即配置里 `threshold_eta / activity_eta /
> target_rate_eta` **全是 0、`target_rate: null`** ⇒ **θ 恒定 1.0，ATLIF 的自适应被彻底关掉**，
> 当前跑的东西其实是「固定阈值的非对称 LIF」。
> 这不是本次改动引入的，是源配置本来就如此（T49 的结论）。
> **但它意味着**：无论 cell A′ 训得多好，都**没有证明 ATLIF 的自适应有价值** ——
> 而自适应正是 ATLIF 相对 LIF 的唯一增量。要回答这个问题必须另起一个
> 打开自适应的对照格（见 §4.4）。

三值发放实测（探针 step 180，81 模块）：`ternary_pos_mean 0.0555` /
`ternary_neg_mean 0.0159` / 合计 `0.0715` —— **两个极性都在发**，
`negative_threshold_scale=4.0` 起了作用；残差 `ternary_zero_pos_modules: 1`、
`ternary_zero_neg_modules: 3`（原 25/27）留着以后看。

### 4.4 自适应阈值对照格（用户已选「先做自适应」；探针已挂，等 GPU）

#### 4.4.1 先纠正三件事（都会静默毁掉这一格）

**(1) `activity_eta` 是死键。** `ATLIFTernaryPSN` 只在 `__init__` 里
`self.activity_eta = float(activity_eta)`，**forward 与 `threshold_update` 全程不读它**。
我上一版 launcher 加的 `--activity-eta` 因此会**静默地什么都不做**，跑完只会得到一个
假的「自适应无效」结论。已把它换成真正生效的 `--threshold-eta`。

**(2) `threshold_freeze_after_step: 0` ≠ 解除冻结，而是「从第 0 步起永久冻结」。**
installer 的判据是
`freeze_updates = freeze_after_step is not None and global_step >= freeze_after_step`
⇒ 只有 **`null`** 才不冻结；我上一版写的 `0` 恰好把自适应锁死。已改为 `null`。
（源配置是 `1224`：在 5 epoch ≈ 18360 步里，θ 只在前 1224 步有可能动。）

**(3) 自适应有两个**互不相通**的入口，别混着用**：

| 机制 | 旋钮 | 生效位置 | 门禁 |
|---|---|---|---|
| 质量型（官方 ATLIF，Activity-Pruning-SNN） | `threshold_eta` → `module.sp` | surrogate 内累积 `update_value` | 无 |
| 发放率反馈型（本 repo 自加） | `target_rate` + `target_rate_eta` + `target_rate_mode` | `installer.threshold_update:728-745` | **`threshold_mode != official_atlif` 才生效** |

两者都乘 `threshold_base_lr(3e-6) × threshold_lr_scale(5e4) = 0.15`，并且
`θ ∈ [min_threshold 0.001, max_threshold 2.0]`。

#### 4.4.2 为什么这一格要跑 **binary + official_atlif**（而不是三值）

单变量消融的前提是「对照点本身是好的」。三值 S=4 的零样本起点就是坏的
（AEE 8.263 / DSEC_Fl 74.96，§4.5），在它上面加自适应，精度差无法归因。
而 **binary + official_atlif + `threshold_eta=0` 的对照就是已发布的 ep34 锚点**
（AEE 1.19951 / FR 5.6709%）—— 这是最强、也最省的一个对照：不用重跑基线。
launcher 新增 `--binary-official` 让 `build_config` 保持源配置的
binary/official_atlif 不动，warm start 仍是同一份 ep34 权重 ⇒ **只动
`threshold_eta` 一个键**。

#### 4.4.3 η 量级：必须实测，不能拍脑袋

官方路径每步位移
```
Δθ/step = threshold_eta · mass · threshold_base_lr · threshold_lr_scale
        = threshold_eta · mass · 0.15            （official 模式里 mass 还要 ÷T）
mass = mean_over_modules( (zif(x−θ,θ) · 1[x≥θ]).sum(0).mean() )
```
源配置 `threshold_eta=0` ⇒ `update_value ≡ 0`（探针实测 `update_mean: 0.0`）
⇒ **mass 从来没有被测出来过，无从推算**。故挂了一个 **60 步标定探针**：

`t52_eta_calib`（`--binary-official --threshold-eta 0.1 --max-steps 60`），
用 0.1 而不是 1.0 是为了让 60 步内 θ 漂移可忽略、测到的 mass 代表锚点工作点。
读 `[H9] step N update: {...}` 里的 `raw_update_mean`（= η·mass）与
`effective_update_mean`（= 真实 Δθ/step），随后按
`η_target = 0.1 × (期望 Δθ/step ÷ 实测 Δθ/step)` 反解。
不解除冻结、不确认 `effective_update_mean ≠ 0`，这一格就是白跑。

**标定实测结果**（`t52_eta_calib`，η=0.1，60 步，2026-09-17 19:41）：见下方 §4.4.4 表。

#### 4.4.4 标定实测 + 已起的三个 cell

**标定实测**（`t52_eta_calib`，`--binary-official --threshold-eta 0.1 --max-steps 60`）：

| step | `threshold_mean` | `activity_mean` | `raw_update_mean` | `effective_update_mean` |
|---:|---:|---:|---:|---:|
| 20 | 1.00673 | 0.06700 | 0.0022983 | 3.4473e-4 |
| 40 | 1.01345 | 0.06478 | 0.0022382 | 3.3572e-4 |
| 60 | 1.02011 | 0.06554 | 0.0022588 | 3.3882e-4 |

- **冻结修复验证通过**：`threshold_updates_frozen: 0`（写 `null` 确实解冻了）。
- `official_atlif_modules: 81`、`num_modules: 81` ⇒ 装的就是 81 个活模块。
- `negative_scale_mean: 4.0` 伴随 `output_mode: binary` ⇒ 这个键在二值路径下**不参与前向**
  （`neg_thre` 只在 ternary surrogate 里用），所以二值格不受它影响。
- θ 单调上升 ⇒ 自适应确实在推 θ（`SOPS` 的唯一旋钮在工作）。
- 反解：`mass·0.15 = raw_update_mean/η = 2.265e-2` ⇒ `Δθ/步 = η · 3.4e-3`。
  取 **η = 0.008** ⇒ 1 个 epoch（3672 步）θ 目标 **+0.1**。
- ⚠️ `activity_mean` 在 20→40→60 步上是 0.0670→0.0648→0.0655，**非单调**：
  这是**逐 batch** 的量，混着 batch 噪声，不能拿它当 `d(rate)/dθ`。
  真正的斜率只能从评测的**全局 FR** 读（这正是长跑要给的数）。
- 局域估计（θ: 1.0067→1.0134, act: 0.0670→0.0648）给 `d(rate)/dθ ≈ −0.33`
  ⇒ θ=1.1 时发放率约 0.034（锚点 0.057 的一半）。**这个数不可信**，
  只用来定 η 的量级；FR 降幅必须由长跑实测。

**已起的三个 cell**（`bash /root/t52_adaptive_run.sh 0.008 2.5e-5 1`）：
两个 arm 各 1 epoch（3672 步）、同一 ep34 warm start、同一 LR 2.5e-5、只差 `threshold_eta`：

| cell | 配置 | 作用 |
|---|---|---|
| `t52_eta_off` | `threshold_eta=0.0`，freeze=`null` | **匹配步数对照**：隔离「多训 1 epoch」本身的代价 |
| `t52_eta_on` | `threshold_eta=0.008`，freeze=`null` | 自适应净效应；SOPS 由实测新 FR 反算 |
| （已发布锚点 ep34） | 0 epoch | AEE 1.19951 / DSEC_Fl 5.31336 / FR 5.6709% |

判读方式：
- `anchor → eta_off` = 多训 1 epoch 的代价（与自适应无关）
- `eta_off → eta_on` = **自适应阈值的净代价/收益**（同预算、同起点、单变量）

**为什么 LR 用 2.5e-5**：§4.2 的四臂匹配预算扫描已把「加 LR」这条路证伪
（2.5e-5 比 1e-4 好 14.7%，且严格单调恶化）⇒ 长跑统一用 2.5e-5。

标定产物：`t52_adaptive_chain.sh`（本目录；远端 `/root/t52_adaptive_chain.sh`）。

#### 4.4.5 上界已实测：ATLIF 的 θ 控制着 **100%** 的脉冲

把锚点的 `spike_profile.json`（93 个逐层 `spikes/elements/firing_rate`）与
checkpoint 里 81 个活 ATLIF 模块的路径对齐（`t52_atlif_share.py`，纯 CPU）：

| 量 | 值 |
|---|---:|
| 锚点 `total_spikes`（= `synops_total`） | 72,891,240,701 |
| 81 个活 ATLIF 模块覆盖的 spikes | **72,891,240,701（100.00%）** |
| 覆盖的 elements | 1,205,523,000,000（93.79%） |
| ATLIF 平均发放率 | 0.0605 |

**结论有两层，都要写进论文的账本：**

1. **自适应阈值是整个网络唯一的 SOPS 旋钮。** 全部脉冲都出自 ATLIF 控制的神经元，
   所以 `SOPS = f(θ)` 是**全网**的函数，不存在「ATLIF 管不到的那部分脉冲」。
   这比 §4.5 那个「三值只会加 SOPS」强得多：**方向对的是自适应，不是三值**。
2. **但上界是 100% 并不意味着可及**：θ 上去后发放率趋于非零平台，
   而且精度要付代价 —— 这正是 §4.4.4 那对 arm 要量的东西。

脉冲最集中的模块（占大头的都在这几个，θ 抬升的收益主要来自它们）：

| 模块 | spikes | rate |
|---|---:|---:|
| `encoders.swin3d.patch_embed.head.sn` | 8.93G | 0.0734 |
| `decoders.3.sn` | 7.98G | **0.2597** |
| `preds.3.sn` | 4.63G | 0.0762 |
| `encoders.swin3d.patch_embed.proj.sn` | 3.31G | 0.0543 |
| `encoders.swin3d.layers.0.downsample.sn` | 3.06G | 0.2015 |
| `decoders.0/1/2.sn` | 0.49 / 1.14 / 2.36G | 0.13 / 0.15 / 0.154 |

注意 `decoders.*` 与 `downsample.sn` 的发放率（0.13–0.26）是全网最高的，
而它们的 spikes 又大 ⇒ **它们才是 θ 抬升的主要收益来源**，
调参时应该盯这几个模块的 `firing_rate` 而不是全局平均。

### 4.3 待跑

| 实验 | 目的 | 依赖 |
|---|---|---|
| **ep34 锚点回归评测** | 用修好的 overlay 重评 ep34，确认 **AEE 仍 == 1.19951 / DSEC_Fl == 5.31336**（前向逐位不变） | 评测链结束腾 GPU |
| Local5 复现（valid825） | 把 1.2819 这个数复核一遍再决定换不换 | 同上 |
| 长跑 cell A′（含本次修复） | 拿到修复后的三值基线 | 扫描出胜者 LR |

> **ep34 锚点回归为什么必须做**：我声称三处修复「前向逐位不变、不需要重训就生效」，
> 但这是**推导**，不是实测。而现在评测路径刚被证明有独立的守卫（§2.4），
> 说明「训练路径没事」**不等于**「评测路径没事」。唯一能一次证明
> 「修复没扰动已发布锚点 + 评测路径已修好」的实验，就是拿 ep34 重评一遍看 AEE 是否复现。

### 4.5 三值的 SOPS 账本：一个把前提推翻的实测

`atlif_ternary_psn.py:20` 是决定性的一行：
```python
neg_thre = thre * float(neg_scale)     # ⇒ θ_neg = S · θ
```
即 `negative_threshold_scale` **越大 = 负阈值越高 = 负脉冲越少**，
`S → ∞` 时负分支彻底关掉，**退化成二值**。把它和零样本扫描放在一起：

| 配置 | FR% | synops_total | 相对二值 | AEE | DSEC_Fl |
|---|---:|---:|---:|---:|---:|
| 二值锚点 | 5.6709 | 7.289e10 | **1.00×** | **1.19951** | **5.31336** |
| 三值 S=8 | 5.7137 | 7.344e10 | 1.008× | 1.425 | 8.048 |
| 三值 S=4（重训选的） | 6.8799 | 8.843e10 | 1.21× | 8.263 | 74.96 |
| 三值 S=2 | 14.0638 | 1.808e11 | 2.48× | 17.16 | 95.78 |
| 三值 S=1 | 46.4622 | 5.972e11 | **8.19×** | 20.19 | 97.84 |

三条结论，每条都和原假设相反：

1. **三值是加 SOPS 的，不是省的**：S=4 是 **1.21×**、S=1 是 **8.19×**。
   此前记忆里那个「FR 8.19×」**一直是变差倍数**，从来不是节省倍数，用词要改。
2. **S=8 的「好 AEE」是假象**：它 FR 5.7137% 与二值锚点 5.6709% 只差 **0.04pp**
   ⇒ S=8 实际上**就是二值锚点**，它的 AEE 1.425 不能当作「三值有用」的证据。
3. **重训起点 S=4 本身就是坏的**（零样本 AEE **8.263** / DSEC_Fl **74.96**）。
   这就解释了为什么 cell A 的 loss 一直卡在 8–9：**不是训练有 bug，是起点坏**。
   （§4.2 里那个「train −0.2% / val +9%」的 epoch-1 gate 也因此应当读成
   「从坏起点上洗掉坏初值」而非「LR 太小」。）

### 4.5b 「再去试三指」之前必须先修的一件事：**极性旋钮此前一直是关的**

查 `_ternary_alpha_xnor_token_scores`（`bsa_attention.py:1736`，h60 走这里）
之后发现一个比 SOPS 账更要紧的问题：**此前所有三值 arm 的三值特性是被关掉的**。

```python
q_event = _ternary_sign_ste(...)          # 硬 {-1,0,+1}
k_event = _ternary_sign_ste(...)
opposite = (q_event == -k_event) & q_active & k_active
score = same_nonzero + alpha0*same_zero - mismatch_penalty*opposite
                        - single_active_penalty*single_active
if cfg.binary_motion_xor_alpha:
    score += binary_motion_xor_alpha * _binary_temporal_k_xor_popcount(q, k)
```

| 位置 | 代码事实 | 后果 |
|---|---|---|
| 神经元输出 | `official_atlif` 输出 ∈ `{0,+θ}`（严格非负） | `opposite` **恒为空集**（死代码） |
| `mismatch_penalty`（源配置） | `0.0` | 即便 `opposite` 非空，**系数也是 0**（双重关闭） |
| `binary_motion_xor_alpha`（源配置） | `0.125`，走 `_binary_temporal_k_xor_popcount`；**改前**内部 `_binary_event_ste = x.gt(0)`（`bsa_attention.py:3143`） | **改前把负事件压成 0**（motion 项也退化成二值）；**本轮已修**为 `_ternary_sign_ste`（§5），修复后它自己就区分极性 ⇒ 本格**不再动它**（单变量） |
| `k_magnitude_alpha`（源配置） | `0.0` | NTX-11 的 `|K|` 修正项关闭 |

⇒ **§4.2 的四个 LR 扫描 arm（以及更早的 cell A）跑的都是
`ter8/ter4 + motion_alpha=0.125 + mismatch_penalty=0.0`**，
即「三值的神经元 + 二值的注意力」。它们量到的是**三值的代价**，
而三值**唯一的机制收益（有符号事件匹配）根本没上电**。
`docstring` 自己写着 "Our ATLIF path is signed ternary" —— 代码要的是有符号事件，
配置喂给它的是非负事件。

**关键张力（决定了这一格怎么设计）**：极性份额 ∝ 1/S，而 SOPS 代价也 ∝ 1/S。
`neg_thre = θ·S` ⇒ S 越小负脉冲越多 ⇒ `opposite` 越非空，但能耗越贵。
**极性和能耗共用同一个旋钮，没有免费的有符号注意力** ⇒
只能按 **Pareto 曲线**记账（每点的 FR 与 AEE 一起报），不能只报 AEE。

**T53 零样本矩阵（已接力到 T52 之后，`t53_polarity_zeroshot.py`）**：
不训练，只换配置，一次问清两件事 ——

| arm | output | S | mismatch_penalty | 要回答的问题 |
|---|---|---:|---:|---|
| `bin_ref` | binary | — | 0.0 | 锚点回归：应 **仍 == 1.19951 / 5.31336** |
| `bin_mp_on` | binary | — | **1.0** | **A**：二值下开 `opposite` 应**逐位不变** ⇒ 证明它在二值是死代码 |
| `ter8_ref` | ternary | 8 | 0.0 | 与旧零样本 1.42507 的差 = **motion 修复在三值下的单独贡献** |
| `ter8_mp_on` | ternary | 8 | **1.0** | **B**：与 `ter8_ref` 的差 = **`opposite` 的单独贡献**（FR≈锚点） |
| `ter4_mp_on` | ternary | 4 | 1.0 | B：极性份额上来之后的收益/代价（负脉冲 ~17%） |
| `ter2_mp_on` | ternary | 2 | 1.0 | B：极性份额最大、能耗也最贵 |

**⚠️ 预期值修正（推 T53 前查证发现）**：motion 项的修复
（`_binary_event_ste`(x.gt(0)) → `_ternary_sign_ste`，`bsa_attention.py:1807`）
**在二值下是前向逐位中性的** —— 二值 ATLIF 输出 ∈ `{0,+θ}` 严格非负
⇒ `sign(x) ≡ gt(0)(x)`（`x>0→1=1`、`x=0→0=0`，没有 `x<0`；
`_qkformer_token_q` 只是 `permute+reshape`，不产生负值）⇒ **`bin_ref` 仍应精确复现锚点**，
「三处修复前向逐位不变」这个声明在二值下成立。
**但三值下 `k_orig` 可以取负**（`−θ` 或 `−S·θ`）⇒ 同一个修复会改 motion 项
⇒ 历史零样本数（S=8 **1.42507**、S=4 8.263、S=2 17.16，全带旧 `gt(0)` motion 项）
**不再可复现**。这不是 bug，是多了一个自变量；上表已把它拆成两步归因：
`旧1.42507 → ter8_ref` 量 motion 修复，`ter8_ref → ter8_mp_on` 量 `opposite`。

- **单变量纪律**：只翻 `mismatch_penalty`。`binary_motion_xor_alpha` **保持 0.125 不动**，
  因为该 motion 项已在本次修复里换成 `_ternary_sign_ste` ⇒ 它**现在自己就区分极性**
  （`|Δ| ∈ {0,1,2}` = 同极性 / 一侧静默 / 反极性），且 0.125 是调过的先验。
  把它一并归零会引入第二个变量。
- **A 是判定实验**：若 `bin_mp_on` ≠ `bin_ref`，说明这个旋钮改了前向里
  **与极性无关**的东西（那三值格的归因就得重做）。
- **B 是上界**：零样本下负半轴元素全是「权重没学过」的 −θ，所以 B 大概率仍然差；
  但 `ter*_mp_on` 与 `ter*_ref` 的**差值**才是「极性机制本身的贡献」，
  这是 §4.2 的四臂 LR 扫描**没量到**的东西（那些 arm 的 `mp` 都是 0.0）。
- 只有 B 显示极性朝着正确方向动，才值得投训练预算去做 §4.5c 的退火；
  否则三值就该按「纯代价」处理，不进论文的机制栏。
- **`mp` 的量级（1.0）是个选择，不是实测值**：score 会 `÷head_dim` 再 `×score_scale(1.0)`，
  `alpha0=0.02`、`single_active_penalty=0.0` ⇒ 1.0 的语义是「一对反极性抵消一对同极性」。
  若 B 的方向对但幅度小，第一件该扫的是 **mp ∈ {0.25, 0.5, 1.0, 2.0}**（仍是单变量）。
- **A 的预测为什么是硬的**：`_qkformer_token_q`（`bsa_attention.py:1456`）
  只是 `permute + reshape`，**没有任何算术** ⇒ `q_event = sign(q_orig)`
  ∈ `{0,+1}`，而 `opposite` 要求一侧 `+1` 另一侧 `−1` ⇒ **可证明为空集**。
  若实测 `bin_mp_on ≠ bin_ref`，就是我漏看了某条路径，三值格要重做归因。

### 4.5c 若 B 通过：退火怎么实现（已查清，不用改代码）

installer 里 `stage_*` 那套是 **swin stage（层编号）** 的覆盖，
`_iter_attention_modules` 的 `stage_selection` 也是按层选，**没有时间维度的调度**。
所以退火只能走 **分段 warm-start**：S=8 训 N epoch → 存盘 → S=4 从该 checkpoint
起训 N epoch → S=2 再训（warm-start 协议 §4.4 已验证可用，binary 权重可合法
灌进 ternary 模块）。除非愿意在 overlay 里加一个 epoch 边界改
`module.negative_threshold_scale` 的 hook，否则不要指望一个 config 搞定。

**已证伪的形态**：直接以 S=4 冷启动（§4.2 四臂，全 LR 都比零样本更差）。

---

### 4.6 产物与脚本

| 脚本 | 作用 |
|---|---|
| `t50_apply_overlay_fixes.py` | 推三处修复到 sd5ai（`--dry-run` / `--apply` / `--smoke`） |
| `t51_lr_sweep.sh` | 三档 LR 600 步探针（1e-4 / 3e-4 / 1e-3） |
| `t51_lr_chain.sh` | 等扫描完 → 同预算 2.5e-5 对照 → 四个 arm 的 valid825 评测 |
| `t51_eval_lrprobe.py` | valid825 评测驱动（`--mode valid`，与锚点同口径） |
| `t52_adaptive_chain.sh` | 等评测链完 → 60 步自适应标定探针（§4.4.3） |
| `t52_adaptive_run.sh` | 自适应正式长跑（`<eta> <lr> [epochs]`）：**eta-off 同预算对照 + eta-on** 两个 arm，再自动评测 `--epoch $((34+epochs))` |
| `t52_eval_runs.py` | 按**完整 run 名**做 valid825 评测（自适应格/三值格通用；SOPS = `spike_profile.json.total_spikes`） |
| `t52_atlif_share.py` | 纯 CPU：把 spike_profile 的逐层发放率与 ATLIF 模块路径对齐，算 θ 能控的脉冲份额（§4.4.5） |
| `t53_polarity_zeroshot.py` | h60 极性零样本矩阵（6 arm，§4.5b）；只换 config、不训练 |
| `t53_polarity_chain.sh` | 等 `T52 ADAPTIVE COMPLETE` → 跑 T53 矩阵（远端 PID 820724，2026-09-17 19:47 挂上） |
| `t49_ternary_retrain.py` | warm-start 重训 launcher（已加 `--threshold-eta` / `--binary-official`） |
| `backup_overlay_20260917/` | 四个 overlay 文件的改前快照 |

---

## 5. 改动文件

| 文件 | 改动 | 改后 md5 |
|---|---|---|
| `H9_.../overlay/models/STSwinNet_SNN/bsa_attention.py` | 4800 `_binary_event_ste` → `_bounded_binary_event_ste`（+3 调用点）；1807 motion 项 → `_ternary_sign_ste` | `0aaab8f6448d6a91f0a2cada01a69593` |
| `H9_.../overlay/models/STSwinNet_SNN/atlif_ternary_psn/installer.py` | 新增 `_is_shiftmax_gate_bypassed_path`；`iter_non_qk_spiking_neuron_paths` 与两处显式 paths 列表过该过滤；跳过时打印告警 | `169377f0df884c4ae1a645068a4ce7cb` |
| `H9_.../entrypoints/train.py` | 新增 `_h9_is_dead_atlif_key`；`overlay_checkpoint_keys` / `overlay_unexpected` 双双过滤（§2.4） | `07949a93198d48f1486f5c86c91ccbaa` |
| `H9_.../overlay/models/STSwinNet_SNN/h9_load_audit.py` | **同一守卫的第二份**：新增 `is_dead_atlif_key` + 同样过滤（评测路径，§2.4） | `74730d7d1783e27c54943aba0ec373c4` |
| `t49_ternary_retrain.py`（本 trials 目录） | 新增 `--threshold-eta`（替换死键 `--activity-eta`）；`--target-rate/-eta`；`--binary-official`；冻结键 `0`→`null`（§4.4.1） | `1206877a1581f522133136b32da414e7` |
| `backup_overlay_20260917/`（本 trials 目录） | 四个 overlay 文件的**改前快照** | — |

> ⚠ 这四个文件**未被 git 跟踪**（`git ls-files` 查不到），
> 所以回滚只能靠上面那份快照。sd5ai 上另有一份远端副本
> （`/root/private_data/work/sdformer_codex/SDformer/...`），推送时 `--apply` 会先做
> `.bak.20260917_t50` 备份。**本地与远端四个文件 md5 已逐一核对一致。**
