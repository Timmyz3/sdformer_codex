# T49 报告：ATLIF 阈值为什么恒等于 1 —— 以及把它换成真三值要动哪些模块

日期：2026-09-17　　state：**§1–§4 诊断闭合 + §6 零样本实测闭合（决定性负结果）；
§6.2 scale 扫描与 §7 三值 warm-start 重训进行中**

一句话总览：**θ≡1 不是训练崩，是 ATLIF 自适应机制从未生效**（§1–§4）；
**零样本直接换三值崩 16.8×（§6）**，但**负阈值旋钮能把发放率压回二值水平（§6.3）**，
所以剩下唯一的开放问题是「在这个发放率预算下，极性的精度收益是否为正」（§7 重训在测）。

**触发**：用户看到全网 105 个 `spiking_neuron.thresh` 全在 [0.99988, 1.0]，
怀疑「训练出了问题，根本没发挥 atlif 的作用」。

**一句话结论**：**不是训练崩了，是 ATLIF 的自适应机制从未生效。**
该 checkpoint 里的 “ATLIF” 实际就是**固定阈值二值 IF 神经元**（θ=1），
“adaptive” 只是残留命名。θ≡1 不掉精度，因为 **θ 是一个冗余的尺度参数**。

---

## 1. 观测：θ 在整段训练里一个 ULP 都没动

| 检查 | 结果 |
|---|---|
| `checkpoint_epoch34.pth` 的 105 个 `thresh` | min **0.9998828** / max **1.0**；**95 个精确 = 1.0** |
| `train.log` 中 921 次 `[H9] step N update:` 的 `threshold_mean` | 恒为 `0.9999957510403225`（16 位有效数字全同） |
| 同上 `update_mean`（ATLIF 手动生长量） | 恒为 **0.0**（3666/3666 次） |
| 加载 ep29 时的 `neuron summary` | 已经是同一个 `0.9999957510403225` |

⇒ θ 是**从 ep29 继承来的**，ep29→ep34 这 5 个 epoch 对本参数**零贡献**。
`thresh` 的 shape 是 `()`，即**每层一个标量**，不是逐神经元。

---

## 2. 真因 a：ATLIF 的机制被配置关掉了

`dsec_c12_alpha0125_ep29_resume5_20260830.yml`：

```yaml
atlif_ternary_psn:
  output_mode: binary
  threshold_mode: official_atlif
  threshold_init: 1.0
  threshold_eta: 0.0        # ← ATLIF 的活动驱动生长强度（= module.sp）
  activity_eta: 0.0         # ← 发放率正则
  target_rate: null
  target_rate_eta: 0.0
  threshold_freeze_after_step: 1224
optimizer.param_groups:
  threshold_lr: 5.0e-06     # ep34 时实际生效 1.25e-06
```

`threshold_eta: 0.0` → `module.sp = 0.0`（日志 repr 里 `sp=0.0`）⇒
代理 surrogate 里 `thre_updates = (sp * zif_backward(...) * out).sum(0).mean()`
**恒等于 0** ⇒ `update_value ≡ 0` ⇒ `threshold_update()` 是 no-op。

生产树里的 `hw_autoresearch_nts07/system_simulator/scripts/m2270_atlif_semantics_probe.py`
早已独立量到同一件事（`results/m2270_atlif_semantics_probe_20260905/result.json`）：

```
counts: {installed: 105, invoked: 93, result_consumed: 81, installed_theta_exact_one: 95}
manual_growth_zero_all_directed: true
settings: {threshold_eta: 0.0, activity_eta: 0.0, target_rate: null, threshold_eta...}
```

---

## 3. 真因 b：残存的梯度路径在数值上也是死的

设计上还有第二条路：`thresh` 是 `nn.Parameter`，
`ATLIFTernaryPSN` 的 surrogate `backward` 会返回 `grad_thre`，
它被分到 optimizer 的 `atlif_threshold` 组（lr 5e-6，ep34 时 1.25e-6）。
但这条路同样没走通。

### 3.1 optimizer state 直接给出证据（`t49_optimizer_probe.py`）

`checkpoint_epoch34_state_dict.pth` 里 `optimizer.param_groups`：

| group | lr (ep34) | n | present in `state` | MISSING |
|---|---:|---:|---:|---:|
| backbone | 2.5e-05 | 95 | 95 | 0 |
| backbone_norm_bias | 2.5e-05 | 172 | 172 | 0 |
| atlif_neuron | 1.25e-05 | 105 | 81 | **24** |
| atlif_neuron_no_decay | 1.25e-05 | 105 | 81 | **24** |
| **atlif_threshold** | **1.25e-06** | 105 | **81** | **24** |

AdamW 只在 `p.grad is not None` 时建 state 条目 ⇒
**24 个 ATLIF 模块（三个组缺席槽位完全相同）一辈子没收到过一次梯度**
（与 m2270 的 `invoked 93 / result_consumed 81` 对齐）。

剩下 81 个有 state，但动量是死的：

| | exp_avg 取值范围 |
|---|---|
| ATLIF θ（81 个标量参数） | **[2.2e-17, 8.3e-12]**，其中 12 个**恰好为 0** |
| backbone 非标量参数（429 个，对照组） | 中位数 **5.8e-08** |

**差 5 个数量级**，且 lr 还小 20 倍。AdamW 的更新
`lr·m̂/(√v̂+ε) ≈ 1.25e-6 × 1e-13 / 1e-8 ≈ 1e-11`/步，
乘以 step 数后仍低于 θ=1.0 在 float32 下的 ULP（1.19e-7）⇒ **参数不动**。

### 3.2 根因：surrogate 的返回梯度漏掉了幅度项

`atlif_ternary_psn.py:105-112`：

```python
class OfficialATLIFSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thre, sp):
        out = (input >= thre).float()
        return out * thre, thre_updates          # 输出 = 1[h≥θ]·θ

    @staticmethod
    def backward(ctx, grad_input, _dummy):
        input, thre = ctx.saved_tensors
        normalized = (input - thre) / thre
        tmp = (1.0 - normalized.abs()).clamp(min=0)
        grad_input = grad_input * tmp
        grad_thre = -(grad_input * tmp).mean()   # ← 只返回这一项
        return grad_input, grad_thre, None
```

真实的 `∂(1[h≥θ]·θ)/∂θ = 1[h≥θ] + (Dirac)`，
但 `backward` 只返回**代理窗项** `-(grad·(1-|·|)₊).mean()`，
不含幅度项 `active`；而且窗 `(1-|(h-θ)/θ|)₊` 只在 `h ∈ (0, 2θ)` 上非零（薄壳），
最后还对**全张量**求 mean。

本地实测（`Overlay` 直接调用，随机 h）：

| 配置 | σ | 返回的 dL/dθ | 解析幅度项 | 比值 |
|---|---:|---:|---:|---:|
| binary/official | 1.0 | −2.58e−04 | +1.43e+00 | **1.8e−04** |
| binary/official | 4.0 | −7.94e−04 | +7.55e+01 | **1.1e−05** |
| ternary/asym | 1.0 | −4.61e−02 | +3.29e+00 | 1.4e−02 |
| ternary/asym | 4.0 | −2.00e−01 | +7.64e+01 | 2.6e−03 |

乘上网内 `grad_out` 的尺度（~1e-8）正好落到观测到的 1e-13。

### 3.3 为什么 θ 本来就该不动：它是冗余尺度

`official_atlif` 输出 `1[h≥θ]·θ ∈ {0, +θ}`。
**θ 的幅度部分等价于给下游权重整体乘一个常数** ⇒ 不可辨识方向，梯度天然被稀释。
所以「θ≡1」既不奇怪、也不掉精度 —— 网络把这份自由度全部交给了下游权重。

---

## 4. 顺手查出的结构性问题：注意力本是三值代码，却喂了非负二值

ep34 用的 `bsa_attention.mode: h60` 走
`_tx_sc_fusion_score_pair → _ternary_alpha_xnor_token_scores`，而后者：

```python
"""Ternary extension of CVPR 2025 alpha-XNOR spike similarity.
   ...
   Our ATLIF path is signed ternary, so same nonzero polarity is a strong match,
   same silence gets the paper's small alpha reward, and opposite polarity is
   penalized because it is harmful for flow direction."""
q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))   # ← sign()
k_event = _ternary_sign_ste(k_orig)                      # ← sign()
q_active = q_event.ne(0); k_active = k_event.ne(0)
same_nonzero = (q_event == k_event) & q_active & k_active
opposite     = (q_event == -k_event) & q_active & k_active
```

`official_atlif` 的输出**严格非负** ∈ {0,+θ} ⇒ `sign()` 之后只有 {0,+1}，于是

- `opposite = (q_event == -k_event) & 两侧都活跃` —— **恒为空集（死代码）**；
- `same_nonzero` 退化成单纯的 overlap。

也就是说：**注意力是照着「有符号三值脉冲」写的，神经元却吐非负二值。**
目前网络里的极性证据其实是靠另一条路硬造出来的 ——
`_dualrail_binary_tx_token_scores` 把 head_dim 对半切成**正轨/负轨**
（其 docstring 明写 "restores same/opposite polarity evidence for **all-binary ATLIF**,
whose scalar output is otherwise {0,+1}"）。

另外 ep34 的 `bsa_attention.mismatch_penalty: 0.0`，把负极性惩罚项的系数也配成了 0。

---

## 5. 换成真三值要动什么（code-level 清单）

**已查清的前置条件**：

1. **config 两处必须同时改**，否则 `ATLIFTernaryPSN.__init__` 直接 raise：
   ```python
   if threshold_mode == "official_atlif" and output_mode != "binary":
       raise ValueError("official_atlif follows the official binary ATLIF output {0, thresh}")
   ```
   ⇒ `output_mode: binary → ternary` **且** `threshold_mode: official_atlif → asymmetric_scale`
   （顶层 + `target_groups` 里两个组都要改）。
2. **负阈值**由 `negative_threshold_scale` 决定（ep34 = 1.0 ⇒ 对称 ±θ）。
   换三值后 `TernarySurrogate` 输出 `(pos_active − neg_active)·θ ∈ {−θ, 0, +θ}`。
3. **注意力侧：h60 路径不需要改代码就能跑三值，但有一处真实的极性泄漏。**
   按 h60 的实际调用链逐条核过（`bsa_attention.py`）：

   - **已经是对的**：h60 → `_tx_sc_fusion_score_pair`(3411) →
     `_ternary_alpha_xnor_token_scores`(1736)，它对 **Q 和 K 都取 `_ternary_sign_ste`**，
     并且 `same_nonzero` 用的是 `q_event == k_event` —— 这个等式**本身就区分极性**
     （(+,+) 与 (−,−) 都算 match，(+,−) 不算）。所以三值一进来，极性立刻参与打分，
     `opposite` 那一行也从恒空变成真实集合。**h60 无需代码改动。**
   - **真实的极性泄漏（唯一一处）**：`binary_motion_xor_alpha: 0.125` 走的
     `_binary_temporal_k_xor_popcount`(1800) 里是
     `k_event = _binary_event_ste(k_orig)` = `x.gt(0)`（3143）——
     **负事件一律被压成 0**，运动项对极性完全失明。
     代码级修法：换成 `_ternary_sign_ste`；该处最后做的是 `(k_event - paired).abs()`，
     换成 sign 后天然给出 **0 = 同极性 / 1 = 一侧静默 / 2 = 反极性**，
     正好是三值的运动语义。**config 级修法：把 `binary_motion_xor_alpha` 置 0。**
     本次重训分两格做归因（见 §7）：cell A 保持 0.125 不动（只换神经元），
     cell B 才动注意力（0 → 0 + `mismatch_penalty` 0.0 → 1.0）。
   - **`mismatch_penalty: 0.0` 要一起调**：负极性惩罚项系数在 ep34 配成了 0。
     三值下该项才有意义（`−β·opposite`）；不调的话反极性只是「不加分」而非「扣分」。
   - **可以不管的**：`_dualrail_binary_tx_token_scores`(3189) 只在
     `dualrail_binary_tx_qkselector_shiftmax` / `binary_dualrail_tx_shiftmax` / `date11_drtx`
     三个 mode 下被调用（5788），**h60 根本不走它**（它是 DATE11 那条线的产物）。
     `_binary_temporal_pair_stats`(594)/`_token_time_bundle_stats`(469) 对
     `dtype != torch.bool` raise，但它们只被 `_maybe_emit_h60_profile`(1244) 调用，
     而该函数在 `collector is None and bit_trace_collector is None` 时直接 return
     ⇒ **训练/评测里根本不执行**，且调用点自己就 `q_orig.detach().gt(0)` 造 bool；
     改了也只是让 profile 统计变成三值口径，不影响训练。
     （H11 的 `pp+α·nn−β·mismatch` 是另一条独立实现的先例，可作参考但不是必须。）
4. **必须重训**。§6 的零样本实测证明这不是可选项：权重是在 {0,+θ} 输入分布上学的，
   直接切分布 AEE 崩 16.8×。
5. **硬件侧代价（接到 T47/T48）**：现在「二值操作数 ⇒ 乘法退化成 select ⇒ 0 DSP」
   （T47 实测）。三值 ⇒ 2 bit 有符号操作数 ⇒ 要**条件取反 + select**
   （或 dual-rail），面积/控制会比二值版本大，需重新做 PPA。
   但**极性这一维是净增的信息**，属于「同一 2 bit 预算下信息量翻倍」。

---

## 6. 实测①：零样本直接换三值 —— 决定性负结果

`t49_ternary_config.py` 造出 `configs/generated/t49_ternary_zeroshot_ep34.yml`
（binary→ternary、official_atlif→asymmetric_scale，两个 `target_groups` 同步翻），
把 ep34 的 binary checkpoint 原样拷进来，在 **ep34 权重上做 zero-shot**
valid825 eval（**不训练**），输出落在
`results/t49_ternary_zeroshot_ep34/standard_valid825/epoch34/`。

加载审计是干净的，说明**这次换装确实生效了**、不是加载失败：
`checkpoint_load_audit: {missing_count: 0, unexpected_count: 0, checkpoint_overlay_keys: 210, remap: v1}`，
`module_counts: {ATLIFTernaryPSN: 105, ShiftmaxAttention: 12}`。

| 指标 | binary（ep34，口径锚点） | ternary zero-shot | 比值 |
|---|---:|---:|---:|
| AEE | **1.19951** | 20.18936 | **16.831×** |
| DSEC_Fl | **5.31336** | 97.84175 | **18.414×** |
| AAE_Benchmark | 5.10636 | 82.93073 | 16.241× |
| spikes | 72.89124 G | 597.20526 G | **8.193×** |
| 全局发放率 | 5.671% | 46.462% | 8.193× |
| dense FLOPs | 3796.67866 G | 3796.67866 G | 1.000× |
| effective FLOPs | 215.30571 G | 1764.02133 G | 8.193× |
| sparsity | **94.329%** | 53.538% | 0.568× |
| energy | 63865.914 uJ | 438776.518 uJ | **6.870×** |

**结论：零样本换三值不可行，而且是两个独立的失败叠在一起。**

1. **精度崩（16.8× AEE）**：二值神经元把整个负半轴丢掉了。`official_atlif`
   输出 ∈ {0,+θ}，`h ≤ −θ` 的元素一律记 0；换成三值后这批元素全变成 −θ 进入下游。
   网络从没见过这个分布 —— 权重里**没有**任何对「负极性」的处理能力。
   ⇒ **权重确实把「非负二值」这件事学进去了，重训是硬成本**，不是调参能绕过的。
2. **稀疏前提崩（94.3% → 53.5%，FR 8.19×）**：这是**硬件侧**的坏消息。
   T48 的整套成本账（946.38G 里 5.9% 是真工作）建立在「94% 的 MAC 落在 0 上」；
   负半轴一开火，这个前提就没了，energy 直接 6.87×。
   ⇒ 三值的**极性信息不是免费的**，它要拿发放率去换。

唯一的补偿旋钮是 `negative_threshold_scale`（`neg_thre = θ × scale`；
ep34 配的是 1.0，类默认值是 5.0）。§6.2 就是把这一维扫清楚：**大 scale 能把
FR 压回二值水平的那个点，AEE 是不是也跟着回来**——如果 FR 回来了 AEE 仍然崩，
就坐实「代价是权重没学过极性，而非阈值没调好」。

### 6.2 实测②：`negative_threshold_scale` 零样本扫描（进行中）

`t49_ternary_negscale_sweep.py` 在 sd5ai 上跑 scale ∈ {1, 2, 4, 8}
（权重/注意力/优化器全部不动，只动这一个标量），输出落在
`neuron_experiments/H9_bipolar_self_attention/results/t49_ternary_negscale/ns{1,2,4,8}/`。
结果回来后填这张表：

| neg_scale | AEE | DSEC_Fl | 全局发放率 | sparsity | spikes | energy |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 20.18936 | 97.84175 | 46.462% | 53.538% | 597.21 G | 438776.5 uJ |
| 2 | 待填 | | | | | |
| 4 | 待填 | | | | | |
| 8 | 待填 | | | | | |

判读口径：找到「FR 压回 ~5–6%」的 scale；若该点 AEE 仍在 10 以上
⇒ 坐实必须重训；若该点 AEE 快速回落 ⇒ 说明三值只需重设负阈值即可能可用，
重训的价值在于**把负阈值调到能保住稀疏的位置**。

### 6.3 冒烟测试顺带量到的关键数字：scale 就是发放率预算旋钮

起真重训前先做了一次 3 step 的冒烟测试（`--name t49_ternary_smoke --max-steps 3`，
确认装/载/优化器分组/前向全通），它顺带打出了一份 `ATLIFTernaryPSN summary`
（**scale = 4**，ep34 权重 warm start 后的第一次前向）：

| 量 | 值 | 含义 |
|---|---:|---|
| `activity_mean` | **0.0571** | 三值总发放率 = **5.71%** |
| `ternary_pos_mean` | 0.0438 | 正脉冲密度 |
| `ternary_neg_mean` | **0.0133** | **实际**负脉冲密度（用的是 −4θ 门槛） |
| `negative_trigger_mean` | **0.3636** | `mean(h ≤ −θ)`，即**负半轴在 scale=1 门槛下的原始密度** |
| `ternary_pos_neg_ratio` | 3.29 | 正/负比 |
| `negative_scale_mean` | 4.0 | 105 个模块全部生效 |
| `asymmetric_scale_modules` / `official_atlif_modules` | 105 / **0** | 换装确实生效 |
| `update_mean` | 0.0 | `threshold_eta=0`，自适应仍关着（刻意单变量） |

读法：`negative_trigger_mean` 是**不随 scale 变**的量（模块里写死
`self.negative_trigger_r = h_seq.le(-thresh)`，`atlif_ternary_psn.py:387`），
所以 0.3636 就是「负半轴本来有多大」；而真正决定发放率的是
`ternary_neg_mean`。**scale 1 → 4 把负脉冲从 ~36% 压到 1.33%（约 27×）**，
总发放率回到 5.71%，与二值基线的 5.671% 基本持平。

⇒ **稀疏性这一关是可以过的**：`negative_threshold_scale` 就是一个
「用多少发放率预算去买极性信息」的旋钮，且存在一个点让三值的成本与二值持平。
剩下要回答的就只有精度：**在这个点的极性信息值不值钱** —— 这正是 §7 重训要测的。

---

## 7. 实测③：真·三值重训（warm start，进行中）

`t49_ternary_retrain.py`（远端 `/root/t49_ternary_retrain.py`）在 sd5ai 上起。

**为什么可以 warm start（而不是照 config 里写的 "reload from a baseline checkpoint"）**：
H9 的装载顺序是「**先按 config 装 overlay，再把 checkpoint 灌进去（strict=False）**」——
`entrypoints/train.py` 的 `LOAD_MODEL_PATCH` 把 `install_atlif_ternary_psn`
插在 `load_model` **之前**，且 `MODEL_CHECKPOINT_SAVE_PATCH` 把 checkpoint 存成
`{"model_state_dict": ...}`（纯 state_dict，不是 pickle 的 module）。
所以 binary 权重可以合法灌进 ternary 模块，
`_configure_existing_atlif` 的 mode 守卫（守的是"已装好的模块被重新配置"）
在这条路径上**不会触发**。§6 的 zero-shot eval 已经用
`missing=0 unexpected=0` 实测验证过这一点。
⇒ 不必从 NB0 基线重训 30 epoch，直接拿 ep34 权重 5 epoch 微调，便宜 6 倍。

**协议**（复刻源 run 的实际超参，只动神经元语义）：

| 项 | 值 | 理由 |
|---|---|---|
| warm start | ep34（binary）权重 | 见上 |
| optimizer | **fresh**（不传 `--resume`），flat LR | 语义变了，旧动量无意义 |
| backbone / norm LR | **2.5e-5** | = 源 run（ep29→34）实际生效值（配置 1e-4 × 0.25，scheduler 在 ep29 已消费 milestones [20,25]、γ=0.5 硬编码） |
| neuron / threshold LR | 1.25e-5 / 1.25e-6 | 同上，各 ×0.5 / ×0.05 |
| milestones | [999] | 5 epoch 内不衰减（flat） |
| epochs | 5（每 epoch 存盘，看曲线） | 单卡 A800 ≈ 67 min/epoch |
| `negative_threshold_scale` | **4.0** | §6.3：该点三值发放率 5.71% ≈ 二值基线 5.671%，稀疏前提保住 |
| `threshold_eta` | 0.0（不动） | 刻意单变量；且自适应阈值在硬件上要付逐神经元阈值存储 |
| `--finetune 1` | 必须 | `loader.crop: null`，非 finetune 分支会索引 `None` |

**两格做归因**：

- **cell A**（`t49_ternary_ws4_ep5`）：只换神经元，
  `mismatch_penalty: 0.0`、`binary_motion_xor_alpha: 0.125` **保持源配置不动**
  ⇒ 「AEE 差多少」干净地归因到三值本身。
- **cell B**（`--mismatch-penalty 1.0 --motion-alpha 0.0`）：再动注意力，
  把 §5 查出的两个极性问题一起修掉 ⇒ 测「注意力要不要跟着调」。
  cell B 只在 cell A 不达标时才需要（省 5.6 GPU·h）。

冒烟测试（3 step）已通过：装/载/分组/前向全对
（`asymmetric_scale_modules: 105`、`official_atlif_modules: 0`、
`thresh_mean 0.99999575` 与 ep34 一致 ⇒ 权重确实载进来了）。

**评测**：`entrypoints/run_h9_standard_valid825_eval.py --config <cfg>
--run-dir results/t49_ternary_ws4_ep5 --epoch {37,39,40} --ranking-mode aee`
（落盘 epoch = raw + 35）。口径锚点：**AEE 1.19951 / DSEC_Fl 5.31336**。

结果回来后填：

| 格 | epoch | AEE | DSEC_Fl | FR | sparsity |
|---|---:|---:|---:|---:|---:|
| cell A | 37 | 待填 | | | |
| cell A | 39 | 待填 | | | |
| cell A | 40 | 待填 | | | |
| cell B | 40 | 待填 | | | |

---

## 8. 文件

| 文件 | 作用 |
|---|---|
| `t49_optimizer_probe.py` | 读 `checkpoint_epoch34_state_dict.pth` 的 optimizer state，判 θ 是否收到过梯度 |
| `t49_ternary_config.py` | 造三值 config + 复制 checkpoint + 起 zero-shot eval |
| `t49_ternary_negscale_sweep.py` | `negative_threshold_scale` ∈ {1,2,4,8} 零样本扫描（远端 `/root/`） |
| `t49_ternary_retrain.py` | 三值 warm-start 重训（cell A/B），含 `--dry-run` / `--max-steps` 冒烟 |
| `tools/sd5ai_ssh.py` | sd5ai 非交互 SSH（pty 驱动，无 sshpass/paramiko） |

**出处**：
`hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs/`
（`checkpoint_epoch34.pth` + `.yml` + `spike_profile.json`）；
`neuron_experiments/H9_bipolar_self_attention/results/dsec_c12_alpha0125_ep29_resume5_20260830/`
（`train.log`、`checkpoint_epoch34_state_dict.pth`、`summary.md`）；
`hw_autoresearch_nts07/results/m2270_atlif_semantics_probe_20260905/result.json`。
