# Motion(H60) 注意力计算化简评审 R1

- 日期：2026-09-17
- 范围：只读代码分析 + CPU 复现实验（无 GPU、无训练进程接触、未改动任何仓库文件）
- 现网配置：`neuron_experiments/H9_bipolar_self_attention/configs/generated/t49_ternary_ws4_ep5.yml:118-161`（mode=h60, binary_motion_xor_alpha=0.125）
- 部署配置对照：`neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_paper_w15_h67_motion_ep19_ft30_dyadic_q7q17_deploy.yml:118-169`（同 mode，alpha=0.25，hardware_quant_enabled=true）
- 主文件：`neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py`（6739 行）
- 基线对照：`third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py`
- 证据分档：`[模型]`=解析/恒等式；`[prof]`(CPU)=本目录 CPU 微基准（脚本见文末）；`[rtl]`=仓库内既有 RTL/仿真/综合文档。

## 0. 结论摘要

1. 现网 h60 一次注意力调用实际只保留 3 个有效项：**TX(same_nonzero + α0·same_zero) + Motion-XOR(0.125) → center → Shiftmax → (preserve_mean) → gate×K**；`mu=bipolar_mu=0` 使整条 **SC(signed consensus) 计算完全死掉**（照算不误，占比约 23% `[prof]`(CPU)）；`single_active`/`opposite`/`k_magnitude`/`castling`/`temperature`/`context_broadcast`/`bipolar_gate_clamp` 全部被 0/False 关闭但部分仍无条件计算 `[模型]`。
2. **语义发现（重要）**：TX 的 4 个 bool mask 公式对 Q **没有任何梯度路径**（PyTorch bool→float cast 不可导），`[prof]`(CPU) 梯度审计实测 `dq = None`；现网 mu=0 又使唯一可导的 SC 分支被乘 0。即当前配置下注意力分数对 Q 的梯度恒为 0、对 K 的梯度只来自 motion-XOR 一项。这解释了为什么 Motion 线“注意力太繁杂但收益不透明”。
3. 化简候选三条（可叠加）：
   - **A1 dedup-SC-off**（删死路径，前反向双精确）：`mu==0` 时不构造 SC；`single_active_penalty==0` 不构造单边掩码；`mismatch_penalty==0` 不对极掩码。预期 −30% 上下（CPU 实测 −23% 只算 SC）。
   - **A2 shared-ternarize**（并重复事件化，前向精确、反向需补偿）：Q/K 每次调用只 `_ternary_sign_ste` 一次，motion-XOR 复用同一 K event。
   - **A3 canonical-popcount**（规范式重构，前向 1 ulp 内一致、整型 Q7 域精确）：用 4 个统计量 `{s=Σq·k, o=Σ|q·k|, q_act, k_act}` 表示全部 TX 项——正是 RTL 侧 MSSB5 五充分统计量的既有形式（`docs/388`）。
4. 硬件收益主要在“验证模型与 RTL 对齐 + score-front 叶级 CSE”，不是新面积；RTL 现网本来就没有 SC（`docs/42`）。真正的硬件数字（MSSB5 相对 CSE7 面积代理 −15.20%、面积延迟积代理 −29.76%）是既有 `[rtl]`（开放映射代理）证据，不是本评审新测。

---

## 1. 现网部署实际走的注意力计算路径（逐段 file:line）

### 1.1 安装与调用点

| 段 | file:line | 说明 |
|---|---|---|
| 安装（12 个块） | `bsa_attention.py:6637-6670` `install_shiftmax_attention` | `_iter_attention_modules` (`5306-5335`) 按 `target_blocks` 选 12 个 `attn`；`module.forward = MethodType(_qk_shiftmax_gate_forward, module)` (`6668`) |
| 入口 forward | `bsa_attention.py:5338-5346` | 被 patch 的 `Spiking_QK_WindowAttention3D.forward` |
| 张量布局 | `5347-5364` | `x=[T,B_,H,W,C]`；`q_orig=[T,B_,H,N,D]`、`k_orig=[B_,H,T·N,D]`；`T=2, N=225(15×15), D=32` |
| 基线载体（被 h60 取代） | `third_party/.../Spiking_swin_transformer3D.py:692-694` | 基线只有 `att_token=sn2_q(sum_C q); attn=k·att_token`；h60 不保留该 carrier |
| h60 分支 | `bsa_attention.py:6092-6147` | 本评审主体 |

### 1.2 h60 分支逐段拆解（`bsa_attention.py:6092-6147`）

```
6099  mu = _apply_hardware_mu_quant(_scheduled_bipolar_mu(self, cfg), cfg)   -> 0.0（现网）
6100  tx_scores, sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)   -> 见 §1.3/§1.4
6101  scores = tx_scores + mu * sc_scores                                     -> SC 被乘 0（死）
6102  pre_quant_scores = scores                                              -> 别名，供诊断
6103-6118  hardware_rtl_shiftmax_enabled 支路（现网 false，训练期会 raise）
6119-6129  常规支路：
6120-6121     center: scores -= scores.mean(dim=2)      （行均值 = 450 token 维）
6122          _event_selective_temperature(...)         （event_temperature_enabled=false → 恒等早退 3166-3167）
6124          _apply_hardware_score_quant(...)          （t49: false；deploy: Q7 clamp[-2,2]+round STE）
6125          gate = shiftmax(scores, dim=2, eps)       （232-242: 2^x / 2^ceil(log2 Σ2^x)）
6126          row_sum = gate.sum(dim=2)                 （注意：缩放前的行和）
6127-6128     preserve_mean: gate *= 450
6129          _apply_hardware_gate_quant(...)           （t49: false；deploy: clamp[0,2]+round Q7）
6130-6140  _maybe_emit_h60_profile(...)                 （train.py 未安装 collector → 2 次 getattr 后早退 1244-1247）
6141      attn = k_orig.mul(gate)                       ← 真·输出（no-carrier）
6142-6145 _castling_aux_weight → 0 → 跳过（3118-3128）
6146      _window_context_broadcast → 恒等（context_broadcast_enabled=false，3179-3189）
6147      self.h9_castling_aux_weight = 0.0
```

其后收敛回公共尾声：`6626-6634`（attn_drop → reshape → `attn_sn` → `proj` → `proj_bn`），以及**每调用必跑的诊断尾部** `6617-6624`：`row_sum.mean()/min()/max()`、`gate.mean()`、`scores.mean()` 五次 `.detach().cpu()`（**5 次 host 同步/块/次**，无开关）。

### 1.3 TX 分数前端（`_ternary_alpha_xnor_token_scores`，`1736-1784`）

```
1751  q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))   # 第1次切分+事件化(Q)；q_token 另一次在 SC 1610
1752  k_event = _ternary_sign_ste(k_orig)                       # 第1次事件化(K)
1753-1757  q_active/k_active/same_nonzero/same_zero/opposite     # 5 个 bool mask（bool→float cast 不可导）
1758-1763  single_active = q_active ^ k_active                   # 无条件计算（singel_active_penalty=0 → 死项）
1764-1770  score = Σ_D [ same_nz + 0.02*same_zero − 0*opposite − 0*single_active ]   ← 4 次 cast + 加权加
1771-1774  + 0.125 * _binary_temporal_k_xor_popcount(q_orig,k_orig)                 ← Motion-XOR（唯一活梯度）
1776-1779  k_magnitude_alpha=0 → 跳过
1781-1784  _normalize_consensus_score(...): /head_dim(=32) * score_scale(=1.0)
```

> `[模型]` 恒等式：`same_nonzero=(o+s)/2`、`opposite=(o−s)/2`、`same_zero=D−q_act−k_act+o`、`single=q_act+k_act−2o`，其中 `s=Σq·k, o=Σ|q·k|, q_act=Σ|q|, k_act=Σ|k|`；因此现网全部 TX 项只依赖 **4 个统计量**（与 RTL MSSB5 的五个充分统计量同构，`docs/388_...:18-30`）。

### 1.4 Motion-XOR（`_binary_temporal_k_xor_popcount`，`1787-1814`）

```
1802-1803 硬约束 t_steps==2（Swin 窗 (2,15,15)）
1810      k_event = _ternary_sign_ste(k_orig)   # 第2次事件化(K)（TX 已做过一次）
1811      paired = k_event.flip(dims=(2,))      # 时间配对（RTL: dual-bank K store）
1812-1813 (k_event - paired).abs().sum(-1)      # popcount 型 XOR 证据 [B,H,T·N,1]
```

### 1.5 SC（`_signed_consensus_token_scores`，`1593-1666`）——现网死路径

```
1608  _temporal_motion_from_q_orig(...)  → motion_weight_alpha=0 → None（早退 1480-1481）
1610  q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))   # 第2次事件化(Q)（重复）
1611  k_event = _ternary_sign_ste(k_orig)                       # 第3次事件化(K)（重复）
1627  score = (q_event * k_event).sum(-1)       # 唯一对 Q/K 可导的分数分支
1634-1643 single_active（penalty=0 → 死项，仍无条件计算）
1644-1663 /head_dim 归一化；1666 * score_scale
```

融合函数 `_tx_sc_fusion_score_pair`（`3414-3428`）：先算 SC（`3424`）再算 TX（`3427`）；`k_magnitude_alpha=0` 跳过 `object.__setattr__` 旋转（`3421-3426`）。**mu=0 时 SC 全链计算后被乘 0 丢弃**（`6101`）。

### 1.6 现网 0/禁用 → 死路径/死旋钮清单

| 旋钮（t49 值） | 使用点 | 现网状态 | 类别 |
|---|---|---|---|
| `bipolar_mu=0.0` | 6099 `_scheduled_bipolar_mu`(3595-3611) | 0 → **SC 全链死**（含 1610-1611 重复事件化） | 死计算（照算） |
| `single_active_penalty=0.0` (+`_grad/ste_slope/ste_margin`) | 1758-1763 / 1634-1643 | 掩码无条件计算后被乘 0 | 死计算（照算） |
| `mismatch_penalty=0.0` | 1757/1768 | opposite 掩码照算后乘 0 | 死计算（照算） |
| `k_magnitude_alpha=0.0` | 1776-1779 / 1630-1633 | `if` 守卫，跳过 | 死旋钮（廉价） |
| `consensus_bias=0.02` | 只在 l1norm 分支（5489/5594/…/6531） | h60 完全不用 | 误导性死旋钮 |
| `value_mode=threshold` | 只在 1727-1729/3142 等 | h60 无 V 分支（`attn=k·gate`） | 死旋钮 |
| `relu_k_floor=0.0` | 3680/5268（非 h60 分支） | 死旋钮 | — |
| `bipolar_lambda/gate_min/gate_max` | 3652/3587-3592 | h60 不调用 | 死旋钮 |
| `sc_mu_schedule_enabled=false` | 3597-3598 早退 | 死旋钮（廉价） | — |
| `castling_matrix_aux_weight=0.0` | 6142-6145 | 训练期外也为 0 | 死旋钮 |
| `event_temperature_enabled` 缺省 false | 3166-3167 早退 | 死旋钮 | — |
| `context_broadcast_enabled` 缺省 false | 3185-3186 早退 | 死旋钮 | — |
| `hardware_quant_enabled`（t49 false / deploy true） | 252-259, 291-298, 364-367 | **t49 训练图与 deploy 图不同** | 配置分裂风险 |
| `hardware_rtl_shiftmax_enabled=false` | 6103-6118 | 训练期 raise 的部署专用支路 | 未启用 |
| `sn2_q`（模块） | 5360/…; h60 分支不调用 | **部署前向里是死模块**（基线 carrier 的 sn2_q 被丢弃，但仍在 checkpoint 与 ATLIF 105 模块清单里） | 死模块 |

`_ensure_match_code`（`5201-5257`）对 h60 是 no-op（`5202-5203`）；`_ensure_independent_value_branch` 不在 h60 列表（`6645-6659`），因此 **h60 不加载任何 overlay 新参数**。

---

## 2. 复杂度量化

### 2.1 张量账本（480×640, B=2, T=10, 窗 (2,15,15)）

`window_partition_v2`（`third_party/.../Spiking_swin_transformer3D.py:100-113`）把 `T=10` 的 5 个时间对与全部空间窗堆进 batch 维：`B_ = B × n_pairs(5) × n_sw`，每窗 `450 token(2×15×15)`、`D=32`。

| stage | heads | n_sw | B_ | lane 数/块/前向 (B_·h·450·32) | 块数 |
|---|---:|---:|---:|---:|---:|
| 0 | 3 | 1376 | 13760 | 594.5 M | 2 |
| 1 | 6 | 352 | 3520 | 304.1 M | 2 |
| 2 | 12 | 88 | 880 | 152.1 M | 6 |
| 3 | 24 | 24 | 240 | 83.0 M | 2 |
| 合计 | | | | **2.875 G lane/前向** | 12 |

（n_sw 与 `_D1_SPATIAL_WINDOW_CANDIDATES`（`1843`）完全一致，`[模型]` 由 480×640÷15 网格推导。）

### 2.2 每调用 pass 账本（1 pass = 对 `[B_,h,450,32]` 全域读写一次）

| # | 段 | pass | 现网去留 |
|---|---|---:|---|
| 1 | `_qkformer_token_q` permute+reshape（TX） | 1 | 活（SC 里又做一次，+1 死） |
| 2 | `ternarize(q_token)` sign+STE | 3 | 活（SC 重复一次，+3 死） |
| 3 | `ternarize(k_orig)` sign+STE | 3 | 活 ×2（TX+motion）（SC 再 +3 死） |
| 4 | TX bool 掩码（active×2, same_nz×3, same_zero×3, opposite×4, single×2） | ~15 | 其中 opposite(~4)+single(~2) 死 |
| 5 | 4×bool→float cast + 4 乘积 + 3 加 + 列归约 | ~9 | 其中 2 cast/2 项死 |
| 6 | Motion-XOR（重事件化 3 + reshape/flip 2 + sub/abs/sum 3） | ~8 | 全部活 |
| 7 | SC（permute 1 + ternarize 3+3 + mul/sum 2） | ~9 | **全死**（mu=0） |
| 8 | `tx + 0*sc` | 2 | 1 死 |
| 9 | 行级小张量（450×h×B_）：center 2 + shiftmax ~8 + preserve_mean 1 + hw quant 2 + attn-mul | ~14 | 活（pass 小 32×，按 lane 折算 ≈ 0.4 pass） |
| 10 | 诊断尾部 `6617-6624` | 5 次归约 + **5 次 host 同步** | 无开关常开 |

**约 50 个全域 pass**，其中死/可去 ≈ SC(9) + 重复事件化(3) + opposite(4+2 cast) + single(2+2) + dup permute(1) + 0·sc(1) ≈ **24（≈48%）**。最小核心（A1+A2+A3）≈ 15-18 + 行级 ≈ **26 pass（−48%）**。
访存：1 lane = 691 KB/窗(fp32)；50 pass ⇒ ≈ 69 MB/窗读写，化简后 ≈ 36 MB/窗 `[模型]`。

### 2.3 CPU 微基准实测 `[prof]`(CPU, 单线程, 32 窗×12 head×450×32, torch 2.2.2)

| 段 | 实测 (min-of-10) | 占整块 |
|---|---:|---:|
| TX base（α0 + 掩码，无 motion） | 89.1 ms | 53% |
| TX full（+motion 1/8） | 128.0 ms | 77% |
| motion-XOR 增量 | ≈ 38.9 ms | 23% |
| SC（死路径，独立） | 38.2 ms | 23% |
| `_tx_sc_fusion_score_pair`（含死 SC） | 164.5 ms | — |
| h60 全分数块 replica（deploy / train 旋钮） | 166.6 / 167.0 ms | 100% |
| center+shiftmax+preserve_mean+score/gate quant+`k·gate` 合计 | ≈ 4.1 ms | 2.5% |
| 诊断尾部（5 次归约+.cpu()） | 0.08 ms | ≤0.1%（GPU 上为 5 次 D2H 同步） |
| 稀疏事件输入（~5%，对齐 t49 实测 activity≈0.054）| TX 118.9 / motion 22.9 / SC 38.7 ms | 稀疏在 CPU 上几乎不减 |

两次独立运行的可复现结论：`fusion ≈ TX_full + SC`、`block ≈ fusion + 4 ms`、motion 增量 29–39 ms。单段数字受同机 t49 dataloader 噪声影响 ±2×，比值稳定。

### 2.4 最贵/最冗余 Top-5

1. **SC 全链死算**（`1593-1666` + `6101`）：占块 23%（CPU 实测），RTL 侧根本不存在（`docs/42`：H67 明确“不增加 SC”），可 0 精度代价删除（前反向双精确，见 §3-A1 证明）。
2. **TX 掩码树**（`1753-1770`）：占块 ≈ 40%（=TX 89 ms − 事件化 ~25-65 ms）；其中 opposite+single 两项被 0 系数乘（≈ 6 个 pass 纯浪费）；且整段 **梯度为零**，等价于“只用不学”的只读检查器。
3. **重复事件化**：Q 事件化 2×（1751/1610）、K 事件化 3×（1752/1810/1611），合计 5 次 sign+STE 而最小只需 2 次；removal 后前向精确。
4. **诊断尾部 5 次 host 同步**（`6617-6624`）：无任何开关，训练/推理每次调用都付；GPU 上是 5×12 次同步/前向，属纯延迟税。
5. **误导性死旋钮**（`consensus_bias=0.02`、`value_mode`、`relu_k_floor`、`bipolar_*`、`target_rate` 等 11 项）：配置里“看起来在工作”的旋钮实际对 h60 无效，是维护与评审成本的主要来源。

依赖链（每 token 行）：事件化 → 掩码/popcount 树(并行 4 支) → 加权加 → **行均值(450)** → **行 max(450)** → 2^x LUT → **行和(450)** → ceil_log2 → 移位/除 → gate 量化 → ×K ⇒ ≈11-12 级串行，其中 6 级是全行归约；A1/A2 不缩短该链（并行分支），A3 可把 4 棵掩码树并成 1 组 popcount → 串行级数 −2~3 `[模型]`。

---

## 3. 化简候选（改法草案 / 精度风险 / 硬件收益 / 最小验证）

> 全部候选都**不需要现在改代码**：草案给出精确 file:line 与伪代码；验证可在 CPU 上先做（本目录已有脚本），GPU 探针待 t49 结束后入队。

### A1 `dedup-SC-off`（删死路径；前向+反向双精确）

**改法**（`bsa_attention.py`）：
```python
# 3414-3428 _tx_sc_fusion_score_pair
def _tx_sc_fusion_score_pair(q_orig, k_orig, cfg, *, need_sc=True):
    if not need_sc:                       # ← 新增：mu==0 且无 profile collector 时
        return _ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg), None
    ...                                   # 原逻辑不动
# 6099-6101
mu = _apply_hardware_mu_quant(_scheduled_bipolar_mu(self, cfg), cfg)
need_sc = (mu != 0.0) or getattr(self, "_h9_profile_collector", None) is not None
tx_scores, sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg, need_sc=need_sc)
scores = tx_scores if sc_scores is None else tx_scores + mu * sc_scores
# 1736-1770 TX 与 1593-1643 SC：single_active/opposite 的掩码与 cast 用 if 守卫
if cfg.single_active_penalty: ...  # 反例：需要同时把 bias 项放进同一表达式，保持逐位一致
if cfg.mismatch_penalty: ...       # opposite 掩码只在 mismatch != 0 时构造
```
**精度风险**：现网配置下 **前向逐位不变**；反向也不变（被乘 0 的项梯度本来就是 0；`[prof]`(CPU) 梯度审计：live fused `dq ≡ 0/1,036,800`，`dk` 只来自 motion）。
**预期硬件收益方向**：RTL 无 SC 对象 → 面积不变；软件侧 score-front 时间 −23%，同时把模型验证面收缩到与 RTL 同构（少 3 个重复事件化 pass）。
**最小验证**：CPU：`equivalent_audit_cpu.py` 的 C1b（已跑：`max|diff|=5.96e-8` 为 float32 1 ulp，见 §4 注）；再补一条 deploy 旋钮（hw quant 开）的逐位比对。GPU 探针（排队）：300 step 训练 loss/AEE 与 t49 对照，期望 0 差。

### A2 `shared-ternarize`（并重复事件化；前向精确，反向需补偿因子）

**改法**：
```python
# 新增内部辅助，仅供 h60 分支内联使用（不改公共签名）
q_token = _qkformer_token_q(q_orig)          # 一次
q_event = _ternary_sign_ste(q_token)         # 一次
k_event = _ternary_sign_ste(k_orig)          # 一次
tx = _tx_from_events(q_event, k_event, cfg)  # 掩码/统计复用
motion = _motion_from_kevent(k_event, T)     # 1787-1814 的 reshape/flip 版本，跳过一次 sign+STE
sc  = (q_event * k_event).sum(...) if need_sc else None   # SC 复用同两张事件张量
```
**精度风险**：前向逐位一致（`_ternary_sign_ste` 前向即 `sign`，确定性）。**反向不等价**：STE 梯度 = 恒等 1，每去掉一次重复 pass 就少 1 单位入梯度（实测 legacy `|dk|/cand|dk|=0.3845`，即现网 K 的分数侧梯度只有候选式的 38%）。若要保持训练动态，需要在共享事件张量处乘回使用次数（3× q / 3× k 的替代口径），否则必须重训验证。
**预期硬件收益方向**：neutral（RTL 已 CSE，`docs/388` MSSB5）；收益在模型侧 ≈ −15~25% 时间与显存带宽，并为 A3 铺路。
**最小验证**：CPU 前向逐位（`equivalent_audit_cpu.py` C2 已跑，1 ulp）；反向用 `torch.autograd.gradcheck`-式比对（本目录脚本 C3/C4 已给梯度账本），再做一个 2 层小模型单步 loss/grad 一致性测试。

### A3 `canonical-popcount`（规范式重构；前向 1 ulp、Q7 整数域逐位）

**改法**（替换 1753-1770 的掩码块）：
```python
s   = (q_event * k_event).sum(-1, keepdim=True)          # +1/-1 证据
o   = ((q_event != 0) & (k_event != 0)).to(dtype).sum(-1, keepdim=True)
qa  = q_event.abs().sum(-1, keepdim=True); ka = k_event.abs().sum(-1, keepdim=True)
same_nonzero = (o + s)/2;  opposite = (o - s)/2
same_zero    = D - qa - ka + o;  single = qa + ka - 2*o
score_raw = same_nonzero + a0*same_zero - m*opposite - p*single + alpha_motion*motion
```
（Q7 部署域：`s,o,qa,ka` 全为整数，`a0=1/64` 为 dyadic，`score_q7 = 64*(4*same_nonzero + same_zero) + 8*motion` 一类整型式直接对齐 RTL，`docs/42` 的 `S64=64·count(q=1,k=1)+count(q=0,k=0)` 即其二元特例。）
**精度风险**：前向 float32 1 ulp（实测 5.96e-8）；**整型域精确**。反向**改变训练语义**：给 Q 补上梯度路径（实测 `dq: None → max 0.0163`），可能显著改变 Q 分支学习行为 → 必须要 GPU 对照训练；建议作为“训练侧机制升级”而非纯工程化简来立项。
**预期硬件收益方向**：0 新增面积（这就是 MSSB5 的软件同构式）；收益是 model↔RTL miter 可直接在整型域逐项对拍，叶级逻辑数下降（参照 `docs/44`：H67 score 198 cell vs TTX 131 cell，方向为“整数 score 替代零类生成逻辑”）。
**最小验证**：CPU：已跑 C1/C2；补一条 `int64` 域断言（`s,o,qa,ka` 全整数）与 Q7 round 后逐位比对。GPU 探针（排队）：t49 收尾后用简化式重跑 5 ep，AEE 门 ≤ t49 同基线段。

### D（附）诊断尾部整改（非数值）

`6617-6624` 用 `if getattr(self, "_h9_profile_collector", None) is not None or self.training and flag` 之类守卫包起来；部署推理图完全跳过。CPU 代价 0.08 ms，**GPU 上是每次调用 5 次 host 同步 ×12 块**，属于纯延迟税 `[模型]`（不计入本篇的精度主张）。

---

## 4. 证据与风险备注

- `[prof]`(CPU) 全部由本目录脚本产生（`torch 2.2.2+cu121` CPU 路径，单线程，未触 GPU）：
  - `prof_h60_attention_cpu.py` → `prof_h60_attention_cpu.json`（分段表，上文 §2.3）
  - `prof_h60_tx_ab_cpu.py`（TX motion=0 vs 1/8 的 A/B 复核：delta 29.5 ms ≈ 独立 motion 23.3 ms）
  - `equivalent_audit_cpu.py`（C1/C2 前向一致性 5.96e-8 = 1 ulp；C3/C4 梯度账本：TX `dq=None`、SC `dq/dk max=0.03125`、live mu=0 `dq≡0`、`|dk|` 只来自 motion）
- `[rtl]`（既有文档，非本次新测）：
  - `hw_autoresearch_nts07/docs/42_H67运动XOR与有界TTX硬件增量.md`：H67 = TX→Shiftmax→gate·K，**不加 SC/carrier/N×N**；motion 增量 = D=32 K temporal buffer + 32-bit XOR + popcount 树，**无通用乘法器**（alpha dyadic）。
  - `hw_autoresearch_nts07/docs/388_双线创新筛选_MSSB5晋级与Local5SourceWavefront否决_20260814.md:18-55`：MSSB5 五充分统计 `{overlap0, same-zero0, overlap1, same-zero1, motion}`；相对 CSE7 面积代理 −15.20%、关键路径 −17.17%、面积延迟积代理 −29.76%（开放映射代理，非签核）。
  - `hw_autoresearch_nts07/docs/394_双线Pass与物化对象审计_20260814.md:33`：score-front 每对 Q/K 读一次算两个时间 score 的 CSE 口径。
  - `hw_autoresearch_nts07/docs/44_H67主线切换与增量RTL设计验证.md:131-143`：Yosys 通用 cell（TTX score 131 / H67 score 198 / 两个 top hierarchy 3070 vs 3096）；`:128` **“center → quant 与 RTL 顺序反例：整行 bit-exact 未关闭”**——本次 A1/A3 若要写“部署等价”，必须连这条顺序分歧一起关。
  - `hw_autoresearch_nts07/docs/CLAUDE_MOTION_SIDECAR_MODEB_DESIGN_20260819.md`：侧车 Mode B（one-vote normalization）存储 −22.2%、无位宽增长——同方向（row-sum 归一对象可简）的既有 `[模型]+[prof]` 先例。
- 未关闭风险：① model 用 float `shiftmax`+Q7 量化，RTL 用 LUT-Q8+Q17（`_rtl_shiftmax_gate_q17` 只在 `hardware_rtl_shiftmax_enabled=true` 时启用，且训练期 raise）——部署 bit-exact 只在 `docs/44:128` 指出的顺序分歧解决后成立；② `row_sum` 语义两条支路不一致（`6126` 缩放前 vs `6116-6118` 归一后）；③ `sn2_q` 死模块仍在 checkpoint 与 ATLIF 105 模块清单中；④ A3 会激活 Q 梯度（`docs` 与本脚本均已给账）。
- 硬约束遵守：未改任何代码/配置/RTL/编号文档，未触碰 `docs/359/362/366`，未删文件，未重启 H82/H86，未发任何 GPU 命令；只在本评审目录内新增文件。

## 5. 建议下一步（按成本排序）

1. （CPU，即刻）在 `equivalent_audit_cpu.py` 上补 deploy 旋钮（hw quant 开 + α_motion=1/4）的逐位比对与 int 域断言 → 关闭 A1/A3 的精度风险。
2. （CPU，即刻）给 A1 做一次“钩住 12 个块、跑一个 CPU 前向（bs1、小分辨率）”的端到端 0-diff 验证（需要 checkpoint 与数据子集；不改配置，仅只读加载）。
3. （GPU 队列，t49 结束后）① A1 重跑 5 ep 的 AEE 对照（期望位级同分）；② A2 的梯度补偿因子扫描（1×/3×）各 5 ep；③ A3 单独作为机制实验重跑，观察 Q 梯度激活后的收敛。
4. （文档）若 A1 冻结，把 `consensus_bias / value_mode / relu_k_floor / bipolar_*` 等 11 个 h60 死旋钮在配置模板注释中标注 `h60-unused`，防止后续误接线。
