# ATLIF 死神经元 + 阈值冻结 诊断（2026-09-17）

用户报告两个执行问题：① 很多神经元是死的；② 阈值似乎从未被更新、始终是 1。
两问题同根，均有逐行日志/权重/capture 证据。本文只诊断与建议，未改动任何代码。

## 0. 结论一句话

当前 ATLIF 实现里阈值**没有任何可供下降的通道**（手动项=0、速率反馈=0、Adam 通道因梯度缺陷只能升），而死模块的梯度在**所有**路径上恒为 0，形成"阈值 1.0 ↔ 零发放"的自锁死态；H12 的 backward 修正单独**不能**救活死模块，必须同时开 bidirectional target-rate 反馈。

## 1. 证据（2026-09-17 实况）

### 1.1 t49_ternary_ws4_ep5 运行日志（368 条 update 记录，全量解析）

跑批：`results/t49_ternary_ws4_ep5/`（warmstart ep34，ternary，neg_scale=4.0，`threshold_freeze_after_step=1224`）。

| 指标 | 首条(step 20) | 中段(step 1940) | 末条(≈step 3700) |
|---|---:|---:|---:|
| activity_mean | 0.0578 | 0.0562 | 0.0553 |
| **ternary_zero_pos_modules** | **25/105** | **25/105** | **25/105** |
| **ternary_zero_neg_modules** | **27/105** | **27/105** | **27/105** |
| threshold_mean | 0.999996 | 1.00080 | **1.00251** |
| threshold_max | 1.0 | 1.00165 | **1.00514** |
| **update_mean（手动通道）** | **0.0** | **0.0** | **0.0** |
| target_rate_control_modules | 0 | 0 | 0 |

- 死模块数全程不变（24–25 正 / 25–27 负，仅 ±1 抖动）→ 无自愈。
- 阈值**单调上升**（1.00000→1.00514，从未下降），min 恰为 1.0 → 用户观感"始终是 1"成立。
- 训练侧：Epoch0 train loss 8.72 / val 7.72 → Epoch1 train 8.70 / val **8.41**（恶化）；此前零样本版 t49_ternary_zeroshot_ep34 AEE≈20.19、spikes 597G（崩溃）。

### 1.2 ep34 父模型权重（ATLIF_EP34_AUDIT_20260906 + 本次复核）

- 105 个阈值中 **95 个精确等于 float32 1.0**，10 个在 [0.99988, 1.0)；24 个 Q/K 阈值全部精确 =1.0。
- ep29→ep34 五个 epoch **逐位不变** → 历史上这些阈值也从未被有效训练。
- capture `m1458_..._20260831/forensic_samples/sample_39/`（40 次调用）：12 行零活动 = 诊断用 attn_sn（`deployment_dead_result=true`，结构性不消费）；另有一批"功能存活但近死"模块：`layers.1.swin_blocks.0.attn.sn_k` 仅 5 事件/414.7M（rate 1.2e-8）、`layers.2.swin_blocks.3.attn.sn_q` 2468/207M（1.19e-5）、`layers.0.swin_blocks.0.attn.sn_q` 4.1e-5、`layers.1.swin_blocks.1.attn.sn_q` 8.5e-5。

## 2. 根因链

### A. 阈值为什么只升不降（三重封锁）

1. **手动活跃度通道 = 0**：`installer.py:704-705` 是手动项唯一应用点：`thresh += update_tensor * lr * threshold_lr_scale`，而 `update_tensor` 来自 `module.update_value`，其源头 `sp * zif(...) * active` 在 `sparsity_eta(sp)=0` 时恒为 0（配置 `threshold_eta: 0.0 / activity_eta: 0.0`）→ 日志 `update_mean: 0.0` 逐行证实。
2. **速率反馈通道 = 0**：`installer.py:684-698` 要求 `target_rate ≠ null` 且 `target_rate_eta ≠ 0`；本配置 `target_rate: null`、`target_rate_eta: 0.0` → `target_rate_control_modules: 0`。且即便开启，`upper_bound` 模式把 `rate_error` 截断为 `max(r−target, 0)`——**r=0 的死模块反馈恒 0**（H12 §3.4 已预警的系统级第二因）。
3. **Adam 通道只能升**：四 surrogate backward 的缺陷（`atlif_ternary_psn.py:36-38 / 57 / 82 / 111`）先 `grad_input = grad_input * tmp` 覆盖再复乘 → tmp²，且缺恒等项 → `grad_thre ≤ 0` 恒成立 → Adam 单调推高阈值；lr 仅 1.25e-6（config `param_groups.threshold_lr`），叠加 `threshold_freeze_after_step=1224`（`installer.py:646-650`，仅冻结手动项、不冻结 Adam）。三者合成实测曲线：单调 +0.5% 上升。

### B. 死神经元为什么自锁（关键结论）

死模块膜电位分布低于 thre=1 → `ternary = 0` → 前向输出恒 0 → backward 的两条路径**同时**为零：STE 窗口项 `tmp`（|h−thre| 超窗）为 0，恒等项 `(g·ternary)` 因 ternary=0 也为 0 → 该模块的 weight/bias/thresh 梯度全零 → **永久冻结、不能自救**。

由此推出两条设计约束：
- **H12 的 backward 修正（恢复 `(g·ternary).mean()` 恒等项）只能救"低发放但仍在触发窗内"的模块**，对 ternary=0 的模块无效（恒等项仍为 0）。
- **唯一能对死模块产生下降力的通道是 bidirectional target-rate 反馈**（r=0 → rate_error=−target<0 → 阈值下降）——不依赖梯度。因此 H12-G1 必须同时满足：修正 backward **且** `target_rate>0, target_rate_eta>0, target_rate_mode=bidirectional`。

### C. 叠加因素

- **结构性死亡 24/105**：12 个 `attn.sn2_q` 在当前 H60 no-carrier 分支（`attn = k_orig.mul(gate)`）根本不被调用；12 个 `attn.attn_sn` 仅为诊断返回（capture 标 `deployment_dead_result=true`）。统计"死亡率"时应单列，避免混入。
- **neg_scale=4.0 过深**：负侧阈值是正侧 4 倍 → neg firing 仅 1.28%、27 模块零负发放；三元负支路近乎空转（且与 P1 负发放 guard 的语义冲突，待重定）。
- **worst_pos_neg_ratio = 4.7e7**：存在极端单侧失衡模块。

### D. Q 分支梯度断流（当日新增，来源 `hw_autoresearch_nts07/reviews/motion_attention_simplification_r1_20260917/review.md`）

h60 注意力 TX 分数前端全由 **bool 掩码**构成（`bsa_attention.py:1753-1770`，bool→float cast 无梯度），唯一可导的 SC 分支（`1627` 事件乘积 STE）又被 `mu=0` 乘 0（`6101`）；且 Q 在输出侧无其他通路（`attn = k_orig.mul(gate)`，`6141`）→ **注意力对 Q 的梯度恒为 0**（CPU 梯度审计实测 `dq ≡ 0`；`dk` 的分数侧只来自 motion-XOR）。

- 与 §1.2 的 Q 侧近死模块（`sn_q` rate 1.2e-8~8.5e-5，含 `layers.2.swin_blocks.3.attn.sn_q` 1.19e-5）互为印证：Q 侧"死"除 ATLIF 缺陷外还有**无梯度可用**这一结构性第二因。
- 含义：即便 H12 + 双向 rate 反馈落地，Q 侧要真正恢复学习还需打开 Q 梯度通路（候选 A3 canonical-popcount 会恢复 `dq`，但改变训练语义，须 GPU 对照；或维持 mu>0 让 SC 承担；或保守 A1 仅删死路径、不解决该问题）。

## 3. 修复建议（按优先级）

1. **H12 落地 + 三件套同时开**：修正 backward（H12 §3.3）；`target_rate≈0.05–0.1`、`target_rate_eta>0`、`target_rate_mode=bidirectional`；`threshold_lr_scale` 扫 {5e3, 1.5e4, 5e4}（H12 §3.6）；去/延后 `threshold_freeze_after_step`。
2. **死模块复活（一次性、不依赖梯度）**：用 H9 已有 `_h9_calibration_observer` 挂点（`atlif_ternary_psn.py:353-355`）做 per-module 膜电位分位数捕获；对 r=0 模块把 thresh 初始化到 |h| 的 q99（或先降到 0.1–0.3 再训），作为独立 A/B（对照组：只开 bidirectional、不重初始化）。
3. **neg_scale 4.0 → 2.0**（或先关负支路做 A/B），并重定 P1 neg_rate 统计口径。
4. **统计口径**：`零发放模块数` 与 `结构性 bypass 24 个` 分开报；报告里以 93 个动态调用/81 个功能存活为分母。
5. **t49_ternary_ws4_ep5 止损判定**：val loss 7.72→8.41 恶化、死模块数不变、阈值只升不降 → 该 run 不具备自愈条件；是否停跑由用户决定（本次未动 GPU）。

## 4. 复现命令（只读）

```bash
# 阈值/死模块时间序列
grep -o "num_modules': 105[^|]*" neuron_experiments/H9_bipolar_self_attention/results/t49_ternary_ws4_ep5/train.log | tail -1
# ep34 逐模块活动（capture）
python -c "import json;rows=json.load(open('hw_autoresearch_nts07/results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831/forensic_samples/sample_39/atlif_activity_cumulative.json'));print(sorted((r['active']/max(r['elements'],1),r['name']) for r in rows)[:12])"
```
