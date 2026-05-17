# I13 代码审查：SOC Attention + Angular Loss

日期: 2026-05-17 | Reviewed: soc_attention.py, train.py, full.yml, flow_supervised.py

---

## 审查总览

| 维度 | 评分 | 说明 |
|------|:---:|------|
| 实现正确性 | 3/5 | 整体可运行，但有隐蔽退化风险 |
| 目标对齐 | 3/5 | 论文声称与实现有 gap |
| 代码质量 | 3/5 | 可用但不稳健 |
| 审稿防御性 | 2/5 | 多个审稿人会抓的弱点 |

---

## 一、`soc_attention.py` — 逐行审查

### 1.1 高稀疏退化风险 (第 38 行) — 严重

```python
consensus = (agree - disagree) / (agree + disagree + silent + eps)
```

**问题**: ATLIF 将 Q/K firing rate 压到 ~0.07。两者都在同一通道同时激活的概率 ≈ 0.07² ≈ 0.005。对于 head_dim=32 的注意力头，期望的 `agree + disagree ≈ 32 × 0.005 ≈ 0.16` 而 `silent ≈ 31.84`，所以：

```
consensus ≈ (0.16 - 0) / (0.16 + 31.84) ≈ 0.005
```

所有 token 的 consensus 都接近 0。接下来：

```python
pos = torch.clamp(consensus, min=0.0)    # ≈ 0.005 for all
denom = pos.sum(dim=2, keepdim=True) + eps  # ≈ n_tokens × 0.005
gate = pos / denom * n                     # ≈ 1.0 for all (uniform!)
```

**在高稀疏条件下，SOC 退化为 uniform attention。** 所有 token 的 gate 近似相等，失去了注意力选择能力。这是 SOC 最致命的弱点，审稿人会发现。

**修复建议**:
- 监控 `consensus.mean()` 和 `consensus.std()` — 如果 std < 0.01 说明退化
- 添加 temperature 参数放大差异: `consensus = (agree - disagree) / (agree + disagree + eps) * temperature`
- 去掉分母中的 `silent` 项 — 只除以活跃通道数，静默通道不参与归一化

### 1.2 `gate * n` 乘法器声称矛盾 (第 45 行) — 中等

```python
gate = gate * float(n)
```

Docstring 声称 "Zero exponentiation, zero multiplication beyond the final gate application"，但这一行就是一个额外的乘法。加上第 86 行的 `k_orig.mul(gate)`，共有两次乘法。

**审稿人会说**: 你声称零乘法器，但你的代码里有两个乘法。

**修复**: 要么去掉声称，要么改用 bit-shift 近似（n 固定，可预计算）。

### 1.3 除法器成本未计入 (第 38, 42 行) — 中等

```
consensus = (agree - disagree) / (agree + disagree + silent + eps)  # 除法 1
gate = pos / denom                                                    # 除法 2
```

每个 head 每个 token 两次除法。12 heads × 162 tokens = 1944 次除法/window。审稿人如果要硬件报告，这会是主要质疑点。

**修复**: 去掉 "1 divider per token" 这种模糊表述，或在论文中诚实列出除法器数量。可以用近似倒数（`1/(x+eps) ≈ 右移近似`）替代，但需要验证精度。

### 1.4 `q_orig.sign()` 丢掉 θ 量级信息 (第 75 行) — 设计选择需辩护

```python
q_sign = q_orig.sign()     # {-θ, 0, +θ} → {-1, 0, +1}
```

θ 携带了置信度信息——高 θ 神经元的脉冲更稀缺更有信息量。`.sign()` 让 θ=0.13 和 θ=0.001 的脉冲完全等价。K 侧的 θ 保留在了 `k_orig.mul(gate)` 中，但 Q 侧的 θ 被丢弃。

**审稿人会问**: 为什么 Q 侧不需要置信度？论文需要解释 "Q 只管检索 (方向)，K 只管响应 (幅值)" 的不对称设计。

### 1.5 缺少梯度回退 (第 40-42 行) — 中等

```python
pos = torch.clamp(consensus, min=0.0)
denom = pos.sum(dim=2, keepdim=True) + eps
gate = pos / denom
```

当所有 consensus ≤ 0 时（训练早期可能发生），`pos = [0, ..., 0]`，`denom = eps`，`gate = 0/eps * n = 0`。此时：
- 所有 token 的注意力权重为零
- `attn = k_orig * 0 = 0`
- 梯度通过 K 路径丢失（`k_orig` 的梯度为 0）
- 只有通过其他网络路径（残差连接）的梯度能反向传播

**修复**: 添加 fallback to uniform attention:
```python
if denom < eps * 10:
    gate = torch.ones_like(pos)  # uniform fallback
else:
    gate = pos / denom * n
```

### 1.6 Patch 方式粗糙 (第 108 行) — 低优先级

```python
module.forward = MethodType(_soc_attention_forward, module)
```

这会替换所有 `Spiking_QK_WindowAttention3D` 实例的 forward，包括 decoder 侧的（如果有的话）。当前模型只有 encoder 侧有 QK attention，但代码没有做此检查。如果模型架构变化，会导致静默错误。

---

## 二、`train.py` — 逐行审查

### 2.1 Import Hook 可能无效 (第 255 行) — 严重

```python
_install_angular_loss_hook(repo_root)
```

Import hook 在 `sys.path` 设置之**前**安装。但 `loss.flow_supervised` 的 import 发生在训练脚本的 `exec(code, ...)` 内部——此时 sys.path 已经有 baseline_root。如果 Python 已经缓存了 `loss.flow_supervised` 模块（`sys.modules` 中有），import hook 不会再次触发。

**风险**: 如果 H9 overlay 或其他依赖已经导入了 `loss.flow_supervised`，hook 可能静默失效，angular loss 不会被启用。

**验证方法**: 检查训练日志中是否有 angular loss 相关的输出。如果没有明确证据表明 `curr_loss` 包含了 `lambda_ang * ang_loss` 项，则 hook 可能未生效。

### 2.2 重复代码 (第 60-84 行) — 低优先级

SCALER_STEP 和 OPTIMIZER_STEP 的 PAT CH 几乎相同（差异仅在第 65 行 vs 第 77 行变量名），应该合并。

### 2.3 SAVE_PATCH 语义空操作 (第 113-118 行) — 低优先级

```python
SAVE_ANCHOR = """            should_save_model = epoch_loss < best_loss or ...
SAVE_PATCH = """            should_save_model = (
                epoch_loss < best_loss or ...
            )
"""
```

PATCH 只是加了括号——完全不改变语义。应该删除这个无意义的 patch。

### 2.4 缺少 `use_angular_loss` 检查 — 中等

ANG_LOSS_PATCH 检查 `self.lambda_ang > 0` 但不检查 config 中是否有 `use_angular_loss: true` 标志。在 H9h config 中有这个标志但 I13 的 config 中没有，而 I13 设置 `lambda_ang: 1.0`。两者语义不一致。如果评审者 grep config 找不到 `use_angular_loss` 标志，可能会认为这个功能未启用。

---

## 三、`full.yml` — 配置审查

### 3.1 缺少 progressive schedule 实现 — 关键

Config note 声称 "progressive schedule"，但实际上 `activity_eta: 0.02` 是一个固定值——根本没有渐进调度。I13 实际上只包含了 **SOC + angular loss**，progressive schedule (A3) 没有被实现。

**审稿人会发现**: "你声称用了 progressive sparsity schedule 但你只有一个固定的 activity_eta"。

### 3.2 没有 downsample 替换 — 与 H9a 不同

| 组件 | H9a | I13 |
|------|:---:|:---:|
| Q/K | 三元 ATLIF | 三元 ATLIF |
| Stage0 FFN | 二元 | 二元 |
| Stage1 FFN | — | **二元** (新增) |
| Stage0 downsample | 二元 | — |
| Stage2 downsample | 二元 | — |

I13 加了 stage1 FFN 但又去掉了 downsample 替换。这不是一个 apple-to-apple 的比较——无法区分改进是来自 SOC 还是 FFN 目标组合的变化。

### 3.3 `lambda_ang: 1.0` 缺乏消融依据 — 低优先级

为什么是 1.0 而不是 0.5 或 2.0？没有 sweep 记录。审稿人会质疑这个值的合理性。

---

## 四、`flow_supervised.py` — angular loss 审查

### 4.1 基线实现是正确的 (第 32-39 行)

```python
cosine = torch.clamp(cosine, min=-1. + epsilon, max=1. - epsilon)
return torch.sum(torch.acos(cosine) * mask) / (num_valid_px + 1e-9)
```

使用 `torch.acos` 并且正确 clamp。这个实现是可靠的。

### 4.2 Import hook 注入的代码 (train.py 第 160-164 行)

```python
if self.lambda_ang > 0:
    ang_loss = self.angular_loss_function(flow, gt_flow, mask, num_valid_px)
    curr_loss += self.lambda_mod * mod_loss + self.lambda_ang * ang_loss
else:
    curr_loss += self.lambda_mod * mod_loss
```

注入逻辑是正确的——有 `lambda_ang > 0` 的分支判断。问题仅在 hook 是否能被触发（见 2.1）。

---

## 五、目标对齐检查

| 声称 | 实际 | 差距 |
|------|------|------|
| "零指数运算" | ✅ 确实没 exp | — |
| "零乘法器 (除 gate 应用外)" | ❌ gate*n 是乘法 | 一次额外乘法 |
| "纯 popcount" | ⚠️ 两次除法未计入 | 除法器成本 |
| "θ 与 attention 解耦" | ⚠️ Q 侧丢弃 θ，K 侧保留 | 不对称设计未解释 |
| "渐进稀疏调度" | ❌ 完全未实现 | 只是固定 activity_eta |
| "角度 loss 保护 AAE" | ✅ angular loss 函数正确 | hook 是否生效待验证 |

---

## 六、审稿防御性评估

### 审稿人1 (系统/架构): 关注点
- "你说的 progressive schedule 在哪里？"
- "gate * n 和 k_orig * gate 至少两次乘法，不是零"
- "除法器为什么不计入硬件成本？"

### 审稿人2 (理论): 关注点
- "丢弃 θ 的理论依据是什么？"
- "高稀疏下的共识退化你分析过吗？"
- "为什么 Q 侧丢 θ 但 K 侧保留？"

### 审稿人3 (实验): 关注点
- "只有一个数据集"
- "没有多 seed"
- "没有 H9a+angle loss 的消融"
- "I13 替换的 FFN 集合和 H9a 不一样，不可比较"

---

## 七、紧急修复清单 (按优先级)

### P0 — 训练结果出来前必须修复

1. **验证 angular loss hook 是否生效**
   - 在 log 中 grep `ang_loss` 或 `lambda_ang`
   - 如果未生效，在 LOSS_PATCH 中直接调用 `loss_function.angular_loss_function` 替代 import hook

2. **修复高稀疏退化**
   - 在 soc_gate 中加 temperature: `consensus = consensus / max(consensus.std(), 1e-6)`
   - 或者去掉分母中的 `silent`（只除活跃通道数）

3. **去掉 "progressive schedule" 声称**（未实现）

### P1 — 论文写之前必须修复

4. 诚实声明乘法器和除法器数量
5. 添加 H9a+angle loss 消融实验
6. 修复 FFN 目标组合与 H9a 对齐

### P2 — 有时间就修

7. 添加 consensus 退化监控 (std, min, max)
8. 清理无效的 SAVE_PATCH
9. Temperature sweep
