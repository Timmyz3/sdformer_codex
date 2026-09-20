# 全栈三元防护设计 + ATLIF 修复规范

- 版本：v1.0（2026-09-17）
- 基线：H67 ep35（neuron_type: psn, v_th: 0.1，DSEC valid825 AEE 1.3287）
- 资源：A800 80GB 单卡；DSEC 数据 `/root/private_data/SothisAI/dataset/Console/DSEC/main/DSEC/saved_flow_data`
- 代码根：`work/sdformer_codex/SDformer/`（下文所有相对路径以此为根）
- 证据基础：取证报告 C（ATLIF 定位，行号已逐行核实）；报告 A/B/D 缺失，其内容分别由 H67 ep35 checkpoint 侧已知事实、`neuron_experiments/H_SERIES_SUMMARY.md`（2026-05-12 实测表）与推导补齐，凡属推导处均已标注依据等级（[实测]/[推导]）。

## 0. 结论摘要

先修 ATLIF 梯度（它同时是 SOPs 收益的使能器），再按「gate 先行、carrier 不动」的三段式扩全栈。H10 的 AAE 71.6 不是三元本身的失败，而是「主 carrier 替换」的失败；本设计以五项防护（负发放 guard、per-layer 2 的幂 scale、carrier/gate 分离、分段阈值、angular loss）把三元严格限制在「神经元取值域」，从而绕开 H10 的三条失败机理。

## 1. 设计原则：两条路线的区分与五项防护清单

### 1.1 为什么 H10 会失败（失败机理，[实测]）

H10 全栈直接替换把「主 carrier」（承载连续幅值的信息主干：V/proj 输出、decoder 特征、flow head 回归输出）整体换成三值运算。三条机理：

1. **幅值坍缩**：光流是回归任务，三元码 {-1,0,+1}×s 只有 3 个幅值档位，carrier 被替换后幅值信息逐层坍缩，EPE 均值尚可掩盖方向错误，AAE 直接爆炸（71.6，正常区间 7–9）。
2. **无防护误差累积**：全栈一次性替换，每层引入的量化误差无 scale 重标定、无蒸馏、无方向约束，逐层放大。
3. **密度反升**：三元发放率远高于基线二值发放。实测：H2 adaptive ternary firing 0.204 / SOPs 8.68G（基线 E0 firing 0.085 / 3.62G）；H5b FFN 三元 firing 0.171 / SOPs 7.31G；H5c downsample 三元 firing 0.180 / SOPs 7.69G。三元扩到高发放层后 SOPs 不降反升，硬件收益为负。

### 1.2 路线区分（本设计的根本立场）

- **神经元取值域三元（本设计采用）**：只改神经元输出侧的离散码，out = ternary × thre，thre 即 per-layer scale；训练期权重与 carrier 全精度浮点，梯度经 STE 三角窗直通；三元只上 gate 侧（Q/K 打分、稀疏选择），并配归一化路径（BSA/Shiftmax）。
- **主 carrier 替换（H10 路线，已证伪）**：把幅值主干的三值化当成「更激进的量化」直接替换。二者在数学上的区别：前者是对膜电位的分段线性读出（可逆地保留幅值统计），后者是对信息载体的有损压缩（不可逆）。

### 1.3 五项防护清单

**P1 负发放 guard（neg_rate 监控与钳制）**。三元允许负发放（H9 bipolar 路线），但负发放破坏 SOPs 统计与硬件计数器语义。执行：训练与评估全程统计每层 neg_rate = P(ternary=-1)；上限默认 0.30；neg_scale 训练期 clamp 到 [0.5, 2.0]；任一层 neg_rate 超限即触发该段回退（见第 4 节门槛）。硬件侧配负发放计数器寄存器（第 5 节）。

**P2 per-layer 2 的幂 scale**。s_l = thre_l（修复梯度后 out = ternary × thre，thre 天然就是 scale）。部署前把每层 thre round 到最近的 2 的幂，单层相对误差上界 2 倍（相邻 2 的幂之比），在 finetune 最后 1 个 epoch 做 round-trip 校准（用 round 后的 thre 前向，微调其余参数）。收益：三元码与权重的乘法在部署侧退化为移位 + 符号选择。约束：scale 为 per-layer 标量（存一个寄存器），不做 per-channel，保住 RTL 面积。

**P3 carrier/gate 分离**。注意力中 Q·K^T 打分是 gate（信息是「选谁」，离散决策天然兼容三元），V/proj 是 carrier（承载幅值）。三元只上 gate；carrier 保持二值 spike 幅值或全精度。已证伪的边界也要记入：proj 三元属 carrier 侧侵蚀（H6b：SOPs 3.69G→4.52G，[实测]），默认不做。

**P4 分段阈值（非对称 pos/neg）**。沿用 ATLIFTernaryPSN 的 neg_thre = thre × neg_scale，正负通路独立阈值；每个全栈扩展段（FFN/patch merge/decoder/flow head）独立持有 threshold 组，禁止全局阈值联动——这是「分段」的两层含义。负侧阈值独立可调，避免对称强制导致负侧过发放（P1 的前置防线）。

**P5 angular loss**。L = L_EPE + λ·L_ang，L_ang = 1 − cos(û_pred, û_ref)，对有效像素均值；λ 默认 0.1，随段推进线性 warmup。依据：H6a all-params AAE 21.6、H8m AAE 22.8、H10 AAE 71.6，全部是 AEE 接近基线而角度/幅值一致性崩掉的形态（[实测]）——EPE 类损失约束不住方向，必须在扩展 carrier 侧操作时显式加角度项。

## 2. 分算子三元适配表

| 算子/模块 | 角色 | 适配方式 | 风险 | 依据 |
|---|---|---|---|---|
| PSN 神经元本体（ATLIFTernaryPSN） | 离散化执行点 | 三元 {-thre,0,+thre}，先修梯度闭环（第 3 节） | 低（修复后） | 报告 C [实测] |
| Q/K attention 打分 | gate | 三元码 × 2 的幂 scale，BSA/Shiftmax 完整归一化路径 | 中：只换神经元不换归一化会角度崩 | H6a/H8 结论、H11 [实测] |
| V / attention proj | carrier | 不三元；proj 二值化可作可选支线 | 高：proj 三元 SOPs +22% | H5a/H6b [实测] |
| stage0 FFN / MLP | carrier（高发放层） | binary ATLIF {0, thre}，禁三元 | 高：三元 firing 0.171 / SOPs 7.31G | H5b/H6 [实测] |
| downsample / patch merge | carrier（高发放层） | binary + per-layer 2 的幂 scale | 高：SOPs 7.69G | H5c [实测] |
| patch embed | carrier（幅值小、通道多） | 三元加法树可试（收益在硬件侧：缺失 RTL 变加法树） | 中 | [推导] |
| decoder | carrier（末端） | 受控三元：分段阈值 + carrier 输出保留连续幅值 + angular loss | 极高：H10 AAE 71.6 | H10 [实测] |
| flow head | carrier（回归输出） | 最后一段：内部三元、末端线性 scale 层全精度保幅值 | 极高 | H10 [推导] |

表的核心纪律：**三元密度与该层基线发放率负相关**——firing 越高的层越只能用 binary；三元只允许出现在「符号信息有价值且发放稀疏」的位置（Q/K gate、末段 carrier 的内部）。

## 3. ATLIF 梯度修复规范

目标文件：`neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py`（修复落点为其 H12 副本，见 3.4）。基线参照：`third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py`。

### 3.1 缺陷描述（四个变体全部有缺陷，[实测]）

结构性根因：四个变体的 forward 都只 `ctx.save_for_backward(input, thre[, neg_thre])`（第 27/49/72/102 行），**没有保存离散码**（ternary/active/out），backward 因此无法构造恒等直传项。共同缺陷写法：

- `grad_input = grad_input * tmp`（第 36/56/81/110 行）先覆盖原始上游梯度 g，随后第 37/57/82/111 行复用被覆盖的 grad_input，导致 tmp 被乘两次（tmp²）；
- Ternary/Binary/SymmetricBinary 三变体在缺陷行用了 `.abs()`，符号被毁，grad_thre ≤ 0 恒成立；
- OfficialATLIFSurrogate（第 111 行，**H67 ep35 主线引用的正是这个变体**）无 `.abs()` 但同样 tmp²，且偏离官方单 tmp 语义；
- 四变体全部缺失恒等项 d(out)/d(thre)|_ternary。

| 变体 | 类/缺陷行 | 缺陷代码 |
|---|---|---|
| TernarySurrogate | 15-38 / 37 | `grad_thre = -(grad_input.abs() * tmp).mean()` |
| BinarySurrogate | 41-58 / 57 | 同上 |
| SymmetricBinarySurrogate | 61-83 / 82 | 同上 |
| OfficialATLIFSurrogate | 86-112 / 111 | `grad_thre = -(grad_input * tmp).mean()`（tmp² 且单侧负） |

后果：Adam 通道（`h28_optimizer.py:79-80` 把 `.thresh` 归入 atlif_threshold 组，threshold_lr=5e-6）是阈值唯一闭环通道，而 grad_thre 恒 ≤ 0 使 Adam 单调推高 thresh → 发放率单调下降 → 训练期抬高的阈值经 deployment 链路（threshold_eta=0 时 installer.py:78/253/316 映射的 sp=0，forward 增量恒 0；部署纯推理无 backward）**永久固化进部署模型**。若观察到发放率随 checkpoint 迭代单调下降、AEE 变差，可直接归因于此。

### 3.2 数学依据

out = thre · H(input − thre)，对 thre 求导：d out/d thre = H + thre · dH/d thre。第二项经 zif_backward（第 11-12 行）三角窗给出 −tmp（|slope|=1/thre，乘 thre 后量级 1），即现有负项的来源；缺失的正是 H 本身，即 +(g · ternary).mean()。修正后 grad_thre 在发放率低时为正（阈值回落）、发放率过高时为负（阈值抬升），形成真正的双向闭环。

### 3.3 修复规范（精确到文件:行号）

统一改法（四变体同构）。以 TernarySurrogate 为例：

```python
# forward：仅改第 27 行（ternary 已在第 23 行算出）
ctx.save_for_backward(input, thre, neg_thre, ternary)   # 原：ctx.save_for_backward(input, thre, neg_thre)

# backward（第 30-38 行整体替换）：
@staticmethod
def backward(ctx, grad_input: torch.Tensor, _dummy):
    input, thre, neg_thre, ternary = ctx.saved_tensors
    g = grad_input                    # 保留原始上游梯度，禁止在求 grad_thre 前覆盖
    pos_tmp = (1.0 - ((input - thre) / thre).abs()).clamp(min=0)
    neg_tmp = (1.0 - (((-input) - neg_thre) / neg_thre).abs()).clamp(min=0)
    tmp = torch.maximum(pos_tmp, neg_tmp)
    grad_thre = (g * ternary).mean() - (g * tmp).mean()
    return g * tmp, grad_thre, None, None
```

各变体差异（要点：去掉 `.abs()`；grad_input 只乘一次 tmp；grad_thre 用未掩码的 g；新增项符号为正、量级与 spike 幅值同阶）：

| 变体 | forward 保存（行） | backward 新公式 | 返回签名 |
|---|---|---|---|
| TernarySurrogate | 27 行保存 23 行 ternary | grad_thre = (g·ternary).mean() − (g·tmp).mean() | g·tmp, grad_thre, None, None |
| BinarySurrogate | 49 行保存 46 行 active | grad_thre = (g·active).mean() − (g·tmp).mean() | g·tmp, grad_thre, None |
| SymmetricBinarySurrogate | 72 行保存 68 行 active（max(pos,neg)） | 同 Binary | g·tmp, grad_thre, None |
| OfficialATLIFSurrogate | 102 行保存 100 行 out（0/1 掩码） | grad_thre = (g·out).mean() − (g·tmp).mean()（恢复官方单 tmp 语义并补恒等项） | g·tmp, grad_thre, None |

### 3.4 落点与执行流程

约束：`hw_autoresearch_nts07/autoresearch.md:42-44` 禁止修改 `neuron_experiments/**/overlay/**`；H9 overlay 被 1000+ 生成配置引用，且 H67 ep35 的 checkpoint 记账（capture 脚本对 atlif_impl 的 path/sha256）会因原地修改变更而作废。

执行流程（两步走）：

1. **短程验证（monkey-patch，非最终落点）**：在 `install_atlif_ternary_psn` 之前用自定义 autograd.Function 替换 ATLIFTernaryPSN.act，跑 1–2 个短程对照实验拿阈值曲线证据。注意：monkey-patch 会让 checkpoint 内实现与运行期实现不一致，破坏 receipts，不得作为最终落点。
2. **正式落点**：新建 `neuron_experiments/H12_atlif_gradfix/`（configs/ docs/ entrypoints/ overlay/ tests 同构），复制 H9 overlay 的 `models/STSwinNet_SNN/atlif_ternary_psn/{__init__,atlif_ternary_psn,installer,training}.py`、`bsa_attention.py`、`h28_optimizer.py`，在副本上按 3.3 改 4 个 backward + 4 处 save_for_backward；复制 H9 `entrypoints/train.py` 作为 H12 入口。

随行修复（E2 anchor fail-fast）：E2 `entrypoints/train.py:179-190` 的裸 `source.replace(...)` 中，LOSS_REGULARIZATION_ANCHOR 在当前基线实测 count=0，replace 静默 no-op、正则失效。将 183-190 的 8 处 replace 改为带期望次数的 fail-fast 替换：新增 `_replace(source, anchor, replacement, name, expected)`，count 不符即 RuntimeError 并打印基线 sha256；逐个调用 TRAIN_BLOCK=1、LOSS_REGULARIZATION=1、PIN_MEMORY=2、CONFIGURE_BACKEND=1、BACKWARD=1、SCALER_STEP=1、OPTIMIZER_STEP=1；再补后置断言：补丁串必须出现 regularize_activity(、threshold_update(、sanitize_threshold_grads(。此修复一并搬进 H12 入口。

系统级第二因（与梯度修复同组评估）：`installer.py:691-692` 的 target_rate 反馈在 upper_bound 模式下 rate_error = max(r−target, 0)，r 低于目标时反馈恒 0，无法抵消 Adam 单向抬升。G1 组同时切 `target_rate_mode=bidirectional` 做对照。

### 3.5 回归判据（单元测试，合入 H12/tests）

单模块合成测试：thresh 初值设偏大（输入无发放），修正版 grad_thre 应 > 0，若干步 SGD 后 thresh 下降、发放率 r 上升至 target_rate 附近；当前缺陷代码该场景 grad_thre 恒为 0 或负。另加四变体梯度数值检查：有限差分 d(out)/d(thre) 与 backward 输出符号一致（容忍 STE 窗口内的区间差异）。

### 3.6 超参重标定

补上 (g·ternary) 项后，阈值梯度量级与 spike 幅值同阶。现配置 `threshold_base_lr=3e-6 × threshold_lr_scale=50000 → 每步 update×0.15`（configs/generated/date11_allbinary_tx_ft_ep19_deploy_float_ref.yml:81-82）很可能过大导致阈值震荡。G1 组先对 threshold_lr_scale ∈ {5e3, 1.5e4, 5e4} 短程扫参（对应每步 0.015/0.045/0.15），h28_optimizer.py:108 的 threshold_lr（5e-6）暂不动。

## 4. 实验矩阵

| 组 | 变量（相对上一组） | 对照组 | 步数 | 评估协议（valid825） | 通过门槛 | 回退门槛与动作 |
|---|---|---|---|---|---|---|
| G0 基线复现 | 无（H67 ep35 原样） | — | 仅推理 | AEE/SOPs/firing/neg_rate | AEE = 1.3287 ± 0.01 复现 | 复现失败先查环境，不进后续组 |
| G0b 梯度单元验证 | monkey-patch 修正版 backward，不训练 | 缺陷版同配置 | 合成测试 + 1 ep 微调 | grad_thre 符号、thresh 均值曲线、r 曲线 | 无发放场景 grad_thre > 0，thresh 下降、r 回升至 target 附近 | grad_thre 恒 ≤ 0 → 修复无效，回查 3.3 实现 |
| G1 闭环验证 | H12 修正 backward（正式落点）+ threshold_eta=0.1 + target_rate_mode=bidirectional；threshold_lr_scale 扫 {5e3, 1.5e4, 5e4} | 同配置未修 backward（复刻缺陷版） | 2–3 ep probe → 10 ep | AEE/SOPs/firing/neg_rate + thresh 均值曲线 | AEE ≤ 1.345；SOPs 较 G0 降 ≥ 10%；thresh 均值回落至 [0.5, 1.5] 区间；全层 neg_rate ≤ 0.30 | AEE > 1.40，或 SOPs 上升，或 thresh 仍单调上升 → 冻结扫参重试一次；仍败则回退 G0b 仅保留单元级证据 |
| G2 QK gate 三元 | 最优 G1 + Q/K 三元（BSA + Shiftmax 完整归一化），carrier（V/proj）不动 | G1 最优 checkpoint | 2–3 ep probe → 10–20 ep | AEE/SOPs/firing/neg_rate + AAE（angular guard） | AEE ≤ 1.40；SOPs 较 G1 再降 ≥ 15%；AAE ≤ 9.0 | AEE > 1.50 或 AAE > 12 → 加 angular loss λ=0.1 warmup 重试一次；仍败 → 回退 G1，论文 claim 收缩为 ATLIF 闭环故事 |
| G3a FFN binary | G2 + stage0 FFN binary ATLIF（禁三元） | G2 最优 | 2–3 ep | 同 G2 | AEE 退化 ≤ 3%（相对 G2）；SOPs 再降 ≥ 10% | 退化 > 5% 或 firing 上升 → 撤该段，跳到 G3b |
| G3b patch merge binary | G3a + downsample/patch merge binary + per-layer 2 的幂 scale round-trip 校准 | G3a | 2–3 ep | 同 G2 | AEE 累计退化 ≤ 6%（相对 G2）；SOPs 累计降 ≥ 20%（相对 G1） | 同 G3a；另查 neg_rate ≤ 0.30 |
| G3c decoder 受控三元 | G3b + decoder 分段阈值三元（P4 非对称阈值 + P5 angular loss），逐层分段解冻 | G3b | 3–5 ep，逐层 gate | 同 G2 + 每层 neg_rate | AEE 累计退化 ≤ 10%；AAE ≤ 12 | AAE > 15 或任一层 neg_rate > 0.30 → 撤 decoder 段，止步 G3b |
| G3d flow head | G3c + flow head 内部三元、末端线性 scale 层全精度保幅值 | G3c | 3–5 ep | 同 G2 | 最终 AEE ≤ 1.55；SOPs 较 G0 降 ≥ 40%；AAE ≤ 10 | AEE > 1.65 → flow head 保全精度，止步 G3c |
| GH 硬件综合 | 最终通过组的 checkpoint → operator 统计 → DC/PTPX | m931 基线（≈80150 μm²，WNS −4.91 ns，6.25 mW） | 综合 + 后仿 | 面积/WNS/功耗 vs m931 | WNS ≥ 0；logic-only 面积增幅 ≤ 15%（三元编码器+计数器）；动态功耗 ≤ 6.25 mW | WNS 仍为负 → 降频至 250 MHz 重综或缩减三元覆盖段 |

## 5. 硬件适配影响

真 RTL 在 `hw_autoresearch_nts07/`（2329 个 .sv）；现有综合基线：TSMC 28nm HPC+ DC 3 ns/333 MHz，logic-only 3902 μm²，macro-aware m931 ≈80150 μm² 但 WNS −4.91 ns 未收敛，PTPX 6.25 mW。本设计的硬件收益分四条：

1. **三元 2-bit 通路替代 15-bit 位串行决策**。现有 PSN 累加-比较路径按 15-bit 位串行实现发放决策，是 m931 关键路径的一部分；三元输出 {-1,0,+1} 只需 2-bit（符号位 + 有效位）编码，膜电位越限判断退化为 2-bit 符号逻辑，直接缓解 WNS 负裕量，m931 的 −4.91 ns 是首要受益点。
2. **乘法变符号选择 + 加法**。Q/K 三元化后，attention 内积 q·k 的每个乘积项 ∈ {-1,0,+1}×{-1,0,+1}，等于符号异或 + 零掩码，累加退化为加减法器（2's complement sign-select add），无 DSP 乘法器；配合 P2 的 2 的幂 scale，scale 乘法进一步退化为移位。三元权重 MAC（第 3 条）同理。
3. **缺失 RTL 变加法树**。patch embed / MLP / PSN 目前没有独立 RTL；三元化后这些模块的乘加退化为加法树（幂零加减），可以直接以加法树形式综合补齐，避免为全精度乘法补 DSP/macro，面积与功耗同时受益。
4. **新增小单元**：三元编码器（双比较器 + 符号位，每神经元）、per-layer 2 的幂 scale 移位器（每层一个常量移位寄存器）、负发放计数器（在线统计 neg_rate，为 P1 guard 提供部署期观测点）。

验收口径：G3 通过后对最终 checkpoint 重新生成 operator 统计并跑 DC/PTPX，报告 logic-only 与 macro-aware 两组面积/时序/功耗，与 m931 基线同表对比；WNS 收敛（≥0）应作为硬件侧主 claim 之一。

## 6. 风险与回退

1. **历史复现基线作废**：修复 grad_thre 改变所有含 ATLIF 的历史 checkpoint 复现语义；必须新建 H12 目录在副本上改（autoresearch.md:42-44 禁改 overlay），绝不原地改 H9——H9 被 1000+ 生成配置引用，且 H67 ep35 的 capture 记账（atlif_impl path/sha256）会因原地修改变更而作废已发布数字；OfficialATLIFSurrogate 恰是 H67 ep35 主线实现，其修复需单独 receipt 与 A/B 对照。
2. **梯度量级重标定**：补 (g·ternary) 恒等项后 grad_thre 与 spike 幅值同阶，现有 threshold_base_lr=3e-6 × threshold_lr_scale=50000（每步 update×0.15）可能过大导致阈值震荡甚至发散；G1 必须先做 scale ∈ {5e3, 1.5e4, 5e4} 扫参，不通过则回退。
3. **系统级无闭环第二因**：installer.py:691-692 的 target_rate_mode=upper_bound 使 rate_error = max(r−target, 0) 单向截断，r 低于目标时反馈恒 0，无法抵消 Adam 单向抬升；若 G1 不同步切 bidirectional 对照，梯度修好后仍可能只有单向动力学，闭环验证（thresh 回落、r 回升）将失败。
4. **全栈扩展 AAE 爆炸**（前科：H10 71.6、H6a all-params 21.6、H8m 22.8）：decoder/flow head 是 carrier 侧操作，必须严格走 carrier/gate 分离 + 分段阈值 + angular loss + 每段独立门槛（AAE ≤ 12/10），任一段超限立即回退到上一段终点，不整栈作废。
5. **部署侧固化与评估链路不一致**：threshold_eta=0 时部署评估（eval_m260/m263/m266 同构链路）为纯推理、阈值固化常量，训练期阈值动力学完全不可见；所有 SOPs/发放率结论必须用与部署同构的 install_atlif_ternary_psn 重新加载后复测，且 E2 侧 LOSS_REGULARIZATION anchor 已实测 count=0 静默失效——锚点 fail-fast 修复必须随 H12 一并落地，否则正则通道形同虚设。

回退总原则：单变量推进 + 每段独立回退点 + checkpoint 命名带段标识（G1/G2/G3a–d），任何一段失败只回退到上一段终点，不整栈作废；H9 overlay 与 H67 ep35 发布数字永远不动，全部实验在 H12 副本上进行。
