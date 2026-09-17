# T13a：τ/BN 矩来源查证（2026-09-15）

> 回应 Codex 审计（shared_execution_20260915/claude_review）点名问题：
> "原脚本从当前完整 Y 计算动态 BN 矩和 τ，没有证明是训练期冻结常量"。
> 脚本：`t13_tau_provenance.py`；数据：`results/t13_tau_provenance.json`。
> 只读 checkpoint/traces，未修改生产树。

## 1. 事实链（checkpoint 直接证据）

部署模型 Motion C12 ep34
（`SDformer/hw_autoresearch_nts07/system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth`）：

- **12 个 `mlp.bn1.norm_layer` 全部带 running_mean / running_var / num_batches_tracked
  （=312120）**——标准 track 模式 `nn.BatchNorm2d`（模型定义见
  `SDformer/neuron_experiments/E1_exp_sn/overlay/models/STSwinNet_SNN/Spiking_modules.py:131`，
  norm='BN' 分支；若为 BN_notrack 分支则 state dict 不会有这三个 key）；
- γ/β/bias/center/A/θ 全部为 checkpoint 常数（trace 内数值与 checkpoint 逐值一致已断言）；
- eval 模式下 BN 使用 running stats → **部署 τ(t,h) 是纯静态常数**，可入 ROM。

## 2. 双口径 τ 对照（每 trace × 20000 组）

tau_dyn = T5 原口径（trace 全域 Y 矩）；tau_ck = checkpoint running stats。

| trace | τ 相对差 med / p90 | 48bit thr 相同 | 判决翻转率 | BF+cert 拍比(ck) | 拍比(dyn) |
|---|---|---|---:|---:|---:|
| s0_stage0 | 4.5% / 12.5% | 否 | 0.456% | **17.51%** | 17.55% |
| s0_stage3 | 5.7% / 14.1% | 否 | 0.911% | **17.42%** | 17.40% |
| s10_stage0 | 4.5% / 11.9% | 否 | 0.544% | **17.37%** | 17.29% |
| s10_stage3 | 6.4% / 17.0% | 否 | 1.038% | **17.59%** | 17.66% |

两口径下证书 vs 全深度整数判决均零差（断言通过，`cert_vs_full_mismatches=0`）。

## 3. 结论

1. **供数结论与 τ 来源无关**：冻结 τ 下 BF+证书拍比 17.37–17.59%，与 T5 动态 τ 口径
   （17.3–17.7%）差异 <0.1pp。且 τ 相对差 4.5–6.4% 落在 T6 已测扰动带（3–7%，
   拍比 16.5–17.7%）内，两实验互相印证。
2. **审计批评成立且已闭合**：部署态 τ 应取 checkpoint running stats（静态常数），
   T5 的动态矩只是近似代理。thr 表（10×H×48bit ROM）为部署工件，
   **不产生逐组阈值供数费用**——Codex cert_transport 计费里"每组 10 个阈值字"
   在部署口径下可摊销为一次性 ROM 装载。
3. **部署 τ 选择会改变网络输出**（判决翻转 0.46–1.04%）：硬件部署 τ 须与
   AEE 评测协议一致。
   ✅ 已闭合（2026-09-15 二次核查，**修正早先结论方向**）：官方
   `eval_DSEC_flow_SNN.py` 虽有 `model.eval()`（:502），但随后 ：508 施加
   bn_policy，且两个参考 AEE 的记录均为 **`bn_policy="no_running"`**——
   NB0 1.445353（open_fusion_execution/accuracy_baseline/source_metadata.json）
   与 motion ep34 粗头 1.081367（algorithm/run_bn_probe.set_bn_mode 默认
   chosen=() → 全部 BN 置 track_running_stats=False）。
   **即参考 AEE 全部是当前域动态 BN 口径 → 硬件须用动态/流式 τ（逐域矩，
   即 s2/TSBG 流式统计线所做）才与已测 AEE 对应；tau_ck（checkpoint
   running stats）反而引入 0.46–1.04% 判决偏离。**
   本链影响：T5/T17 的 tau 本就是逐 trace 终态矩（dyn 口径）→ 供数/判决链
   与 AEE 协议一致；T13a 第 (2) 条"硬件须用 tau_ck"作废。
   供数结论不受影响（T6：thr 移 0.5–4 倍拍比 16.5–17.7%；T13a 冻结 τ
   拍比 17.37–17.59%）。
4. bn_state 目录的"动态统计"探索线（Gram/A8）是另一条机制线的受限结论
   （"正结果只支持必须保留动态统计的受限条件"），与本结论不冲突：
   部署网络本身的 BN 是 track 模式。

## 4. 对净服务计费（T13b）的影响

- 阈值侧：thr 一次性 ROM，每组 0 拍（原计费口径 +10 字/组 → 0）；
- 数据侧：Y 供数 17.4%（BF+cert）vs 24 拍 FX 基线不变；
- 剩余待计费项：sop 头拍（1 拍/组，已含在 17.4% 内）、BN 折入后无额外矩计算拍。
