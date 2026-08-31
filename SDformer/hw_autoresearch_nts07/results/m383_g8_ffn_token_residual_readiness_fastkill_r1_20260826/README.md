# M383 / G8：FFN whole-token residual skip readiness 与 CPU fast-kill

日期：2026-08-26  
结论：`NO_GO_G8_FROM_EXISTING_ASSETS__GO_MINIMAL_A800_STREAMING_CAPTURE_WHEN_IDLE`

M383 全程只读、CPU-only；没有启动 A800、没有训练，也没有修改任何既有冻结证据或 `docs/359`。

## 结论

现有数据不足以计算 G8 的 tau=0 或任何 residual-mass 阈值 skip-rate。缺失的不是 sample 身份或 FFN 几何，而是唯一关键张量：带 token 身份的完整 post-BN2 FFN residual `F(x)`。

- ordered S10 trace 已冻结 12 个 FFN、120 个动态 FFN group、`sn1→fc1→sn2→fc2` 顺序、shape、MAC、sample/sequence；但 `bit_trace_records=0`，`activation_records` 中 MLP 数值记录为 0。
- M32 的 100 条记录是 tensor SHA / 对象与 storage 同一性，其中仅 20 条 stage3 `sn1→fc1` FFN wiring 记录；没有数值 payload 或 token ID。
- M233 有 10 样本、24 个 BN、240 条 sample/module 记录和 264 个 NPZ array，但全部是按 `T,N,H,W` 归约后的逐 channel gamma/beta/mean/variance/min/max 等统计。token 联合向量已经丢失，无法恢复 exact-zero 或 residual mass。
- M248、M252、M73 的 payload 全部只覆盖四个 bottleneck Conv 输入；PAFT/control/train 与 H67 no-running FFN 语义也不匹配。
- M366 尚无 capture output，且 hook 只装在 `ATLIFTernaryPSN`，看不到完整 MLP residual。

因此本次没有用 aggregate density、BN channel extrema 或 tensor SHA 猜 token skip。结构化结果中的 tau0、严格阈值、source/MAC saved、范数预算和 Delta-AEE 均保持 `null/false`。

## 已冻结、可直接复用的几何

| Stage | Blocks | C | FFN tokens/sample | Dense FFN MACs/sample |
|---:|---:|---:|---:|---:|
| 0 | 2 | 96 | 384,000 | 28,311,552,000 |
| 1 | 2 | 192 | 96,000 | 28,311,552,000 |
| 2 | 6 | 384 | 72,000 | 84,934,656,000 |
| 3 | 2 | 768 | 6,000 | 28,311,552,000 |
| 合计 | 12 | — | 558,000 | 169,869,312,000 |

S10 总计 5,580,000 个 FFN token、1,698,693,120,000 dense FFN MAC。M159 当前计算 envelope 中 FFN Linear+ATLIF 为 205,384,111 cycles/frame，BN/residual 尚未计入。这些数字只用于未来 mask 的加权几何，不产生当前 skip-rate 或 speedup。

## 最小 A800 hook

未来 GPU 空闲时应使用冻结 H67 ep35 checkpoint：

- checkpoint SHA：`4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158`
- config SHA：`8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49`
- checkpoint load：`missing=0 / unexpected=0`
- BN：`no_running` / current-batch，必须与 ordered H67 profile 相同
- `eval()`、dropout/drop-path inactive，每个 sample 前 `functional.reset_net(model)`

目标是全部 12 个：

```text
sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.{block}.mlp
stage blocks = [2, 2, 6, 2]
```

最小 streaming hook：

1. full `.mlp` pre-hook：`x_branch_input [T,N,H,W,C]`，保留每 token `x_l1`。
2. `.mlp.sn1`：保留每 token `sn1_nnz`。
3. `.mlp.sn2`：保留每 token `sn2_nnz`。
4. `.mlp.fc2`：保留 pre-BN2 `l1`，用于证明 BN2 不能被忽略。
5. full `.mlp` output hook：捕获真正的 `F(x)`，即 BN2 后、父 block residual add 前；流式保留 `exact_zero/f_l1/f_l2_sq/f_linf/finite`。

token 身份冻结为：

```text
(checkpoint_sha, bn_policy, sample_id, sample_key, sequence_key,
 stage, block, t, n, h, w)
```

不需要落盘完整 float32 residual；按 `[T,N,H,W]` C-order 隐式 token 顺序保存压缩 metric arrays 即可。

## 阈值、proxy 与可证误差

定义：

```text
rho = ||F_token||_1 / max(||x_token||_1, 2^-24)
```

冻结阈值：`0, 2^-16, 2^-14, 2^-12, 2^-10, 2^-8, 2^-6`。tau=0 只有 post-BN2 `F(x)` 每个 channel 数值精确为零且 finite 才跳过。

未来 oracle mask 下：

- source-work proxy：`sn1_nnz*(4C) + sn2_nnz*C`
- dense MAC envelope：每 skipped token 为 `8*C*C`
- `y=x+F, y'=x`，所以局部 `Delta y=-F` 精确成立
- cohort `L1=sum(f_l1)`、`L2=sqrt(sum(f_l2_sq))`、`Linf=max(f_linf)` 都可直接出具回执
- threshold-only bound：每个 selected token `||Delta y||1 <= tau*max(||x||1,2^-24)`

但 `F(x)` mask 是 post-compute oracle：它能给理想 envelope，不能直接证明同一 token 的可执行 hardware skip。需要另行证明 pre-compute predictor/certificate。

## S10 与 valid825

- S10：先做 opportunity/threshold sweep，不声明 Delta-AEE。
- 只有 S10 出现非平凡 Pareto 后，才做 frozen valid825 paired baseline 与 oracle-output-replacement；必须保存 sample/order/output SHA、DSEC-FL/AEE 回执。
- 当前没有 FFN whole-token skip 的冻结 valid825 output，因此本次明确禁止 Delta-AEE 声明。

## M366 是否能复用

结论是部分复用，不能原样运行。

可复用 exact-SHA、S10 loader、checkpoint load audit、no-running BN、每 sample reset、streaming hook 生命周期和四次连续 GPU idle guard。必须新建 M383 capture class/contract；不能复用 M366 的 ATLIF-only attach filter、RemainingBudgetCapture、static-site tables、result schema 或 promotion gates。

## 打铁评分

- 总分：58/100
- audit evidence integrity：98
- identity/geometry readiness：94
- token residual data readiness：0
- CPU threshold sweep readiness：0
- minimum hook specification：96
- valid825 accuracy readiness：0
- P0/P1/P2：1 / 4 / 4

P0 是 post-BN2 token residual 全面缺失，直接阻断任何现有数据上的 G8 sweep。

## 复跑

从仓库根执行：

```bash
/opt/anaconda3/bin/python \
  hw_autoresearch_nts07/system_simulator/scripts/analyze_m383_g8_ffn_token_residual_readiness_fastkill.py \
  --contract hw_autoresearch_nts07/contracts/m383_g8_ffn_token_residual_readiness_fastkill_contract_r1_20260826.json \
  --output-dir <new-empty-output-dir>
```

结构化结果为 `m383_g8_ffn_token_residual_readiness_fastkill_r1.json`；逐资产 readiness 表为 `asset_readiness.csv`。
