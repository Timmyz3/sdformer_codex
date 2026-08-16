# H66d Local-5 主线定点与 RTL 签核

**日期**：2026-07-25  
**checkpoint**：epoch29  
**mode**：`binary_axnor_local5_shiftmax`

> **2026-07-26 状态更正**：本文的 `RTL-exact Shiftmax` 数字来自 Python
> RTL-like 部署模型，不是 Local5 SystemVerilog 逐拍等价签核。后续独立审阅
> 发现边界 mask、Shiftmax Q1.7 缩放和 score RNE 三项软件/Python/SV 合同
> 不一致；在修复并重跑 valid825 前，`AEE=1.4486` 只能作为候选精度证据，
> 不能作为硬件主线已闭环证据。详细审计见
> `docs/150_Local5与H67硬件切线审计及架构创新候选_20260726.md` 和
> `docs/151_DATE独立预审_Local5切线与架构候选_20260726.md`。

## 1. 部署图

```text
binary ATLIF -> Local-5 stencil (self+N/S/E/W)
  alpha-XNOR score (alpha0=1/64)
  Q7 quant -> Shiftmax5 (float 2^x or RTL LUT)
  Q1.7 gate -> sum_j gate_j * K_j
```

无效边界候选 mask 到 score_min（定点）或 -1e4（训练浮点）。

## 2. valid825 结果

| 路径 | AEE | AAE | spikes(G) | energy_proxy(uJ) |
|---|---:|---:|---:|---:|
| float rank-1 (ep29) | 1.4432 | 9.4012 | 27.0403 | 23976.31 |
| dyadic INT8 | 1.4475 | 9.3860 | 26.5517 | 23550.64 |
| RTL-exact Shiftmax | 1.4486 | 9.4210 | 26.5340 | 23535.19 |

| RTL − dyadic AEE | +0.0011 |

## 3. 与 H67 对照（同协议）

| 方法 | float AEE | dyadic AEE | RTL AEE |
|---|---:|---:|---:|
| H67 Motion-XOR | 1.4671 | 1.4626 | 1.4627 |
| H66d Local-5 | 1.4432 | 1.4475 | 1.4486 |

## 4. 主线判定

- **软件精度主线**：float 已是 H66d；若 dyadic 仍优于 H67 dyadic 1.4626，则部署精度主线切 H66d。
- **硬件主线**：需 Stencil-5 row engine RTL（见 `rtl_local5/`），不可复用 H67 Motion-XOR top 冒充。
- spike energy 仍为 proxy，不含 halo/gather/Shiftmax5 控制。

## 5. 产物路径

- dyadic json：`/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid/h66d_epoch29_dyadic_int8_valid825.json`
- RTL json：`/root/private_data/work/sdformer_codex/SDformer/hw_autoresearch_nts07/results/h66d_local5_rtl_exact_valid825.json`

## 6. Fullres checkpoint-bound 重签核状态（2026-08-05）

本页第2--3节是历史 crop epoch29 结果，不能直接作为当前 `480x640 / T=2 / 15x15`
公平训练 checkpoint 的 RTL 签核。当前 fullres Local-5 训练目录为：

`neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805`

正式硬件证据按下表 fail closed；报告未落盘前统一标记 `queued`：

| 证据 | 绑定范围 | 当前状态 |
|---|---|---|
| standard valid825 | ep9/14/19/24/29，选择 rank-1 | queued |
| post-G0 profile100 | rank-1、T450、all12、真实 gate/K | queued |
| score/Shiftmax RTL | rank-1 checkpoint、真实 score trace | queued |
| projection RTL | rank-1 checkpoint、真实 dyadic INT8 weight/bias | queued |
| ATLIF temporal RTL | rank-1 checkpoint、81个 live sites | queued |
| config identity | ep9 optimizer/scheduler/scaler + launch receipt | pending ep9 |

只有三类组件 RTL 报告 checkpoint SHA 一致、profile/acceptance 与训练配置身份一致时，才允许写
`checkpoint_bound_component_rtl_exact`。这仍不是 full-network RTL exact；完整网络 claim 继续禁止。
H67 ep30 和追加训练后的 rank-1 使用同一证据颗粒度，不复用旧 crop bit trace。

## 7. 当前运行状态（2026-08-05 20:50 CST）

- fullres Local-5 已完成 ep0--4，当前 ep5 约13%；首个 model/state 锨点仍为 ep9。
- checkpoint-bound producer PID2286750 正常等待，未提前复用本页历史 crop ep29 的证据。
- 完成后将对 ep9/14/19/24/29 做 standard valid825，选 AEE rank-1 后一次性生成
  T450/all12 profile 及三类同 SHA 组件 RTL 报告。
