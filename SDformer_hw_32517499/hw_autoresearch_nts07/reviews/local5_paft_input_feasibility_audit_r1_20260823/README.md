# Local5 独立 PAFT 输入可行性审计 R1（2026-08-23）

## 结论先行

**Local5 ep44 建立独立 PAFT 线在结构上可行，但现在不能生成 train-only catalog，更不能直接启动训练。** 当前输入就绪度为 **35/100（流程就绪度，不是论文性能分）**。

可以直接复用的是四个 bottleneck Conv 的捕获机制：Local5 ep44 与 H67 ep35 的模块名、顺序和几何完全相同，均为四个 `Conv2d(768,768,3x3,stride=1,padding=1,bias=false)`；输入 shape 也是 `T10×B1×C768×H15×W20`，因而 M40 的 hook/writer、`I_KY_KX` 展平、k16 的 6,912 feature / 432 partition 逻辑可复用。

绝不能复用的是 H67 的 catalog、trace 身份和 loader 合同：当前 M71/M73/M77 链严格绑定 H67/motion。M71 原 catalog 还是已撤销的 valid825 泄漏产物；真实 M73 train trace 和真实 M77 catalog 目前均不存在。Local5 需要新 schema、新 checkpoint/config/operator weight SHA、新 train trace、新 catalog 和新 external admission contract。

## 已确认的 Local5 输入

| 输入 | 审计结果 |
|---|---|
| 选中 checkpoint | `checkpoint_epoch44.pth`，591,167,684 B，SHA `19820bec...c34f57`，本机唯一 Local5 checkpoint |
| 训练配置 | `dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50.yml`，SHA `c5d7be62...f093a`，`binary_axnor_local5_shiftmax`，hardware quant=false |
| hardware-order 配置 | `...hardware_order_q7q17_deploy.yml`，SHA `078bb517...f514d`，同 Local5 mode，hardware quant=true |
| release 身份 | ep44 为已选 best epoch；receipt 状态 PASS；float valid825 AEE `1.2818928578`，hardware-order AEE `1.2804235696` |
| valid list | 存在，825 条、825 unique，SHA `7f3dc280...9325d0`；只能做 accuracy/overlap denylist |
| train list | **缺失**；期望 7,345 条、18 sequences、SHA `919c79c6...fdc10` |
| Local5 train tensors | 本机只有当前 valid 子集文件；catalog cohort 所需 train event/GT/mask 未就绪 |
| Local5 train-only packed source trace | **不存在** |
| 真实 M77 catalog | **不存在（H67 也没有）** |

## 四个 bottleneck operator 与几何

四个名称和顺序为：

1. `sttmultires_unet.resblocks.0.conv1.0`
2. `sttmultires_unet.resblocks.0.conv2.0`
3. `sttmultires_unet.resblocks.1.conv1.0`
4. `sttmultires_unet.resblocks.1.conv2.0`

审计器不依赖本机 torch，直接从 SHA-pinned ep44 PyTorch zip checkpoint 的 module/tensor metadata 读取并核对：四者全是 `[768,768,3,3]` float32、groups=1、stride/padding/dilation=`[1,1]`、无 bias，每组 weight 21,233,664 B。Local5 ep44 的四组 weight SHA 分别为：

| operator | Local5 ep44 weight SHA | H67 ep35 weight SHA | 内容相同 |
|---|---|---|---|
| resblock0 conv1 | `59474dc5...2bbeda` | `b07ebeb1...0cd63` | 否 |
| resblock0 conv2 | `816f8105...c41d3` | `ceb9741d...aa2e` | 否 |
| resblock1 conv1 | `9ff78db9...a51f` | `e2b075c7...db285` | 否 |
| resblock1 conv2 | `cf479eb6...e58a5` | `3ae76c9e...a1040` | 否 |

因此，**operator name/geometry gate 会通过，但 checkpoint/weight identity 绝不会通过**。这正是“复用机制、不复用模型证据”的边界。

## 可复用 trace 入口与已有 trace 的边界

可复用入口有两层：

- `profile_nts11_hardware_p0.py --ordered-trace --dual-line-trace` 已能为 Local5 生成完整网络 execution/operator ledger；
- `trace_m40_bottleneck_packed_sources.py::PackedBottleneckWriter` 已能对上述四个 Conv 输出 exact packed positive/negative/change bitplanes 和 weight identity；M73 在其上实现 train split 选择和 overlap audit。

现有 `results/local_ep44_full_network_ordered_trace_s10_20260821/` 有 1,720 execution records、4,030 dual-line operator records，四个目标算子各覆盖 10 次，load audit 为 0/0。它适合作为“Local5 模块存在、顺序和 shape 正确”的入口证据，但不能做 PAFT catalog，原因有三项：

1. 十个 sample key 精确等于 `valid_split_seq.csv` 前十条，属于 valid825 internal；
2. execution/operator CSV 只有聚合 activity/source-work，不包含 PAFT 所需的逐 Conv3x3 输入 packed bitplanes；
3. profile 内是 checkpoint path 而非可独立准入的 Local5 train-trace schema/manifest/checkpoint SHA 合同。

因此，仓库目前**没有 Local5 train-only source trace**。

## M71/M73/M77 为何严格绑定 H67/motion

| 层 | H67 绑定 |
|---|---|
| M71 builder | 固定读取 H67 ep35 M40 manifest（SHA `e743...`），固定 calibration sample 0–4，固定 4×432×q16；该 cohort 后被证明来自 valid825 |
| M73 tracer | 导入 M40 的 H67 checkpoint/config SHA；输出 schema 固定为 `m73_h67_ep35_train_calibration_packed_source_trace_v1`；launcher 路径、config、checkpoint、结果目录均含 H67 |
| current PAFT/M77 loader | 只接受 `m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1`、H67 ep35 checkpoint SHA `4f33...` 和 H67 M73 trace schema |
| H67 config semantics | H67 `bsa_attention.mode=h60`、`binary_motion_xor_alpha=0.25`；Local5 为 `binary_axnor_local5_shiftmax` |

仓库没有真实 M73 产物，也没有真实 M77 产物。验证器中的 M77 是 synthetic fixture；独立 M75 评审已明确它不是 real catalog/full-install positive path。

## 直接复用 H67 codebook/config 会怎样 fail-closed

| gate | Local5 / 旧产物 | loader 期望 | 结果 |
|---|---|---|---|
| revoked catalog SHA | M71 SHA `142e...` | 不得命中 revoke set | P0 fail-closed |
| catalog schema | `m71_h67_...` | `m77_h67_...` | P0 fail-closed |
| runtime checkpoint SHA | Local5 `1982...` | H67 `4f33...` | P0 fail-closed |
| M73 capture config | Local5 deploy `078b...` | M40 H67 deploy `8be3...` | P0 fail-closed |
| train trace schema | Local5 trace 缺失 | H67 M73 schema | P0 fail-closed |
| operator name/order | 与 H67 相同 | 四个固定名称 | 通过，仅说明 hook 可复用 |
| operator geometry | 与 H67 相同 | 768×768×3×3 | 通过，仅说明 k16/432 可复用 |
| operator weight identity | 四个 SHA 全不同 | H67 ep35 weight | P0 identity mismatch |
| train split file | 本机缺失 | SHA `919c...` | P0 fail-closed |

即使把旧 M71 JSON 文本改名或把 schema 字符串改成 M77，也仍会在 revoked SHA、external contract、train-trace、checkpoint 和 catalog derivation provenance 上失败；更不能绕过这些门。

## 建立 Local5 独立 PAFT 线的最小文件

### 只解锁 preflight / catalog capture

- 已有：ep44 checkpoint、训练配置、hardware-order 配置、valid825 list；
- 需从有完整 DSEC 数据的服务器取得：
  - `sequence_lists/train_split_seq.csv`，必须为 7,345 unique、18 sequences、SHA `919c79c6...fdc10`；
  - 对冻结的 train calibration keys，每个 key 的 `event_tensors/10bins/left/<sequence>/<key>`、`gt_tensors/<key>`、`mask_tensors/<key>`，逐文件 SHA；
- valid825 list 只用于证明 zero overlap，不得进入 catalog、超参选择或 hardware heldout selection。

### 解锁正式五轮训练

- 7,345 个 train keys 对应的完整 train tensors；
- Local5-specific PAFT/no-PAFT paired configs 与 launch manifest；
- Local5 train-only M77 catalog、外部 admission contract、独立重建 receipt；
- 与 calibration 分离的 train hardware-heldout sequence/cohort；
- valid825 仅在候选冻结后做准确率评估。

一个尚需在 L0 显式冻结的选择是 catalog capture config：训练配置为 hardware quant=false，现有 hardware-order 配置为 true。不能把两种 forward 当作同一分布；应选定与正式 PAFT forward 一致的一种，或先给出两者 bottleneck support 等价证明。

## 推荐里程碑

1. **L0 identity/split preflight**：补 train list，核对 7,345 unique、18 sequences、train-valid overlap=0；冻结 Local5 checkpoint、capture config、四组 operator weight SHA。
2. **L1 Local5 train-only capture**：fork M73 为 Local5 schema；至少 32 个 deterministic train samples 覆盖全部 18 sequences，产生 128 条四-Conv record 与逐数据文件 SHA。
3. **L2 Local5 M77 + admission**：只从 L1 做 deterministic weighted Hamming-Lloyd；独立复算 catalog SHA；external contract 固定 builder/seed/tie-break/trace/list/checkpoint/config/operator/geometry。
4. **L3 clean train-heldout hardware screen**：保留未参与 catalog 的完整 train sequence；同时报告 nominal、byte/port/matcher-aware cost 和 equal-activity guardrail；valid825 不参与该筛选。
5. **L4 full-install smoke**：Local5 loader 完成所有 fail-closed 校验，并跑四 hook forward/backward + one optimizer step。
6. **L5 paired 5 epochs**：ep44 同起点、同 seed/data order/epoch 的 PAFT 与 no-PAFT；valid825 只做准确率。
7. **L6 hardware promotion**：只有 clean heldout compute 与准确率同时过门，才进入 matcher/packer RTL、VCS、同资源 DC/STA/SAIF/PTPX 和 address-timed full-system 比较。

## 审计产物与边界

- `local5_paft_input_feasibility_audit.json`：机器可读的全部身份、检查、fail-closed matrix 和里程碑；
- `audit_local5_paft_inputs.py`：只读、无 torch/GPU 依赖的可重复审计器；
- `review_artifact_sha256.json`：本目录产物 SHA 封存。

本审计没有启动训练，没有修改任何现有源码、配置、`docs/359` 或数据文件，也没有把 valid825 internal catalog 当作训练证据。
