# M159 H67 full-FFN scope 独立打铁审阅 r1

## 裁决

M159 的数学和热点归类可复现，但“完整 FFN 语义”尚不能 admission。综合评分 **76/100，P0=1、P1=2、P2=2**，裁决为 `REVISE_BEFORE_ADMISSION`。

可保留的核心数字是：12 个 block、十样本 120 个动态 group；`fc1+fc2=159,784,111`，FFN-local `sn1=9,120,000`、`sn2=36,480,000`，所以当前已计算子图为 `205,384,111 cycles/frame`，占 `620,302,905` compute envelope 的 **33.1102933%**。这是热点分类，不是已实现加速。完全免费删除的 1.4950× 仍只是 Amdahl 上限。

## 独立源码与推理语义

本审阅不导入、不调用 M159 analyzer，而是直接解析 exact-SHA 源码、配置、profile 和 trace。

`MS_Spiking_Mlp.forward` 的实际顺序是：

`sn1 -> dropout1 -> fc1 -> BN1 -> sn2 -> dropout2 -> fc2 -> BN2`

随后 block 执行 residual ADD。两个 dropout 在构造时都是 `p=0`，且 profile 在 `model.eval()` / `torch.no_grad()` 下运行。

M159 有一个结构错误：`drop_path_eval_off` 不是 FFN 分支的节点。源码只对 attention residual 调用 `drop_path`，MLP 返回后直接与 shortcut 相加。评估时 DropPath 确实关闭，因此不改变当前周期数，但完整 topology 列表必须删除它。

## P0：冻结 BN 语义写错

模型构造时的确通过 `spike_norm=BN` 创建默认带 running statistics 的 BatchNorm2d，但这不是冻结 trace 的实际推理语义。配置和 profile 都明确为：

- `bn_policy=no_running`；
- 78 个 BN 模块被设为 `track_running_stats=False`；
- `running_mean/running_var=None`；
- `eval_batch_size=1`。

因此 M159 结果中的 `BN_with_running_stats_by_source_default` 只描述了构造默认值，却把它当成了 resolved inference semantics。冻结评估实际使用当前 batch 统计，不能用 checkpoint running mean/variance 做静态 fold。这会影响 BN 周期、barrier、零保持和剪枝后数值，所以记为 P0，并建议暂时撤销 `complete_ffn_topology_scope=true`。

## Fresh trace 复算

独立正则仅接受：

`layers.<stage>.swin_blocks.<block>.mlp.{sn1.spiking_neuron,fc1,sn2.spiking_neuron,fc2}`

复算结果：

| 检查 | 结果 |
|---|---:|
| execution records / samples | 1,840 / 10 |
| unique blocks / dynamic groups | 12 / 120 |
| stage dynamic groups | 20 / 20 / 60 / 20 |
| sn1 / fc1 / sn2 / fc2 records | 120 / 120 / 120 / 120 |
| order/kind/shape/element mismatch | 0 |
| sn1 ATLIF issue cycles | 9,120,000 |
| sn2 ATLIF issue cycles | 36,480,000 |

每组四条记录在 `call_index` 上连续，stage shape 严格为 `C=96/192/384/768`、hidden=`4C`，`fc2` 返回原 `C` 宽度。

## 周期归类与无双计

24 个 FFN Linear 从 profile100 `operator_transactions.csv` 直接求和为 159,784,111，不依赖 M159 或 FFN review 中间结果。

ATLIF 需先按 `deployment_dead_result` 过滤：93 个 trace-visible ATLIF 模块中 81 个 live、12 个 dead。排除 dead 后，全局 96-lane issue 桶是 128,020,500；24 个 FFN-local sn1/sn2 全部为 live，其 45,600,000 是该桶严格子集，占 35.6193%。所以 `159,784,111 + 45,600,000 = 205,384,111` 只是把已经在全局 ATLIF 桶中的工作重新归到 FFN，没有把它再加到 620.303M 总量。

需保留口径：Linear 是 profile100 activity mean，ATLIF/topology 是 s10 身份与几何工作；这个分区可复现，但不是 address-timed executable schedule。

## BN/residual extent

| 服务 | elements/frame | 96-lane 单遍理想 vector issues |
|---|---:|---:|
| BN1 | 350,208,000 | 3,648,000 |
| BN2 | 87,552,000 | 912,000 |
| residual ADD | 87,552,000 | 912,000 |
| 合计 | 525,312,000 | 5,472,000 |

元素 extent 正确，但 5.472M 只能称为“每元素单遍、96 lane 满利用的 vector-issue 数”。`no_running` BN 还需当前 batch 的 moment reduction、系数生成、affine、读写和 barrier；residual 也需端口与依赖。因此它不是 BN/residual cycle 结果。

## paired prune 修正

扩展 hidden-channel mask 应覆盖：

1. `fc1` output rows；
2. BN1 hidden channels；
3. `sn2` hidden activation columns 和实现中与 lane 相关的临时状态；
4. `fc2` matching input columns。

M159 的“`sn2 temporal parameters/state channels`”过宽。PSN/ATLIF 的 temporal weight 是 `[T,T]`、bias 是 `[T,1]`，对 flatten 后的所有 channel/spatial columns 共享；删一个 hidden channel 不会删一行 temporal parameter。

`sn1` 在 `fc1` 之前且保持原 `C` 宽，不属于 hidden mask。BN2 和 residual 也保持原 `C` 宽，不随 hidden channel 删除，但必须重新验证数值，并在性能模型中作为固定剩余开销。

## 问题分级

### P0

1. 冻结推理 BN 是 `no_running/current-batch`，M159 却报告 running-stat semantics；这使 complete semantic admission 与 BN fold 暗示不安全。

### P1

1. `drop_path_eval_off` 被错放进 FFN topology，实际只属于 attention residual。
2. paired-prune 口径把共享 PSN temporal parameters 误说成 hidden-channel 成员。

### P2

1. 5.472M 需改名为单遍理想 vector issues，不能扩展为 BN/residual cycles。
2. profile100 Linear 与 s10 ATLIF/topology 的混合粒度需始终显式披露。

## admission 建议

保留数学热点结果和全部 `speedup/PPA/RTL/headline=false`。在重新封存之前，撤销 `complete_ffn_topology_scope=true`，删除 FFN DropPath，将 BN 改为 frozen `no_running` 语义，并把 sn2 剪枝合同改为“激活列/lane-state gating，共享 temporal parameters 保留”。

本审阅只写入 `results/m159_independent_hammer_review_r1_20260824/`，未修改 production、contracts、M159 original 或 `docs/359`。`docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 复核

```bash
python3 results/m159_independent_hammer_review_r1_20260824/validate_review.py
sha256sum -c results/m159_independent_hammer_review_r1_20260824/source_manifest.sha256
sha256sum -c results/m159_independent_hammer_review_r1_20260824/manifest.sha256
```

`independent_audit_m159.py` 默认拒绝覆盖现有 `independent_recompute.json`；若要 fresh 复跑，请用 `--output` 指定新文件。
