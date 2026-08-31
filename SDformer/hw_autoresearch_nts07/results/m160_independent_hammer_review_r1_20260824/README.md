# M160 独立打铁评审 r1

结论：`54/100，P0 拒绝接纳 M160 r1 的生产语义`。M160 r1 的拓扑身份、running-stat 反事实代数、437.76M element extent 和 9.86595x value-count 数学可以复现；但冻结 H67 profile 明确选择 `test.bn_policy=no_running`，生产入口在 `build_model` 后会改写全部 78 个 BN。M160 r1 没有执行这一步，审计的是 constructor-default running-stat BN，不是生成冻结 trace 的 BN 语义。

## 独立复核结果

- 未导入 M160 analyzer；只使用 exact-SHA production loader、配置、checkpoint、源码和冻结账本。
- 12 个 `MS_Spiking_Mlp` 身份完整，共 17,664 expanded channels、4,416 output channels；两个 Linear 均无 bias，两个 dropout 均为 0。
- 逐块 ATLIF 均为 `T=10`、`binary/official_atlif/center=zero`；阈值和 10x10 temporal weight 已记录在 JSON。
- 在 M160 r1 的 running-stat 语义下，独立 24 组 Linear->BN miter 最大绝对误差为 `9.12218e-6`，BN alpha/offset 汇总与 producer 精确复现。
- 在冻结 profile 的 `no_running` 语义下，24 组动态 BN 手工公式 miter 最大绝对误差为 `5.72824e-7`；动态 BN 与 r1 静态 fold 的 24 组测试中，最小最大误差仍达 `1.92265`，不是容差问题。
- 全零路径发生语义翻转：running-stat 模式下 `zero-fc1 -> BN1 -> sn2` 有 927 个 active value；生产 `no_running` 模式下为 0。两种模式最终 BN2 后均为 `44,160/44,160` 非零，但常数来源与中间触发完全不同。

## 分级问题

### P0：遗漏生产 `no_running` BN policy

冻结 YAML 为 `bn_policy: no_running`。生产 profile main 在 `build_model` 后调用 `configure_batch_norm_evaluation`，将 BN 的 `track_running_stats=False` 且 running mean/variance 置空。M160 r1 只调用 `build_model`/`model.eval()`，随后还要求 `track_running_stats=True`。

因此以下 r1 项目都不能接纳为冻结 trace 的证明：checkpoint-only alpha/offset、24 个静态 fold、927 trigger、静态 BN1-to-sn2 separable bias，以及静态 BN2 residual-commit fold。修复方向只有两个：

1. 保持 `no_running`，实现并建模 batch/time/spatial reduction、均值/方差、归一化及其端口和周期；或
2. 将评测协议明确改为 running stats，并重新做算法精度、trace 身份和 checkpoint/evaluator 合同，不能只改硬件分析器。

### P1：算法 mask 的 ATLIF 单元定义错误

sn2 的 temporal weight 是每块共享的 `[10,10]`，bias 是共享的 `[10,1]`，不是每 expanded channel 一套参数。可执行原子 mask 应覆盖：fc1 output row、BN1 对应 channel、sn2 expanded activation lane across T，以及 fc2 matching input column。共享 temporal 参数必须保留，不能按 channel mask。

### P1：stage 优先级混用了不同目标

Stage 2 的确占 M159 accounted FFN compute 的 `50.7078%`，适合作为 compute/PAFT 优先级；但本次 M160 的 BN element extent 最大的是 stage 0：

| Stage | BN1+BN2 elements/frame |
|---:|---:|
| 0 | 184,320,000 |
| 1 | 92,160,000 |
| 2 | 138,240,000 |
| 3 | 23,040,000 |

若目标是消除 standalone BN traversal，应先看 stage 0；最终仍须由 port-aware cycle model 决定。

### P2：边界和性能口径

- 源码中 MLP 的 BN2 后直接通过 parent `sew_function(..., ADD)` 做 residual add；`drop_path` 只作用于 attention 分支。M160 的 FFN source-order 列表中 `drop_path_eval_off` 位置不精确，虽不改变 eval 数值，但应修正。
- 独立重算 `350,208,000 + 87,552,000 = 437,760,000`，它只是 BN element-visit extent。
- 独立重算 `176,640 / 17,904 = 9.8659517426x`，它只是常数 value-count 比。
- 两者都不是 SRAM/DRAM traffic、cycle speedup、energy reduction 或 system speedup；M160 当前对此保持 false 是正确的。

## 接纳口径

可接纳：exact-SHA 身份、12 块拓扑主体、running-stat 反事实代数、element extent 数学、value-count 数学，以及“不能直接跳过全零 FFN 分支”的保守结论。

禁止接纳：M160 r1 的 production/checkpoint-bound numeric、冻结 profile 静态 BN fold、冻结 profile zero-path 常数、BN-elision traffic、cycle/system speedup、PPA 和 headline。

复跑：

```bash
cd /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07
/opt/anaconda3/envs/pytorch310/bin/python \
  results/m160_independent_hammer_review_r1_20260824/audit_m160_independent.py
```

机器可读结论见 `m160_independent_hammer_review.json`。
