# DSEC 小量实验模板

## 1. 这份模板解决什么问题

这份模板用于在不跑 full DSEC 的前提下，快速判断改模块、改注意力、做剪枝后是否值得继续。

它直接复用仓库里现成的三样东西：

- [make_dsec_subset_splits.py](/root/private_data/work/SDformer/tools/make_dsec_subset_splits.py)
- [train_DSEC_supervised_SDformerFlow_en4_subset.yml](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_subset.yml)
- [valid_DSEC_supervised_subset.yml](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/valid_DSEC_supervised_subset.yml)

另外我补了一条统一入口脚本：

- [run_dsec_dev_subset.sh](/root/private_data/work/SDformer/scripts/run_dsec_dev_subset.sh)

## 2. 推荐的两个固定子集

### `dev3`

适合最快速筛方向：

- `zurich_city_01_a`
- `zurich_city_07_a`
- `thun_00_a`

### `dev5`

适合正式的小量对比：

- `zurich_city_01_a`
- `zurich_city_07_a`
- `zurich_city_09_a`
- `zurich_city_11_a`
- `thun_00_a`

默认建议：

- 日常开发先用 `dev3`
- 准备决定是否进入 full train 时用 `dev5`

## 3. 默认预算

- train limit per seq: `200`
- valid limit per seq: `40`
- epoch: `5`
- seed: `0`

这套预算的目标不是复现论文结果，而是做稳定的相对比较。

## 4. 标准执行顺序

### 4.1 生成固定 subset split

```bash
cd /root/private_data/work/SDformer
bash scripts/run_dsec_dev_subset.sh make-splits dev5
```

这会生成：

- [train_subset_split_seq.csv](/root/private_data/work/SDformer/data/Datasets/DSEC/saved_flow_data/sequence_lists/train_subset_split_seq.csv)
- [valid_subset_split_seq.csv](/root/private_data/work/SDformer/data/Datasets/DSEC/saved_flow_data/sequence_lists/valid_subset_split_seq.csv)

### 4.2 跑 subset 训练

```bash
cd /root/private_data/work/SDformer
bash scripts/run_dsec_dev_subset.sh train
```

训练实际调用的是：

- [train_flow_parallel_supervised_SNN.py](/root/private_data/work/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)
- 配置：[train_DSEC_supervised_SDformerFlow_en4_subset.yml](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_subset.yml)

### 4.3 跑 subset 评估

```bash
cd /root/private_data/work/SDformer
bash scripts/run_dsec_dev_subset.sh eval <RUN_ID>
```

评估实际调用的是：

- [eval_DSEC_flow_SNN.py](/root/private_data/work/SDformer/third_party/SDformerFlow/eval_DSEC_flow_SNN.py)
- 配置：[valid_DSEC_supervised_subset.yml](/root/private_data/work/SDformer/third_party/SDformerFlow/configs/valid_DSEC_supervised_subset.yml)

### 4.4 一步跑通 split + train

```bash
cd /root/private_data/work/SDformer
bash scripts/run_dsec_dev_subset.sh full-cycle dev5
```

这一步不包含 eval，因为 eval 需要训练产出的 `runid`。

## 5. 你每次改动后的固定流程

1. 先跑单序列 smoke
2. 再跑 `dev3`
3. 如果 `dev3` 趋势不错，再跑 `dev5`
4. 记录结果
5. 只有 `dev5` 稳定优于 baseline，才进入 full DSEC

## 6. 推荐的记录模板

实验记录表已经放在这里：

- [dsec_dev_subset_experiment_template.csv](/root/private_data/work/SDformer/experiments/results/tables/dsec_dev_subset_experiment_template.csv)

建议每次复制一行，至少补齐这些字段：

- `experiment_id`
- `git_commit`
- `baseline_name`
- `change_type`
- `change_summary`
- `subset_preset`
- `train_budget`
- `train_loss`
- `valid_loss`
- `AEE`
- `AAE`
- `step_time_sec`
- `gpu_mem_gib`
- `params_m`
- `flops_g`
- `sparsity`
- `quant_status`
- `hw_risk`
- `decision`

## 7. 决策规则

### 可以直接淘汰

如果出现以下任一情况，通常不用进 full train：

- smoke 已经不稳定
- `dev3` 明显更差
- `dev5` 指标没有改善但显存和延迟更差
- 剪枝后恢复能力很差
- 硬件量化/稀疏模式不可落地

### 可以继续推进

如果出现以下任一组合，可以继续：

- `AEE/AAE` 更好，成本相近
- 精度持平，但 step time / 显存更优
- 剪枝后精度轻微下降，但参数/FLOPs/latency 明显更优
- 硬件友好性明显提升，精度损失在可接受范围内

## 8. 这套模板最适合什么场景

- 替换一个 attention 模块
- 增加一个轻量辅助模块
- 做 channel/head/block 剪枝
- 比较 2 到 3 个候选结构

## 9. 不建议怎么用

- 不要每次改动都重建不同 subset
- 不要在不同实验里随意改 train/valid 样本上限
- 不要只看单次波动，要和固定 baseline 比趋势
- 不要把 subset 结果当成论文最终结论
