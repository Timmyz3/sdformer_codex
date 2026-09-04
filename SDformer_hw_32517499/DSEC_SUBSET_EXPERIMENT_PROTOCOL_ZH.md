# DSEC 小子集实验协议

## 目标

这套协议用于快速判断“模块改动相对 baseline 是好还是坏”，而不是直接跑完整 DSEC。

它分成两层：

1. `single-seq smoke`
   - 单序列 `zurich_city_09_a`
   - 用来检查功能、loss、评估链是否正常
2. `fixed subset ablation`
   - 固定 `3-5` 条序列
   - 固定 train/valid split
   - 固定 epoch / seed / crop / lr
   - 用来判断改动趋势是否稳定优于 baseline

## 推荐子集

先固定一个不变的小子集，不要每次随意换序列。

推荐 3 条版：

- `zurich_city_01_a`
- `zurich_city_07_a`
- `thun_00_a`

推荐 5 条版：

- `zurich_city_01_a`
- `zurich_city_07_a`
- `zurich_city_09_a`
- `zurich_city_11_a`
- `thun_00_a`

## 推荐预算

### smoke

- 数据：单序列
- epoch：`1`
- 目的：代码回归测试

### subset

- 数据：固定 `3-5` 条序列
- 训练样本上限：每条序列 `200`
- 验证样本上限：每条序列 `40`
- epoch：`5`
- seed：`0`

## 评价标准

不要只看一个数。每次至少同时看：

- `train_loss`
- `valid_loss`
- `AEE`
- `AAE`
- 显存占用
- 每 step 或每 epoch 耗时

## 决策规则

1. smoke 明显更差：
   - 直接回滚，不进 subset
2. smoke 持平或略好：
   - 进入 subset
3. subset 稳定更好：
   - 再考虑 full DSEC
4. subset 结果不稳定：
   - 先不要上 full DSEC

## 生成子集 split

在完整 `saved_flow_data` 准备好之后运行：

```bash
cd /root/private_data/work/SDformer
python tools/make_dsec_subset_splits.py \
  --root /root/private_data/work/SDformer/data/Datasets/DSEC/saved_flow_data \
  --sequences zurich_city_01_a zurich_city_07_a zurich_city_09_a zurich_city_11_a thun_00_a \
  --train-limit-per-seq 200 \
  --valid-limit-per-seq 40 \
  --train-output train_subset_split_seq.csv \
  --valid-output valid_subset_split_seq.csv
```

## 训练子集 baseline

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
export KMP_DUPLICATE_LIB_OK=TRUE
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_subset.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow
```

## 评估子集 baseline

```bash
cd /root/private_data/work/SDformer/third_party/SDformerFlow
export KMP_DUPLICATE_LIB_OK=TRUE
python eval_DSEC_flow_SNN.py \
  --config configs/valid_DSEC_supervised_subset.yml \
  --path_mlflow file:///root/private_data/sdformer_mlflow \
  --runid <YOUR_RUN_ID>
```

## 你后面每次改模块时的固定顺序

1. 跑单序列 smoke
2. 跑固定 subset
3. 记录和 baseline 的差值
4. 只有 subset 稳定变好，才跑 full DSEC
