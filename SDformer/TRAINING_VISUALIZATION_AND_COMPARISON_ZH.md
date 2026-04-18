# 训练记录与可视化说明

## 1. 现在已经具备什么

当前训练已经通过 MLflow 自动记录：

- 配置参数
- `train_loss`
- `valid_loss`
- 模型参数量

另外，从现在开始，新的 DSEC 训练还会额外记录：

- `lr`
- `epoch_time_sec`
- `train_step_time_sec`
- `train_samples_per_sec`
- `valid_time_sec`
- `valid_step_time_sec`
- `max_gpu_mem_gib`

对应训练脚本：

- [train_flow_parallel_supervised_SNN.py](/root/private_data/work/SDformer/third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py)

注意：

- 这些新增性能指标只会影响之后启动的新训练
- 当前已经在跑的 full baseline run 不会自动补录这些新字段，除非重启训练

## 2. 交互式查看

可以直接启 MLflow UI：

```bash
mlflow ui --backend-store-uri file:///root/private_data/sdformer_mlflow --port 5001
```

然后在浏览器里看：

- 每个 run 的参数
- `train_loss` / `valid_loss` 曲线
- 多个 run 的横向比较

## 3. 本地导出对比表和曲线

我补了一个导出脚本：

- [export_mlflow_compare.py](/root/private_data/work/SDformer/tools/export_mlflow_compare.py)

示例：

```bash
cd /root/private_data/work/SDformer
python tools/export_mlflow_compare.py \
  --runids 89cc9fdf5c93495fa2ab5f0071edc41d 4d16af49a24a49a880c91af30b495d28
```

默认会输出到：

- `experiments/results/mlflow_compare/run_summary.csv`
- `experiments/results/mlflow_compare/metric_history.csv`
- `experiments/results/mlflow_compare/*.png`
- `experiments/results/mlflow_compare/README.md`

## 4. 适合拿来比较什么

这套导出最适合比较：

- baseline vs 改模块
- baseline vs 改注意力
- baseline vs 剪枝版
- smoke / subset / full 不同阶段 run

## 5. 推荐对比的字段

至少同时比较：

- `train_loss`
- `valid_loss`
- `lr`
- `epoch_time_sec`
- `train_step_time_sec`
- `valid_time_sec`
- `max_gpu_mem_gib`
- 最终 `AEE / AAE`

## 6. 一个实际建议

以后每做一次结构改动，至少保留：

1. smoke run id
2. dev-subset run id
3. full run id

然后用同一个导出脚本统一出图和出表，这样后面对论文图表和实验记录最省事。
