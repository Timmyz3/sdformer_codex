# 原始全量权重 vs bs4 续训权重推理对比

- 日期：`2026-04-25`
- 数据集：DSEC valid split
- 推理配置：[valid_DSEC_supervised_no_vis.yml](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/configs/valid_DSEC_supervised_no_vis.yml)
- 推理脚本：[eval_DSEC_flow_SNN.py](/root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow/eval_DSEC_flow_SNN.py)

## 对比对象

### 原始全量 baseline

- 来源 run id：`66d1fc5322004d59a03c8ab132b11830`
- 权重：[model.pth](/root/private_data/work/SDformer/experiments/mlruns/183153168054988814/models/m-d45133ebdb944b71863a88193bc77132/artifacts/data/model.pth)
- 日志：[infer_compare_original_full_20260425.log](/root/private_data/work/sdformer_codex/SDformer/experiments/logs/infer_compare_original_full_20260425.log)

### bs4 续训最终权重

- 来源 run id：`98d161a3f7144441a60fa79083e0fffd`
- 权重：[checkpoint_epoch59.pth](/root/private_data/work/sdformer_codex/SDformer/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth)
- 日志：[infer_compare_bs4_epoch59_20260425.log](/root/private_data/work/sdformer_codex/SDformer/experiments/logs/infer_compare_bs4_epoch59_20260425.log)

## 指标结果

| 权重 | AEE | AAE | PE1 | PE2 | PE3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始全量 baseline | 2.3922855505 | 12.0129445268 | 0.5333474422 | 0.2491989944 | 0.1581165405 |
| bs4 epoch59 | 1.3306697985 | 7.8131987781 | 0.4266112318 | 0.1526384893 | 0.0728003224 |

## 相对变化

- AEE 降低约 `44.38%`
- AAE 降低约 `34.96%`
- PE1 降低约 `20.01%`
- PE2 降低约 `38.75%`
- PE3 降低约 `53.96%`

## 结论

虽然 bs4 续训阶段的训练 loss 和 valid loss 看起来不如原始 full baseline，但同一套 DSEC valid 推理指标显示，`checkpoint_epoch59.pth` 明显优于原始全量权重。

后续应优先以 AEE/AAE 作为当前优化判断依据，而不是只看训练脚本里的 loss。
