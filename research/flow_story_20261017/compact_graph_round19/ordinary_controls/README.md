# 新图普通评分 IO

`run_native_scores.py` 只适配来源、执行准入和输出标签；数学直接用原 R18 `g0_core.py`。源码/3组CPU检查已写 `SOURCE_FREEZE_V1.json`；尚未读取实际 R19 数组或运行分数。必须先取得独立 source review JSON：schema `flow_r19_native_score_source_review_v1`、status `SOURCE_REVIEW_PASS`、`source_sha256` 精确三个新源及 `math_sha256` 原 R18 核。

完整 R19 执行审计通过后，由根代理传入实际结果/audit SHA：

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B run_native_scores.py \
  --result /absolute/path/to/R19/results.json --result-sha256 ACTUAL_R19_SHA \
  --audit /absolute/path/to/R19/ACTUAL_EXECUTION_AUDIT_V1.json --audit-sha256 ACTUAL_AUDIT_SHA \
  --freeze SOURCE_FREEZE_V1.json --freeze-sha256 ACTUAL_SCORE_FREEZE_SHA \
  --review /absolute/path/to/SCORE_SOURCE_REVIEW_V1.json --review-sha256 ACTUAL_REVIEW_SHA \
  --output /absolute/fresh/R19/scores_v1 --wall-seconds 300
```

只读取八个新 native 的 u/sn1；预计672score/56choice/160policy-frame，0模型/GT/质量。输出 `scores.json` 使用独立 schema `flow_r19_g0_cached_ordinary_v1`、graph_id 和 R19 result/audit SHA，不读取 R18/R17 的实际评分或质量。之后由独立数学实现核验，并接新图自身端点质量。
