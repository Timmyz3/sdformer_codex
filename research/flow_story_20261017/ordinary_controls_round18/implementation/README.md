# R18 G0 CPU 缓存评分入口

冻结文件：`SOURCE_FREEZE_V1.json`。合成检查：`CPU_TESTS_V1.json`，6 项 PASS。实际评分必须先取得独立 source review JSON；本说明不是已经执行的声明。

```bash
/opt/anaconda3/envs/pytorch310_cpu/bin/python -B run_g0.py \
  --freeze SOURCE_FREEZE_V1.json \
  --freeze-sha256 a6a410bbc98f95af8c42636cf392695c53d3c3fa0e8a5931e9e5c931136afe43 \
  --review /absolute/path/to/independent/source_review.json \
  --review-sha256 ACTUAL_REVIEW_SHA256 \
  --output ../actual_g0_v1 \
  --wall-seconds 300
```

从此 implementation 目录执行。脚本拒绝已存在输出目录。读取固定 R17 原执行 JSON、独立执行审计与八个 native NPZ；只载 `u/sn1` 数组。不会读取 GT、质量 JSON 或 flow 数组，不加载模型、checkpoint，也没有 GPU/训练入口。

结果 `scores.json`：`frames[8]`，每帧 `scores[12]` 含七有限分值、`policies[20]` 含七规则/十二固定/均匀随机期望；确定策略提供坐标、4×5 mask、one-hot 权重和真实 singleton arm。随机期望提供十二个 1/12 权重和有理数表示，不构造平均预测。输入 archive/payload SHA、逐 tile 算术/读取账、12 倍计数、11 次/规则的选择比较均保留。该文件不包含 AEE 或 GT 标签。

独立质量分析须先验证完整冻结与实际输入，再把策略映射到 R17 已保存真实端点；不得改变方向、逐帧挑方法或把随机期望当一张平均 flow 的 EPE。缓存原生 PSN 与旧逐 tile FP32 PSN 不宣称位级同义。所有结果仅为已暴露 H8 的无学习普通控制，不是强同监督 generic，也没有实际硬件周期收益。
