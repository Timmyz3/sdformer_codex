# H12_atlif_gradfix — ATLIF 阈值梯度闭环修复实验

创建日期：2026-09-17。设计文档见 `../H12_ATLIF_GRADFIX_TERNARY_DESIGN_20260917.md`。

## 背景

H9 overlay 的 `atlif_ternary_psn.py` 四个 Surrogate 变体 backward 全部有缺陷：

1. `grad_input = grad_input * tmp` 先覆盖上游梯度 g，随后复用被覆盖值 → tmp 被乘两次（tmp²）；
2. Ternary/Binary/SymmetricBinary 用了 `.abs()` → grad_thre 恒 ≤ 0；
3. 全部缺失恒等项 d(out)/d(thre) = ternary/active/out（STE 意义下）。

后果：Adam 通道（`h28_optimizer.py:79-80` 的 atlif_threshold 参数组）是阈值唯一闭环通道，
grad_thre 恒负使 Adam 单调推高阈值 → 发放率单调下降，且部署侧（threshold_eta=0，纯推理）
无法自纠。这就是"阈值都约等于 1"的代码级根因（叠加配置里 threshold_init=1.0、
threshold_eta=0、activity_eta=0 三项全关）。

## 本目录改动

- overlay 整体复制自 H9（遵守 `hw_autoresearch_nts07/autoresearch.md:42-44` 禁改 overlay 约束）。
- `overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py`：
  四个变体 forward 增存离散码（ternary/active/out），backward 改为
  `grad_thre = (g*code).mean() - (g*tmp).mean()`，返回 `g*tmp`（单次 tmp，无 abs）。
  OfficialATLIFSurrogate 同时恢复官方单 tmp 语义。
- `entrypoints/train.py`：复制自 H9，尚未改动。
- `tests/test_gradfix.py`：四项回归判据，全部通过（2026-09-17，env312/torch）：
  1. 无发放 + 希望更多输出 → 修复版 grad_thre > 0（abs 三变体缺陷版恒负，死锁）；
  2. 饱和发放 + 希望更少输出 → 修复版 grad_thre > 0（缺陷版在 tmp=0 处恒 0，无负反馈）；
  3. 恒等项手工微分对照 grad_thre = 1/30 精确一致；
  4. 闭环动力学：OfficialATLIF 修复版 thre 3.00→1.86、firing 0→5.3%；
     缺陷版 Binary 对照 thre 卡死 2.99、firing 恒 0。

## 运行测试

```bash
/root/private_data/work/hardware_innovation_20260908/env312/bin/python \
    tests/test_gradfix.py
```

## 下一步（实验矩阵 G1，未启动）

1. E2 anchor fail-fast 修复搬进本目录入口（设计文档 §3.4）；
2. 生成 G1 配置：修正 backward + threshold_eta=0.1 + target_rate_mode=bidirectional，
   threshold_lr_scale ∈ {5e3, 1.5e4, 5e4} 扫参；
3. 从 H67 ep35 出发 2–3 epoch probe，门槛见设计文档 §4：
   AEE ≤ 1.345、SOPs 降 ≥ 10%、thresh 回落至 [0.5,1.5]、neg_rate ≤ 0.30。

## 红线

- 不改 H9 overlay 与 H67 ep35 已发布数字；capture 记账（atlif_impl path/sha256）只认本目录新 receipt。
- 修复改变了历史 checkpoint 的复现语义，任何对照必须显式声明用的是缺陷版还是修复版 backward。
