# Local5 Ordered 前端 RTL 校准与 V2 否决

> 用既有五 bank RTL 的 calibration/held-out 证据校准候选周期边界。

## 1. 结论

`docs/284...` 的 v2 修复了候选间 descriptor 计时不公平、ERM admission
不一致、多重比较和预注册锚点，但它保留了：

```text
recompute = control + max(450-cycle relation, ordered backend)
```

该重叠不符合仓库当前集成 RTL。旧 post-G0 五 bank 真实回放显示：

```text
RTL cycles ~= fixed relation/frontier phase
           + active_source descriptors
           + product terms
           + term stalls
```

因此 v2 在正式 joint-head profile 产生前再次作废。v3 必须用 held-out
校准过的串行边界作为唯一晋级主模型；理想 FCSR overlap 只能单列为敏感性
上界，除非以后用单顶层 RTL 证明。

## 2. 方法

脚本：

```text
scripts/calibrate_local5_ordered_frontend_rtl.py
```

输入分为两批：

| 用途 | 数据 | 路径 |
|---|---|---|
| calibration | 20260804 profile100 | Direct + QGASR，各100组 |
| held-out | 20260805 bb1e4 profile100 | Direct + QGASR，各100组 |

每组从 RTL 日志直接读取：

```text
cycles, active, terms, term_stall
residual = cycles - active - terms - term_stall
```

用 calibration 的 residual 中位数冻结串行固定项，再在 held-out 比较：

1. `sequential = fixed + active + terms + term_stall`；
2. `v2-max = 4 + max(450, 17 + active + terms + term_stall)`。

这不是在 held-out 上重新拟合；20260805 只用于验证 20260804 得到的固定项。

## 3. 校准结果

20260804 的 200 组 residual 为：

| mean | p50 | p95 | p99 | min | max |
|---:|---:|---:|---:|---:|---:|
| 459.025 | 459 | 465.05 | 471.02 | 456 | 475 |

冻结固定项为中位数 `459` 拍。

## 4. Held-out 结果

| 路径 | 模型 | MAE | p95绝对误差 | 均值有符号误差 | 聚合预测周期 |
|---|---|---:|---:|---:|---:|
| Direct | 串行校准 | 2.68 | 6 | 0.02 | 109232 |
| Direct | v2 max重叠 | 157.74 | 442.05 | -157.74 | 93456 |
| GASR | 串行校准 | 2.43 | 3 | 0.27 | 109757 |
| GASR | v2 max重叠 | 167.35 | 440 | -167.35 | 92995 |

held-out 实际周期为 Direct `109230`、GASR `109730`。对应 Direct/GASR 比：

| 口径 | 比值 |
|---|---:|
| RTL | 0.995443x |
| 串行校准 | 0.995217x |
| v2 max重叠 | 1.004957x |

v2 不只低估绝对周期，还把轻微退化翻成轻微收益。虽然本例没有跨过
`1.20x`，但它证明模型方向性可错，不能用于正式候选晋级。

## 5. V3 约束

v3 必须满足：

1. 主裁决使用 `fixed + active + terms + stall` 串行边界；
2. 固定项不得用正式 joint-head profile 回看后重新拟合；
3. 可以同时报告中位 `459` 与保守 `475` 两个预提交边界，但晋级取更差结果；
4. Direct 与 GASR2C-P 使用相同 fixed/frontier 项；
5. ERM replay 可消除 relation/frontier 固定项，但必须保留 memo read、builder
   capture、term、stall、controller 和 fallback；
6. GASR2C-P 仍是跨 head preserve 候选模型，过筛后才能实现 RTL；
7. final readout 与 scalar serializer 作为共同项保留，后续以 timing miter 校准。

## 6. 证据与产物

```text
results/local5_ordered_frontend_rtl_calibration_20260810/report.json
results/local5_ordered_frontend_rtl_calibration_20260810/report.md
tests/test_calibrate_local5_ordered_frontend_rtl.py
```

验证：`3/3 PASS`。

| 声明 | 证据 |
|---|---|
| 20260804/05 每组 RTL 周期 | `[rtl校准]` 已有真实 RTL 日志 |
| fixed=459 与 held-out 误差 | `[模型校准]` |
| v2 max 边界不适合作主裁决 | `[rtl校准]+[模型]` |
| 新 joint-head workload 收益 | `[待验证]` |
| GASR2C-P preserve | `[待验证]` |
| ASIC PPA | `[待验证]` |

## 7. 限制

1. 校准 trace 是旧 post-G0 100 组，不是正式同窗全 head profile；
2. 校准覆盖当前 Direct/QGASR 单 head 执行，不含跨 head preserve、最终
   readout 和 serializer；
3. 结果否定的是“没有 RTL 的 max overlap 用于晋级”，不是证明未来
   FCSR overlap 永远不可实现；
4. v3 仍需独立评审后才能启动正式 evaluator watcher。

## 8. 独立 DATE 评审

第三轮独立审稿确认本文对 v2 `max()` 相序的否决成立，但否决“仅用
`fixed=459/475` 即可启动 v3 watcher”。原因是校准公式直接读取 RTL
`term_stall`，而候选评估时仍要由失真的 Python 后端预测该值。

held-out 反例足以翻转主结论：

| 口径 | Direct/GASR |
|---|---:|
| 五 bank RTL | `0.995x` |
| 复用 v2 stall predictor 的串行模型 | 约 `1.43x` |

因此 `459/475` 只能说明当前前后端相序近似串行，不能单独构成候选周期模型。
`475` 也不是 ERM 的全局保守端点：增大 recompute 固定项会相对偏向 replay。

修正路径冻结为：

1. Direct/GASR 由完整五 bank RTL 逐 group 回放，不再使用 v2 stall predictor；
2. relation build、scan、controller、flush 和 readout 分项 timing；
3. plan writer 必须原生保留 prereg 绑定；
4. loose Git blob 只标 `[本地字节锚]`，未进入远端不可回写 ref 前不得称
   `[外部时间锚]`。
