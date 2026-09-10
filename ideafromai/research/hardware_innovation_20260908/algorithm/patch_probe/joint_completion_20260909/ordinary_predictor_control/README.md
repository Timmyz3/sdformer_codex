# 固定学生的普通相关预测控制

`probe.py` 已运行；本地 CPU 约 1.43 秒，未用 GPU、未改网络或训练学生。只用真实 FP32 norm1 的 train16 拟合，valid4 各 983,040 门只评价。两学生分别是自选前缀 `[4,6,7]` 的 row34_packed_word、前缀 `[2,3,7]` 的 common3_group。

逐 H 用三个已知 Y 列回归同学生完整十维 margin；中心化 ridge 的唯一正则为 `1e-3*trace(Cpp)/3`。预测残差在 train16 上校准，乘原学生 gamma；A/b/θ/prefix/gamma 全不变。common3 三个已由前缀完整确定的行直接沿用精确式。参数拟合用 Float64，推理表及预测用 FP32，未来 Y 只作训练标签及未决门的完整回退。

原预测器对既有 valid4 的 gate 与 need_column **均 0 mismatch**。以下错误均相对各学生自身完整函数，不是网络 AEE。

| valid4 | row34 原／ridge | common3 原／ridge |
|---|---:|---:|
| 错门 | 213／141 | 275／112 |
| 漏自身非零门 FN | 211／137 | 272／70 |
| 多发 FP | 2／4 | 3／42 |
| 源加权列访问 | 2,843,166／2,865,915 | 2,113,250／2,099,147 |
| prefix＋tail W 向量请求 | 1,778,102／1,790,308 | 1,504,664／1,501,846 |
| 预测非零 FMA／(p,h) | 13／30 | 27／29 |
| 新增 FP32 回归系数 | 0／11,520 B | 0／8,064 B |

每轴完整一次 T10 共享扫描的请求分母为 1,210,788。原 `source_words` 是 P4 OR，计数是该物理组包络上的真实源交集和时间并集；没有逐 W 零过滤、逐 lane 取消、端口或周期含义。ridge 的 W 请求变化分别为 **+0.686%／−0.187%**，不能用更高的接受率替代这些数。

回归 margin 不是原 A 的真实前缀部分和。失败时必须重算 A 行，或另付 `(A_prefix−beta)Y_prefix` 修正。本脚本同时给出保存 Y 后重算失败行，以及保留预测值时的有利算术界。两轴“预测＋完整行回退”FMA 分别 **+37.01%／+3.30%**；相对原预测器可直接续加真实尾部的有利界，则分别 **+75.54%／+11.15%**。这些仍是标量 FMA，原共享 A 的 CMVM/CSE 及新逐 H 系数的实际执行均未计成服务；保留中间值的容量也不免费。

`*_ridge_table.npz` 保存 `coefficient[H,3,T]`、intercept/radius、训练 residual std、固定 lambda、原 A/b/θ/gamma/prefix；`*_valid_decisions.npz` 保存实际 need/gate/accepted，便于后续同资源服务检查。全部详细分母在 `result.json`。

结论限定于本轮：普通相关预测确实减少错门，但在固定 gamma 下没有带来足以抵消新增预测算术的取权优势。保留该普通控制，不把此结果解释为整个相关预测方法失败，也不在本轮继续调 gamma、扫正则或外推 AEE。

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 ../.venv_train312/bin/python probe.py
```
