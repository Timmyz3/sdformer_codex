# Local5 跨 H、跨序列 Phase Anchor 冻结与 DATE 复审

> 日期：2026-08-12  
> 范围：Local5 formal phase telemetry 的真实 workload 回放集合  
> 证据等级：`[prof]+[计划]`  
> formal G0：**DENY**

## 1. 本轮结论

从 joint-head profile100 的 100 个真实样本、1,200 个 canonical block-window 中，
冻结了 30 个后续 phase telemetry anchor：

| 指标 | 结果 |
|---|---:|
| 真实样本 | 100 |
| 输入 canonical block-window | 1,200 |
| 选中 anchor | 30 |
| sequence cluster 边际覆盖 | 18/18 |
| head 配置 | H3/H6/H12/H24 |
| H×cluster 非零格 | 27/72 |
| 固定反压 seed | 20260813、20260814 |

正式包：

```text
results/local5_phase_anchor_selection_v4_final_20260812/
```

该包回答“有限完整 trace anchor 应回放哪些真实 workload”，不证明这些窗口已完成
RTL phase 回放，也不证明周期收益、PPA、formal G0 或架构创新。

## 2. 选择规则

1. 每个 sequence cluster 选择 term-items/head 最接近本 cluster 中位数的窗口；
2. 每个 H 选择 sampled canonical windows 内 term-items/head 的最小和最大点；
3. 每个 H 选择 sampled canonical windows 内 service-cycles/head 的最小和最大点；
4. 每个 H 从 `active_source_ratio>0 && service_cycles>0` 的候选中选择非零中位
   term 密度点，用于固定 seed 随机反压；
5. 同一 identity 命中多条规则时合并 reason，不重复回放。

“sampled extrema”只指上游均匀选出的 1,200 个 canonical window 内的样本极值，
不能写成全分辨率全部空间窗口总体极值。计划只保证 H 和 sequence cluster 的边际
覆盖，不声称完成 4×18 笛卡尔覆盖。

## 3. 首轮负结果

初版脚本直接在 13,800-group 循环内反复访问压缩 NPZ 列，导致同一数组被重复解压；
运行约 2.5 分钟仍未完成，被主动终止。修复为每列只解压一次后，真实生成耗时约
4.5 秒。

初版 v1 选出 29 个 anchor，但独立 DATE 风格审阅仅给出：

```text
3/5，Conditional GO
```

审阅发现三个 P1：

1. H6 的“反压 anchor”是全零 workload，无法激励 backpressure；
2. 报告只给出 H 与 cluster 边际覆盖，未披露 H×cluster 交叉分布；
3. NPZ offset 未验证终点等于实际数组长度，NumPy 越界 slice 可静默截断。

## 4. 评审修复

### 4.1 非零反压锚点

四个 H 的反压候选均强制满足 active 和 service 非零；H6 新锚点为：

```text
sample89/stage1/block1/window87
term_items/head = 149.17
service_cycles/head = 152.17
active_source_ratio = 0.0459
```

空窗口仍保留为 idle/empty corner，但不再兼任反压锚点。

### 4.2 覆盖边界

v4 输出完整 H×cluster 机器可读矩阵。实际 72 个格中 27 个非零；这足以证明
18/18 cluster 与四种 H 的边际覆盖，但不是交叉全覆盖。该边界已同时写入 JSON 和
中文报告。

### 4.3 fail-closed 数组合同

生成器对六个输入数组检查：

- 精确 dtype；
- 一维形状；
- offset 起点、单调性和终点；
- item/descriptor 终点与实际数组长度一致；
- 每个 window 的完整 head 集合和 `heads*tokens` descriptor 数。

负例覆盖 dtype 漂移、非一维数组、item terminal 错误、descriptor 静默截断、缺
cluster、sample 不连续及无非零反压候选。

### 4.4 可复现封包

v4 包显式记录 `package_revision=v4_reviewfix`，绑定 plan、中文报告、生成器、单测和
测试执行 receipt 的 SHA。封包前在 Python 3.12.3、NumPy 1.26.4 下执行 10 项测试，
结果 `10/10 PASS`。

## 5. 二次独立复审

修复后的 v3 独立复审结果为：

```text
4/5，GO
P0：无
P1：无
```

GO 仅允许将该集合用于后续 Local5 anchor replay。剩余 P2 中 package revision、
测试执行 receipt 和主要负例已在 v4 关闭；Git 提交级 provenance 和大型 NPZ 的外部
独立重哈希仍未完成。v4 最终复核仍为 `4/5 GO`、P0/P1 为空；新增 P2 是测试 receipt
尚未单列被测 selector SHA、argv 仍指向包外绝对路径，以及 offset 起点/非单调、三种
descriptor 数组分别漂移和矩阵/seed/receipt 闭环负例仍可继续补。上述 P2 不阻塞
anchor replay，避免继续把工作量耗在计划层包装上。

## 6. 对 Local5 完整度的作用

这一步补齐了 Local5 相对 Motion 缺少的“真实 workload 验证集合定义”，但没有让
Local5 自动追平 Motion 的性能证据。下一阶段仍必须完成：

1. 四种 H 中位锚点的完整 trace 与 compact telemetry 逐事件同构；
2. 30 个 anchor 的 compact RTL 回放，四个非零反压锚点使用两组固定 seed；
3. 全 1,200 window 的 Direct compact telemetry；
4. 100/100 Acc32、462,600 phase 和 admission receipt 的只读合并；
5. Local5 强基线下的周期、mean/p95/p99、SRAM 流量及后续同约束 PPA。

## 7. DATE 表述边界

允许写：

> 我们依据 100-sample ordered workload 冻结跨四种 H、18 个序列 cluster 和密度极值
> 的 phase anchor replay 计划。

禁止写：

- 30 个 anchor 已通过 RTL；
- 27/72 等同 H×cluster 全覆盖；
- sampled extrema 是全 workload 极值；
- anchor 选择、telemetry 或验证框架本身是 DATE 架构创新。
