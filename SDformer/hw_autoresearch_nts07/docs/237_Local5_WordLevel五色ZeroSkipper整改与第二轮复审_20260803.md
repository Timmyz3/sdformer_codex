# Local5 Word-Level 五色 Zero Skipper 整改与第二轮复审

> 日期：2026-08-03  
> 前置文档：`docs/236_双线独立架构重启_归一化域商流与双向五色关系前沿_20260803.md`  
> 证据：只计本机 RTL、TB、Yosys 和日志。

## 1. 第一轮独立 DATE 复审

独立审稿人给出：

| 主线 | 分数 | 推荐 |
|---|---:|---|
| Motion | 2.4/5 | Weak Reject |
| Local5 | 2.7/5 | Borderline Reject |

对 Local5 的主要批评不是 profile，而是实现形态：旧 active index 为五个 bank 各自全深度扫描，默认 T450 等价于五个 90-bit 优先编码器；它无法证明理想 `1.522x` zero-scan 上界可实现。最高优先级是构建 word-level skipper，再连接关系转置与 TCFM-5。

Motion 的主要批评同样成立：当前 quotient 使用两个完整 score 单元，尚未连接 TARE 增量 score；`MAX_DESCRIPTORS=162` 也不是 fullres T450 配置。Motion 暂不提分。

## 2. 本轮整改

新增：`rtl_qfit/qfit_dual_color_word_skipper_index.sv`。

默认 T450 下，每个 color bank 深度为 90 bit，改为三个 32-bit word：

```text
五bank并行word-nonzero检查，共15个word
    -> round-robin bank选择
    -> bank内first-nonzero-word
    -> 一个32-bit next-set-bit编码器
    -> source ID/坐标解码
```

相对旧结构，新结构保留：

- 相同五色写入合同；
- 每 destination 每 bank 最多一个 set；
- 相同 round-robin bank 顺序；
- 相同 bank 内升序 source 顺序；
- 相同 duplicate、unique、conflict 计数。

集成前端 `qfit_dual_color_relation_frontier.sv` 已改用 word-level 索引。

## 3. T450 逐拍等价

新增 TB：`tb_qfit/tb_qfit_dual_color_index_equivalence.sv`。

配置与刺激：

- `HEIGHT=15, WIDTH=15, TIME_PLANES=2`；
- 450 个 destination 全部输入；
- 每个合法五点候选由随机 mask 决定 active；
- 输出随机反压；
- 旧索引与新索引逐周期比较 ready、active、done、valid、payload 和 last。

结果：

```text
PASS T450 full-depth/word-skipper equivalence unique=431 probes=1293
```

这证明本轮结构替换没有改变 source 集合、顺序或协议。`431` 只对应本次随机刺激，不是 workload 稀疏率。

## 4. 结构成本趋势

| 指标 | 旧全深度 | Word-level | 变化 |
|---|---:|---:|---:|
| Yosys generic cell | 2396 | 684 | -71.45% |
| `$mux` | 1121 | 290 | -74.13% |
| wire bit | 20699 | 11769 | -43.14% |

上述结果表明两级选择确实消除了大部分深优先编码逻辑，但仍有三条边界：

1. Yosys generic cell 不是标准单元面积；
2. bitmap 当前是 5 个 packed register bank，共 480 bit，不是已经映射的 SRAM；
3. 只有一个 source/cycle 输出，尚未实现五 source/cycle，也不需要用五路发射夸大主张。

## 5. 架构创新性更新

Local5 当前可辩护的架构链变为：

```text
双向五色拓扑不变量
    -> destination阶段五bank无冲突active set
    -> word-level bank-local zero skipper
    -> active-only关系转置
    -> source quotient term
    -> TCFM-5五destination无冲突累加
```

相对复旦 ISSCC 2023 butterfly zero skipper，本工作没有复制通用 butterfly 网络，而是利用 Local5 五点图的固定五色性质，将全局稀疏提取缩为 `5 x 3 word` 的局部两级选择。真正的新意应写成“拓扑着色同时约束稀疏发现与多播累加”，word skipper 是其物理实现，不应单独宣称发明 zero skipper。

## 6. 尚未解决的接收门槛

本轮解决了第一轮评审中的“深优先编码器不可落地”问题，但没有完成其全部最高优先级：

- 未连接 `qfit_source_multicast_term_builder`；
- 未连接 `qfit_tcfm5_projection_top`；
- 未用真实 15x15x2 post-G0 descriptor 回放；
- 未实现 SRAM latency、双缓冲和窗口重叠；
- 未与同资源 linear-5 做 OpenROAD/SAIF 对照。

下一阶段必须先完成真实 T450 的 `DCRF -> term builder -> TCFM-5` 端到端链，再讨论增加新的 Local5 创新点。

## 7. 可复现入口

```bash
sim_new_arch/run_dual_line_integrated_frontends_checks.sh
```

报告：`results/dual_line_integrated_frontends_rtl_20260803/word_skipper_comparison.md`。
