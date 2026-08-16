# Local5 全 4800 Group 精确五 Bank RTL 回放

> 本轮只关闭一个最高优先级缺口：用完整五 bank RTL 代替失真的 Python
> stall predictor，裁决 Direct 与 GASR-reset。结果是可复现的负结果，不是
> ASIC PPA。

## 1. 结论

在同一批 `4800` 个真实 Local5 ordered group 上：

| 指标 | Direct | GASR-reset | 结论 |
|---|---:|---:|---|
| 总 RTL 周期 | 5,639,279 | 5,684,370 | GASR 为 `0.9921x`，慢 `0.80%` |
| term stall | 533,689 | 580,809 | GASR 多 `8.83%` |
| 1RW SRAM read | 6,775,068 | 1,058,208 | GASR 少 `84.38%` |
| 1RW SRAM write | 7,458,128 | 1,741,268 | GASR 少 `76.65%` |
| 1RW SRAM 总事务 | 14,233,196 | 2,799,476 | GASR 少 `80.33%` |

本地结果产生前已写入的门槛要求：聚合加速至少 `1.20x`，并且总体和每个
stage 的 p95 均不回退。GASR-reset 两项均未通过，裁决为：

```text
REJECT_GASR_RESET_PATH
```

该门槛只有本地“先写规则、后跑结果”的字节证据，没有外部不可变时间戳，
不得写成正式预注册。

## 2. 回放边界

RTL 数据流为：

```text
真实 relation/gate/K 向量
  -> relation frontier + 五色 active bitmap
  -> FIFO2
  -> source-major term builder
  -> Direct 1RW 或 GASR-reset 五 bank backend
  -> flush/done                         <- 周期计数终点
  -> Acc32 readback 与 expected_acc 逐项比较  <- 只用于miter，不计周期
```

输入来自：

```text
results/local5_fullres_bb1e4_postg0_profile100_20260805
```

其 post-G0 qualification 覆盖：

- `100` 个 sample；
- `12` 个 attention block；
- 在 100-sample cohort 上覆盖全部 stage 的所有 head；这是 union coverage，
  不是每个 sample/block 穷举全部 head；
- 每个 block/sample 选 `4` 个 coprime-rotating flat window-head group；
- 合计 `4800` 个 group、`443,948` 个 active source、`2,456,327` 个 term、
  `7,458,128` 次 destination update。

这是 sampled ordered group 范围，不是 full-workload totals，也不是 full encoder
端到端周期。

## 3. 数值与时序合同

两条日志被汇总脚本逐 group 强制检查：

| 路径 | `new1rw` | `mode` | relation latency |
|---|---:|---:|---:|
| Direct | 1 | 0 | 1 |
| GASR-reset | 1 | 1 | 1 |

两个仿真均出现唯一末尾 `PASS`，覆盖全部 `4800` 组。testbench 在每组 flush
后读取 Acc32，并与生成器给出的 `expected_acc` 逐项比较；任一地址不等都会
在 `PASS` 前触发 fatal。每个模式覆盖 `4,320,000` 个 Acc32 坐标，两模式共
执行 `8,640,000` 次比较，mismatch 为 `0`，证据为 `[rtl]`。周期口径只到
flush/done，不包含随后 readback。

descriptor、gate、K 和 relation 来自真实 trace；本轮权重使用 synthetic
int8 合同。权重数值不参与 ready/valid、bank 冲突或 flush 控制，因此周期
裁决仍是实际 RTL 周期；数值等价声明只覆盖该 synthetic weight miter。真实
theta-folded checkpoint 权重的数值回归由既有较小集合承担，本轮不扩张其
证据范围。

## 4. 分 Stage 结果

| Stage | group | Direct 周期 | GASR 周期 | 加速 | Direct p95 | GASR p95 | p95不回退 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 800 | 733,319 | 744,965 | 0.9844x | 2939.2 | 2827.1 | 是 |
| 1 | 800 | 431,445 | 446,330 | 0.9667x | 942.8 | 1137.8 | 否 |
| 2 | 2400 | 2,935,215 | 2,972,785 | 0.9874x | 4408.7 | 3952.4 | 是 |
| 3 | 800 | 1,539,300 | 1,520,290 | 1.0125x | 5495.3 | 4986.7 | 是 |

表中 p95 是 4800 条 sampled group 的等权分布，不是 encoder p95。逐 group
的 win/equal/loss 为 `709/1878/2213`。GASR 在总体和三个 stage
改善尾部，但 Stage 1 明显回退，且聚合吞吐没有收益。不能只挑 p95 改善或
SRAM 事务下降来宣传。

按 `batch_windows × heads` 做 population 近似，并按 sample 做 20,000 次
配对 cluster bootstrap 后：

| 指标 | 值 |
|---|---:|
| population-weighted GASR/Direct | 0.9872x |
| population-weighted 完美双模上界 | 1.0414x |
| paired-sample bootstrap 95% CI | [0.9807, 0.9939] |

该加权使用严格逆概率权重 `(batch_windows × heads) / 4`。它没有改变方向，
反而加强了 GASR-reset 吞吐退化的结论；bootstrap 只重采样 100 个 sample，
条件于当前确定性 rotating window/head selection，不覆盖 selection uncertainty，
也不是 full encoder 实测。

## 5. 为什么 GASR 省访问却不提速

GASR 把同一 source 的部分和驻留在 context 中，显著减少 backing SRAM 的
RMW；但 geometry/source 边界、dirty eviction、refill/writeback 给 term
ready 路径增加了阻塞。实测 term stall 从 `533,689` 增到 `580,809`，抵消
了 SRAM 事务减少带来的潜在收益。

这说明“减少 SRAM transaction”不等价于“减少 cycle”。它可能降低动态
功耗，但在没有 SAIF、SRAM macro 能量和目标工艺 PPA 前，只能列为待验证的
功耗假设。

## 6. 自适应双模是否值得继续

事后令每个 group 零成本选择 Direct/GASR 中更快者，得到不可实现的完美
oracle 上界：

| 项目 | 值 |
|---|---:|
| oracle 周期 | 5,388,120 |
| 相对 Direct 上界 | 1.0466x |
| 最大可节省周期 | 251,159 |
| 每 group 平均预算 | 52.32 cycle |

即使忽略 selector、状态迁移、额外面积和预测错误，等权 `1.0466x` 与
population-weighted `1.0414x` 上界都远低于 `1.20x`
门槛。因此当前 Direct/GASR 两模式上不再扩展 throughput-oriented adaptive
selector。若以后 SAIF 证明 GASR 的能量收益足够大，可以把它作为
energy mode，而不能称为吞吐创新。

## 7. 对 Local5 架构的约束

本轮得到四条可执行结论：

1. Direct 1RW 保持 Local5 当前吞吐基线；
2. GASR-reset 从吞吐主线降级为待 SAIF 证明的能量候选；
3. v2/v3 Python stall predictor 永久退出正式候选裁决，后端周期必须来自
   完整 RTL 回放；
4. 下一轮收益必须来自 relation 生命周期、跨 head/window 的真实复用或
   前后端重叠，而不是在 Direct/GASR-reset 两个现有 backend 间做选择。

这也意味着跨 head preserve 的 `GASR2C-P` 不能因本轮 80.33% 事务下降而
自动晋级；必须先实现 preserve RTL，并在同窗全 head 生命周期下重新做
Acc32 miter、周期和事务对照。

## 8. 产物与回归

```text
scripts/generate_local5_active_projection_postg0_vectors.py
scripts/summarize_local5_exact_backend_rtl_replay.py
sim_qfit/run_local5_exact_backend_rtl_replay.sh
tests/test_summarize_local5_exact_backend_rtl_replay.py
tb_qfit/vectors/local5_bb1e4_active_projection_postg0_all4800/
results/local5_bb1e4_exact_backend_rtl_all4800_v4_20260810/
```

验证结果：

- Direct Verilator：`4800/4800 PASS`；
- GASR-reset Verilator：`4800/4800 PASS`；
- 汇总器单测：`5/5 PASS`；
- producer P0 定向单测：`3/3 PASS`；
- Direct 仿真 wall time：`25.15 s`；
- GASR-reset 仿真 wall time：`30.05 s`；
- runner 的 Bash 特殊变量 `GROUPS` 冲突已修为 `GROUP_COUNT`；
- `REUSE_VECTORS=1` 会重验 manifest、全部 vector artifact 和 source trace SHA，
  并检查 artifact 集合、shape、entries、逐文件行数，不允许静默复用损坏向量；
- `source_sha256.txt` 在仿真前生成，绑定 runner、生成器、汇总器、全部 RTL
  和 vector manifest；`result_sha256.txt` 绑定工具版本、日志与报告；
- 新证据包的 source/result SHA 已逐项 `sha256sum -c` 通过。

## 9. 证据分档

| 声明 | 证据 |
|---|---|
| 4800 组 Acc32 等价与实际 RTL 周期 | `[rtl]` |
| 真实 descriptor/gate/K、100 sample 与 12 block 绑定 | `[prof]+[rtl]` |
| 1RW SRAM transaction 计数 | `[rtl计数]`，不是能量 |
| 完美双模 oracle 1.0466x | `[模型上界]`，事后且不可实现 |
| population 近似 0.9872x、cluster bootstrap | `[prof]+[统计]`，不是full encoder |
| GASR 可能节能 | `[待验证]` |
| 同窗全 head preserve / ERM 联合收益 | `[待验证]` |
| full encoder FPS、DC/STA/SAIF/PTPX | `[待验证]` |

## 10. 下一步

先进行独立 DATE 审稿，重点检查：

1. 该回放是否真正关闭旧 stall predictor 的 P0；
2. synthetic weight 是否被正确限制在数值 miter 范围；
3. sampled group 与 full workload 的边界是否清楚；
4. 是否有理由继续 GASR preserve，还是应把资源转向 exact relation reuse；
5. 哪些 P0/P1 必须在定义 V4 正式候选前修复。

只有评审通过并修完 P0/P1，才冻结 V4 候选和修复 selection-plan writer。

## 11. 第一轮独立 DATE 评审

独立审稿人复算两条日志后确认：

1. 旧 Python stall predictor 的 P0 已对 Direct/GASR-reset 关闭；
2. `0.9921x` 与 `-80.33%` transaction 没有会翻转结论的汇总 bug；
3. GASR-reset 只能保留为功耗候选；
4. 当前双 reset-mode 的自适应吞吐 selector 已被完美 oracle 上界否决；
5. 跨 head preserve 是另一种生命周期，不能由本轮直接否定，也不得自动晋级。

评审分数：

| 维度 | 分数 |
|---|---:|
| 证据可信度 | 3/5 |
| 方法严谨性 | 3/5 |
| 架构价值 | 2/5 |
| 投稿就绪度 | 1/5 |

裁决为 `Reject`。拒绝主因不是本轮负结果不可信，而是发现两个 V4 启动链
P0，以及 provenance、population weighting 和证据口径 P1。

## 12. 评审后修复

### 12.1 Zombie 与 GPU lease P0

旧 producer 在 foreign GPU PID 出现后正确写入 `INVALID`，但 child 成为
zombie。旧 `stop_process_group()` 在 `wait()` 前等待 PGID 消失，导致父进程
永久卡住并持有 GPU lease。

现已改为：

```text
poll/reap exited leader
  -> bounded SIGTERM wait
  -> bounded SIGKILL
  -> wait leader
  -> residual PGID fail-closed
```

zombie 定向回归通过。现场旧 parent `3134907` 与 zombie child `3748413` 已
终止，重启前 watcher lock 与 `gpu_profile_lease.lock` 均验证为 `FREE`。
后续重启过程见第 14 节。

### 12.2 Selection plan 覆盖 P0

selection plan 现改为：

1. `build_selection_plan()` 只构造确定性字节；
2. 首次写入使用 `O_CREAT|O_EXCL` 和 `fsync`；
3. 文件已存在时只能逐字验证；
4. 任一差异直接报错，禁止覆盖。

现有 1200-record plan 与新生成器逐字一致。新增：

```text
contracts/local5_joint_trace_plan_freeze_receipt_v1_20260810.json
```

receipt 绑定 plan SHA、生成器 SHA、sampling seed、cohort、records 和本地 Git
blob；run identity 原生绑定 receipt 路径与 SHA。其状态明确为
`LOCAL_BYTE_ANCHOR_NOT_EXTERNAL_TIMESTAMP`，不是外部时间戳，也不是 V4
candidate preregistration。

本地 blob 已使用 `git hash-object -w` 写入 object database，并通过
`git cat-file -e` 与逐字 `cmp`。producer 和 leaf profiler 都会在启动时读取
该 blob 并与 plan 字节比较，不能只伪造 receipt 字段。

### 12.3 统计与 Provenance P1

汇总器已增加：

- vector metadata 与 source manifest 逐 group 身份核对；
- 唯一末尾 PASS 和失败标记检查；
- population-weighted 近似和 paired-sample cluster bootstrap；
- 精确的 `projection_start -> flush/done` 周期边界；
- accumulator bank 执行期 transaction 边界。

runner 已增加完整 artifact shape/entries/line-count/SHA 复用检查、编译合同、
工具版本、仿真前 source provenance、仿真后 result provenance 和最终
`complete.json`。

### 12.4 尚未关闭

1. V4 尚未定义和启动；
2. 跨 head preserve 尚无 RTL，不进入 V4 throughput 候选；
3. 31 个既有 Verilator `WIDTHEXPAND` 警告仍需清理；
4. 所有关键源当前仍是 untracked，证据只有 content-level SHA，不是
   commit-level reproducibility；
5. 还没有 SAIF、SRAM macro 能量、DC/STA/fmax；
6. 下游旧 CPU evaluator watcher 已主动终止，防止上游完成后
   自动运行已否决的旧 V2/V3 候选；当前没有 V4 evaluator watcher。

## 13. 第二轮独立 DATE 复审

第二轮复审结论分为两层：

| 范围 | 裁决 |
|---|---|
| 本工作包内部裁决 | `Pass` |
| 整篇 DATE 投稿 | `Reject` |

复审确认两个 P0 已关闭，producer 可以重启并且只采集无偏 joint-head
workload，不授权启动 V4 evaluator。评分提升为：

| 维度 | 分数 |
|---|---:|
| 证据可信度 | 4/5 |
| 方法严谨性 | 4/5 |
| 架构价值 | 2/5 |
| 工作包就绪度 | 4/5 |

复审后又关闭了三个非阻塞 P1/P2：

1. leaf profiler 现在直接验证 freeze receipt 路径/SHA、runner SHA 和本地
   blob 内容，绕过 producer 也会 fail-closed；
2. population 绝对 weighted-cycle 补上 `/4`，比值保持 `0.9872x`；
3. 报告改为每模式 `4.32M` 坐标、两模式共 `8.64M` 次比较，并明确
   bootstrap 条件于当前 selection design。

仍然不能把本工作包写成 DATE 正向贡献。它的价值是可靠地否决
GASR-reset throughput 路径，并为下一轮同窗全 head 生命周期数据采集解除
P0。正向架构必须从 exact relation reuse、跨 head preserve 或前后端真实
重叠中取得实测收益。

## 14. Joint-head producer 重启与 GPU PID namespace 审计

### 14.1 两次 fail-closed 负结果

当前容器内 `/proc` 和 NVML 暴露的 PID namespace 不一致。前两次重启在
child 完成 CUDA 初始化后，NVML 分别报告 `619733` 和 `687421`，但它们
无法通过容器 `/proc/*/status` 的 `NSpid` 链映射到 child。旧规则因此把
它们当作 foreign compute PID 并两次 fail-closed：

| 尝试 | 启动时间 | NVML PID | 结果 |
|---|---|---:|---|
| 1 | `2026-08-10 11:53:18` | 619733 | `INVALID`，有界终止并释放两把锁 |
| 2 | `2026-08-10 11:59:05` | 687421 | `INVALID`，有界终止并释放两把锁 |

这两次不是 profile 结果，不进入 workload 证据。它们只证明 zombie
清理和 lease 释放路径已经在真实 CUDA 进程上执行成功。

### 14.2 唯一未映射 PID 认领协议

独立增量评审后，producer 采用下列保守规则：

1. 启动前 NVML compute PID 集必须为空；
2. child 存活期间，若所有 NVML PID 均无法映射，只允许第一次出现的
   **唯一** PID 作为当前 child 的 namespace alias；
3. 出现第二个未知 PID、PID 替换、首次多 PID 歧义，或已直接映射自身
   却又出现 alias，都立即 fail-closed；
4. child 退出后必须先 reap，再等待 NVML compute PID 在 30 秒内清空；
5. 只有 exit code、manifest/payload 完整性和 GPU-clear 后置条件同时成立，
   `gpu_exclusivity_audit.json` 才能从 `RUNNING_UNVERIFIED` 变为 `PASS`。

该协议的单元回归为 `6/6 PASS`；leaf profiler 对 receipt/blob/
runner/plan 的绕过检查在 `sdformerflow` 环境为 `3/3 PASS`。这些是
控制面验证，不是 workload 结果。

### 14.3 第三次运行中状态

第三次于 `2026-08-10 12:06:57` 启动，producer PID 为 `3772496`，
leaf child PID 为 `3772528`。在 `12:09 UTC` 的本轮核对时：

- producer/child 仍存活，已跨过多个 2 秒监控周期；
- NVML 报告唯一未映射 compute PID `777824`，显存约 `5658 MiB`；
- 已生成 checkpoint projection contract，正式 ordered payload 尚未生成；
- 审计状态仍为 `RUNNING_UNVERIFIED`；
- V4 evaluator 和 evaluator watcher 均未启动。

因此，本节只能记为 `[运行中审计]`，**不是** `[prof] PASS`。只有最终
audit 转为 `PASS` 且 manifest/payload SHA 通过时，才能启动联合 head
候选筛选。
