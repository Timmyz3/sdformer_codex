# M1571：M1570 decoder RSS 失败独立法证

结论：M1570 的唯一一次尝试已经消耗，不能重试；它在首个 `DENSE_TYPED_K8` 配置尚未完成时失败，所以没有任何可引用的周期、流量或加速数据。失败点可定位到 D0 第 0 个 destination 结束后的 `retire_destination()` RSS 检查。失败回执没有记录实际 `ru_maxrss` 或同时刻 `VmRSS`，因此只能证明峰值检查读到了至少 8 GiB，不能从现有回执证明哪个分配产生了这 8 GiB。

## 1. 失败事实

- 输出目录严格只有 `WORK_STARTED.json` 与 `FAILED_OR_INCOMPLETE.json`，不存在 `partial_*`、`result.json`、`RUN_COMPLETE.json` 或 SHA 封存。
- 两个回执的文件时间只相隔 0.144002376 秒。
- `completed_configurations=0`、`attempt_consumed=true`、`automatic_retry=false`。
- M1560 的顺序是先调用 `stream_actual_call(config)`，返回后才 append 并写 partial。因此首个 DENSE 没返回，三组配置一组也没有完成。
- 唯一错误文本来自 M1556 `memory_gate()` 的 `ru_maxrss < 8388608 KiB` 条件。

风险等级为 critical：decoder-complete 周期行仍为空。该失败不是负性能结果，也不能用于任何系统倍速或能量结论。

## 2. 第一个 RSS 门准确落点

D0 几何为 `Cin=1536, Cout=384, Hin×Win=15×20, Hout×Wout=30×40`。首个角点 destination 只有一个 source site，因此 DENSE 生成 1536 个 contributor、192 个 K8 group、4 个 output block。静态逐行枚举得到每个 output block 含 commit 为 1153 个 request；加上 source/control 的 3 个初始 request，在首次 destination retirement 前一共 4615 个 request。

这小于 65536-request fallback 门，所以首个可达的 RSS 门不是 `one()` 中的周期门，而是 M1556 第 266 行、destination 0 后的 `retire_destination()`。不可恢复的缺口是 M1556 只把通用异常字符串写入失败回执，没有写 gate ordinal、精确 peak 或 current RSS。

## 3. Python 内存结构分解

以下是结构界，不是 RSS 测量：

| 结构 | 首门前规模 | 判断 |
|---|---:|---|
| 不可变 bit-plane snapshot | 576,000 B | 有界，非 8 GiB 来源 |
| contributor 列表 | 1,536 tuple；deep proxy 154,316 B | 可进一步流化 |
| 返回的 K8 groups | 192 group；deep proxy 164,484 B | 可改 iterator |
| contributor 分桶+group 同时存活 proxy | 178,338 B | MiB 以下 |
| dependency token dict | 4,611 token；synthetic deep proxy 658,267 B | 首要真实减内存对象 |
| weight cache / calendars / outstanding | 9 tile、端口约束队列 | 有界且不能为省内存重置，否则周期模型失真 |

M1560 在写 `WORK_STARTED` 前、同一解释器内完成全 payload preflight。M1521 目录共有 124 个文件、261,265,385 B，最大单文件 4,656,000 B；哈希代码按 1 MiB block 流式读取。可见 preflight 也没有 GiB 级容器，但因为检查指标是进程终身 high-water，任何此前阶段或启动环境的历史峰值都会污染首次 replay 门。

独立 64 MiB mmap 探针验证了指标语义：释放后 current RSS 从 75,856 KiB 回到 10,440 KiB，而 `ru_maxrss` 仍保持 75,856 KiB。故 `ru_maxrss` 不是当前活跃内存。结合 0.144 秒失败与上述 MiB 级可见对象，最强解释是进程历史 high-water 污染，而不是第一个 destination 本身生成 8 GiB live set。此项置信度为 medium-high，不冒充已证明根因；启动 ancestry 和实际两项 RSS 未被记录。

## 4. 最小 successor：不放宽 8 GiB，且真正降内存

必须同时改测量边界和 live-set 形状：

1. 保留 8 GiB 上限，禁止提高或删除；全 payload preflight 在外层进程完成，replay 用 fresh-exec worker。worker 在开 payload 前记录 `VmRSS` 与 `ru_maxrss`，运行时对 current `VmRSS` 做硬门，fresh worker 内的 `ru_maxrss` 保留为 peak 门与遥测。
2. 每次失败必须原子记录 config/timestep/destination/output-block/request-count、`VmRSS`、`ru_maxrss` 和 gate ordinal。这样下一次才能区分当前 live-set 与历史峰值。
3. 每个 `psum_write` group 后就淘汰已消费 dependency token，只保留 source/control barrier 与下一组需要的 previous-psum token。token live set 从 O(destination) 降到 O(K8 group)。
4. 将 contributor 列表加 `bank_unique_groups()` 返回列表改为 bank-order iterator，避免 contributor、八个 queue 和 groups 同时物化。
5. bank calendar、outstanding、address/commit digest 与九 tile weight cache 必须跨整 call 保留；`destination_transactions` 继续逐行消费，严禁改成 call-wide list。否则是牺牲模型正确性伪装省内存。

M1570 保持 `FAILED_OR_INCOMPLETE`。上述修改只能形成新 source，经独立 hammer/ordinary regression 后申请新的 fresh namespace one-shot；不能在 M1570 目录中补跑或覆盖。

## 5. 边界

本次只读检查了 M1570 两个回执、M1560 runner、M1556 source、M1539 request/memory path，并运行 64 MiB 的独立 Linux 指标语义探针。没有打开 canonical payload、没有重跑 pilot、没有 GPU/RTL/EDA、没有修改作者文件、没有产生周期或性能数据。
