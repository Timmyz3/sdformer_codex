# M697｜M693 decoder 官方 Prosperity adapter fresh static hammer

## 裁决

**80/100，P0/P1/P2 = 0/2/2，`NO_GO_M693_FULL_OFFICIAL_CPU_REPLAY__P1_2__AUTHOR_R2_REQUIRED`。**

M693 的数学对象、外部输入身份和 claim boundary 基本干净：M692/M686 顶层与 nested 双封、40-cell payload、D1 非 exact 分流、phase/tap/K 顺序、partial M/K/N、官方仓库 commit/clean/source SHA、fresh FC、禁止 `_fc` 名称、D0-only N128×3 miter、CLI+outer-env+反向 SHA 绑定均通过独立静态检查。Python 3.10 的冻结测试仍为 `27/27`。

但当前不能授权正式 CPU replay，因为 canonical report 和失败协议各有一个 P1 合同缺口。

## P1

1. **缺 contract-required phase aggregate。** `required_reporting` 明确要求 phase / sample-module / module / sample / overall 的整数计数器聚合；实现的 `aggregate_breakdowns` 只有 `overall`、`sample:*`、`module:*`、`record:*`。每条 record 内的 phase 叶节点不等于跨 population 的 `phase:{3,2,1,0}` ratio-of-sums 聚合。修复需加入四个 phase bucket，并测试 `sum(phase counters)==overall counters`。

2. **失败双封只覆盖 `atomic_publish` 内部。** `main` 在进入 `atomic_publish` 前已完成 authorization、preflight、30 exact records 和 10 diagnostic records 的正式执行；这一区间任何异常都不会生成 contract 要求的双封 failure receipt。修复需在运行前建立 fresh 非 canonical failure staging，或给 `main` 加统一 fail-closed wrapper，并注入一次 `execute_records` 失败测试。

## P2

- 1/2/3 workers 一致性目前只有 `executor.map` 保序、最终排序和 worker-local 初始化的静态理由。r2 至少应把唯一命令锁为 `--workers 3` 并写入 receipt；可再用注入 worker 测试补哈希一致性，不要求先把官方 workload 跑三遍。
- `not output.exists()` 与 `os.rename` 之间有单写者假设；如不引入 no-replace publish，需显式记录这个假设。

## 已通过的独立攻击

- 作者 handoff 双封与 exact population；runner/contract/test/docs359 exact SHA；
- strict JSON duplicate/nonfinite、path traversal、leaf symlink、unsealed extra member；
- M692、M686 top/runtime/weights 外部 root 和 nested seals；
- 40-cell 全量 SHA/popcount/tail，D0/D2/D3 exact subset 与 D1 diagnostic-only；
- phase/tap/K/order、partial M/K/N 独立复算；D1/D2/D3 partial-N expansion 拒绝；
- 官方 repo commit/clean/source SHA；fresh FC、名称禁 `_fc`；
- authorization 的 CLI、outer-env、exact status/booleans 和 runner/contract/test/M686/M692 反向绑定；
- exact decoder complete cycle/speedup 仍为 null；`ours/full/system/headline=false`。

本审阅没有运行 production M672 workload、没有导入或调用官方 `Simulator.run_fc`，没有生成周期或倍率，也没有运行 GPU/EDA。因为 P1 非零，`execution_authorized=false`，不提供 CPU 命令。
