# Local5 Formal Phase Telemetry 方案与独立 DATE 审阅

> 日期：2026-08-12  
> 范围：Local5 formal G0 的 phase/resource 证据方案  
> 当前证据：`[rtl-anchor]+[参数化证明候选]+[模型]`  
> formal G0：**DENY**

## 1. 裁决

现有证据不能直接进入 formal G0：

1. numeric v5 只证明 Acc32 数值等价，没有逐 head/tile phase 观测；
2. H3/H6/H12/H24 Phase Array Store 保存服务握手、边界和内部状态，不含正式
   scheduler 所需的 epoch、FIFO 和五个 Acc bank 完整资源命令；
3. `fill/execute` 与 C1-C4 含参数化推导和调度模型，不能统一标成 RTL raw trace。

独立 DATE 风格审阅对当前完成度给出：

```text
2.3/5，Reject for formal G0
```

推荐方法为：

```text
telemetry-only sealed release
  + 完整逐事件 phase canary anchors
  + 参数化流式证明
  + 独立只读 merge/replay
```

该方法设计评分为 `4.1/5 Conditional Accept for G0 methodology`，但尚未实现，
不能改变 G0 状态。

## 2. 为什么不能复用 numeric v5

numeric v5 使用 `USE_MEMO=0 + TRANSACTION_INDEXED_SERVICE=1`。runner 只传入输入、
权重、坐标、服务 seed 和 `ACTUAL_ACC_FILE`，没有 `IDENTITY_TRACE` 或正式 phase
输出。窗口日志中的 `weight_cycles/frontend_cycles/rmw_cycles` 是整窗累计量，无法恢复：

- 462,600 个 phase 的开始、结束与 duration；
- phase 对应的 window/head/role/tile；
- relation workspace、epoch slot、FIFO2、五个 Acc bank、prepare/drain 的 accepted
  command cycle 与 identity；
- ready/valid stall、资源冲突和 phase 间等待；
- `RTL_DIRECT`、`RTL_PRIMITIVE_ANCHOR` 与 `PARAM_DERIVED` 的来源差异。

因此 100/100 numeric Acc32 和 phase formal 是两个独立硬门槛，不能互相替代。

## 3. 为什么不能全量复用旧 phase canary

正式规模为：

| 项 | 数量 |
|---|---:|
| window | 1,200 |
| input-head | 13,800 |
| phase | 462,600 |

若把现有完整状态/握手 trace 直接扩到 1,200 窗，预计约 `17.746B` 行，Phase Array
Store 约 `427 GiB`，还未计原始 CSV。旧 v4 resource-event archive 的主展开事件也超过
`1.175B`。这类全展开既不经济，也不增加语义严格度。

纯 template+patch 同样不够：它只是一种无损表示，不能凭自身成为新的 RTL 观测来源。

## 4. 冻结的三层证据

### 4.1 RTL_DIRECT

由只增加被动 monitor 的 sealed release 在全部 1,200 窗实测：

- phase start/end/duration；
- accepted resource command 的数量、first/last cycle 和有序 identity digest；
- stall、overflow、protocol error 与资源冲突计数；
- prepare/direct/drain 的实际延迟；
- final Acc32 actual 的绑定摘要。

### 4.2 RTL_PRIMITIVE_ANCHOR

对四种 H、18 个 cohort cluster、空/稀疏/稠密极值和随机反压窗口保存完整逐事件
trace。anchor 用来证明 compact monitor 与完整 canary 逐事件同构，并校准 EREP
primitive 的 fill/execute 资源行为。

### 4.3 PARAM_DERIVED 与 SCHEDULE_MODEL

允许由解析规则推导：

- canonical 1,200-window 拓扑、H、head/tile 顺序；
- 462,600 个 phase descriptor；
- HxH task、静态地址和 bank 映射；
- profile 中 source/term identity 与期望命令数；
- C0-C4 调度结果。

这些结果必须标成 `[参数化证明]` 或 `[模型]`，且要与 RTL observed count/digest
比较。禁止把 profile 计数、delay table 或 H 闭式公式直接复制到 actual 字段。

## 5. Phase Telemetry v1 Schema

每个 sample 一个 shard，恰含 4,626 个 phase；100 个 shard 合并后必须恰为
462,600：

```text
phase_window/head/role/tile
phase_origin
phase_start/end/duration
phase_resource_offsets

resource_code/event_count
resource_first/last_cycle
resource_cycle_identity_sha256
resource_calendar_rle_offsets/data

phase_digest
anchor_reference
```

`phase_origin` 冻结为：

```text
RTL_DIRECT
RTL_PRIMITIVE_ANCHOR
PARAM_DERIVED
SCHEDULE_MODEL
```

来源升级、缺失或非法组合必须 fail closed。`PARAM_DERIVED` 和 `SCHEDULE_MODEL` 永远
不能通过重命名晋级为 `[rtl]`。

## 6. 实现顺序

1. 新增被动 semantic monitor，不修改 synthesizable DUT；
2. 建立 H3/H24 pilot 和 sealed telemetry release；
3. 对同一输入同时生成完整 canary 与 compact telemetry，逐事件展开比较；
4. 加入 H6/H12、随机反压与 18-cluster anchor；
5. 实测 RLE/varint archive 的 wall time、RSS 和磁盘；
6. 扩到 100 sample/1,200 window；
7. actual/expected 只读 merge，并由独立 parser 从底层证据重算 C0-C4；
8. 与 100/100 Acc32 numeric 一起生成 admission receipt。

预期资源包络：compact descriptor/digest 目标 `<0.5 GiB`，精确 RLE/varint calendar
预计 `1-5 GiB`，anchors 预计 `5-15 GiB`。这些是设计估计，pilot 前标为 `[模型]`。

## 7. 必须关闭的负测试

- phase 缺失、重复、乱序和 cycle 平移；
- identity 对调和 resource relabel；
- `PARAM_DERIVED -> RTL_DIRECT` 来源伪造；
- 错误 anchor 引用、stage/H 不匹配和截断 shard；
- 错误 executable 或 release binding；
- actual/expected 同源复制和 digest 重绑；
- 空、稀疏、稠密窗口与随机反压；
- C0-C4 上层 scalar 同步篡改但底层 evidence 不变。

## 8. 独立审阅 P0/P1/P2

P0：

1. phase origin 分层，derived 不得冒充 RTL；
2. compact monitor 与完整 canary 逐事件同构；
3. 1,200/13,800/462,600 canonical 覆盖和只读 merge 完成；
4. admission 从底层 evidence 重算，不能接收自报 C0-C4。

P1：

1. anchors 覆盖 18 cluster、四种 H、密度极值和反压；
2. 至少 H3/H12 完成 Icarus/Verilator 交叉验证；
3. compact archive 的 wall/RSS/磁盘完成实测。

P2：

- 建立 Git tag、外部只读 release ledger 或签名信任根。

## 9. DATE 表述边界

允许表述：

> formal G0 使用全量 Direct compact RTL telemetry、分层 primitive RTL anchors 与
> 参数化调度证明；C0-C4 属于 `[rtl-anchor]+[参数化证明]+[模型]`。

禁止表述：

- 全 1,200 窗均完成 candidate RTL phase 回放；
- template/patch、RLE 或 telemetry 是 DATE 架构创新；
- formal G0 的模型周期是最终 candidate RTL 性能；
- G0 可替代投稿前的 candidate 多窗口 RTL、DC/STA/SAIF 或 ASIC PPA。

正式 candidate RTL 实现后，仍需用多窗口、多密度真实回放校验 G0 的模型排序和
性能主张。
