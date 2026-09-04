# Local5 EREP 精确 Relation Epoch 候选预注册

> 日期：2026-08-10  
> 前序：`docs/286_Local5全4800Group精确五BankRTL回放_20260810.md`  
> 本轮唯一问题：在正式 joint-head workload 结果出现前，冻结
> Local5 下一个可被否决的架构候选、强基线、资源和晋级门槛。

## 1. 结论

本轮冻结的唯一可晋级 Local5 候选为：

> **EREP-S2: Exact Relation Epoch Pipeline with a two-output stripe**  
> **双 output stripe 的精确关系世代流水。**

它不是把 relation memo、FIFO 和 ping-pong 分别写成三个贡献，而是一个
统一 dataflow：

1. 把每个 `window/input-head` 的精确 Local5 relation 定义为不可变
   **relation epoch**；
2. 同一 epoch 在两个 output tile 上保持 stationary，权重和五 bank Acc
   上下文按 tile 切换；
3. consumer 执行当前 sealed epoch 时，producer 在另一个独立 1RW slot
   中构建下一 epoch；
4. slot 只能在完整 seal 后可见，并在对应 consumer 退休后复用；
5. 现有 Direct 合法单端口 1RW 五 bank backend 不变。

候选合同已在正式 manifest 不存在、GPU audit 仍为
`RUNNING_UNVERIFIED` 时冻结：

```text
contracts/local5_erep_candidate_prereg_v1_20260810.json
contracts/local5_erep_candidate_prereg_v1_receipt_20260810.json
```

合同 SHA-256 为
`54850b005f3c35884a7466ed0682f1ff872823ad7ce1ce79a9733b7339238775`，
本地 Git blob 为 `97c3b2f1f32acaeb5d52b82ec6d611b912af5a5f`。这只是
本地字节锚，不是外部可信时间戳，也不是 commit-level provenance。

## 2. 为什么不再单独推进旧机制

| 候选 | 当前证据 | 裁决 |
|---|---|---|
| GASR-reset | `[rtl]` `0.9921x`，周期负收益 | 否决 throughput 路径 |
| Direct/GASR reset-mode oracle | `[模型上界]` `1.0466x` | 否决双模式 selector |
| cross-head Acc preserve | `[rtl]` 公共向量边界只有 `1.0557x` | 只保留为能量/存储消融 |
| empty bypass | `[prof]+[模型上界]` 即使全部免费也约 `1.18x` | 不达 `1.20x` 门槛 |
| exact reuse-only | `[待验证]` | 只作 EREP 消融 |
| overlap-only | `[待验证]` | 只作 EREP 消融 |

EREP 的收益假设来自现有串行路径中可见的 relation/frontier 固定项。
4800-group Direct RTL 中 empty group 为 `456` 周期，整体已校准固定项约
`456--459` 周期。这只证明有流水化空间，不证明 EREP 必然有收益。

## 3. 核心数据流

```text
Local5 score/Shiftmax5 + real invalid mask
        |
        v
relation transpose / active frontier producer
        |
        |  one exact 112-bit active-source record per accepted source
        v
+----------------------+    atomic seal    +----------------------+
| relation epoch slot0 | <----------------> | relation epoch slot1 |
| 450 x 112-bit 1RW    |                   | 450 x 112-bit 1RW    |
+----------------------+                   +----------------------+
        | current sealed epoch                  | next epoch fill
        v                                       |
source-major term builder FIFO2                 |
        |                                       |
        v                                       |
Direct legal-1RW TCFM5 backend <----------------+
        |
        | stripe0: output tile 0,1
        | stripe1: output tile 2,3
        | ... one-tile remainder is explicit
        v
two 450x32xAcc32 contexts -> common vector drain/serializer
```

### 3.1 Epoch 身份

一个 epoch 的身份至少包含：

```text
{sample, stage, block, window, input_head, checkpoint/config identity}
```

记录 payload 是真实 `{K bitmap, gate[5], valid mask}` 导出的 active-source
descriptor。v1 明确禁止 cross-block、cross-window 和 cross-head alias，不把
gate-valid 相等写成完整 relation 相等。

### 3.2 S2 调度

output tile 按递增顺序分成宽度 2 的 stripe，尾部不足 2 时显式执行
宽度 1。每个 stripe 内，input head 仍按原 hardware-order 递增：

```text
for output stripe in increasing order:
  for input head in increasing order:
    build/seal relation epoch once
    execute that epoch on every output tile in the stripe
```

因为不同 output tile 使用独立 Acc32 地址空间，这个 loop interchange 不改变
任一 output tile 内的 input-head 累加顺序。该等价性仍必须由 Acc32
miter 而不是文字推理最终闭合。

## 4. 资源与端口合同

| 状态 | 数量 | 单个大小 | 总计 |
|---|---:|---:|---:|
| Acc32 context | 2 | `450x32x32-bit` | `112.5 KiB` |
| relation epoch slot | 2 | `450x112-bit` | `12.3047 KiB` |
| Direct 基线 Acc32 context | 1 | `450x32x32-bit` | `56.25 KiB` |
| EREP 相对 Direct 新增数据状态 | - | - | `68.5547 KiB` |

上表尚未包括 tag、valid、occupancy、顺序号和队列控制 bit；它们在 RTL
和 PPA 中必须完整计入，不得估为 0。

每个 epoch slot 只有一个单端口 1RW 端口。实现上不允许 producer 和
consumer 同时访问同一 slot；并发来自 slot0/slot1 的物理分离，不依赖
虚假 1R1W 存储。

## 5. 四方强基线消融

| ID | relation reuse | 前后端 overlap | stripe | 论文角色 |
|---|:---:|:---:|---:|---|
| C0 Direct-serial | 否 | 否 | 1 | 强基线 |
| C1 reuse-only-S2 | 是 | 否 | 2 | 单机制消融 |
| C2 overlap-only | 否 | 是 | 1 | 单机制消融 |
| C3 EREP-S2 | 是 | 是 | 2 | 唯一可晋级候选 |

四者必须共用：

- Direct 合法单端口 1RW 五 bank backend；
- 相同 output weight 服务顺序和 transaction-indexed 反压；
- 相同 `450x1024-bit` 向量 drain 和 serializer；
- 相同真实 invalid mask、hardware-order 整数语义和 Acc32 边界。

## 6. 周期模型

预注册参考器：

```text
scripts/local5_erep_schedule_reference.py
tests/test_local5_erep_schedule_reference.py
```

模型 SHA-256 为
`904c9e6f670da457ed21bb13ba8625707b2c374bf384c0e1542bd56e8b5775dd`，
`4/4 PASS`。

两 slot 有界规则为：

```text
fill_start[i] = max(fill_done[i-1], consume_done[i-2])
fill_done[i] = fill_start[i] + F[i]
consume_start[i] = max(consume_done[i-1], fill_done[i])
consume_done[i] = consume_start[i] + E[i]
```

其中 `F[i]` 与 `E[i]` 不允许沿用已被 held-out RTL 否决的 v2/v3 Python
stall predictor。正式输入必须来自新增相位计数的真实 Direct 1RW RTL：

- `F`：`projection_start` 到 sealed relation/frontier 可消费；
- `E`：一个 sealed epoch 在一个 output tile 上经 builder、Direct backend 到
  flush 的周期；
- drain：四候选公共显式相加。

正式模型还必须输出 producer slot stall、consumer wait、queue occupancy、
relation read/write 事务和 Acc SRAM 事务，不能只输出一个 ideal speedup。

## 7. 事前晋级门槛

C3 只有同时满足下列条件才允许进入新 RTL：

1. inverse-probability 加权 mean speedup 相对 C0 `>=1.20x`；
2. sequence-cluster bootstrap 95% 下界 `>1.00x`；
3. 每个 stage 的加权 p95 都不回退；
4. C3 相对 C1/C2 中更快者仍 `>=1.05x`，否则不能把两机制组合
   单列为架构贡献；
5. 完整 hit/miss/remainder/epoch invalidation 下 Acc32 mismatch 为 0；
6. 同宏、同 SDC、同 floorplan 规则下，开放代理 EDP 或面积归一吞吐
   改善 `>=20%`；
7. 最终 ASIC PPA 主张仍必须由 DC/STA/SAIF/PTPX 重做。

若 C3 失败第 1--3 条，则暂停 Local5 新架构 RTL，不事后降低门槛。
若只有 C1 或 C2 过门槛，其结果先记为消融，必须重新通过独立
创新性评审才能晋级。

## 8. 正式 workload 需求

joint-head payload 必须 fail-closed 满足 `100 sample / 12 block / 1200 joint
window / 13800 head group`，并输出：

1. 每个 epoch 的完整身份、active source、term、update 和 empty 标志；
2. 按 `{K, gate[5], valid}` 完整全等定义的 relation 等价类；
3. 真实 active record 存储量、epoch read/write 事务和 remainder stripe；
4. 仪器化 RTL 的 front/execute/flush/drain 相位周期；
5. 加权 mean/p50/p95/p99 与 sample/sequence 配对统计；
6. 按 stage 的 producer stall、consumer idle 和端口占用。

## 9. 过模型门槛后的最小 RTL

只在 C3 过门槛后实现：

1. 两个独立 450x112-bit 合法 1RW epoch slot；
2. `{FREE, FILL, SEALED, CONSUME}` 世代状态和原子 seal；
3. S2/remainder 调度器，不改变每 output tile 的 head order；
4. 两个五 bank Acc32 context，同一 Direct 1RW execution lane 时分复用；
5. relation、term、bank 和 final consumer 四边独立随机反压；
6. SVA：未 seal 不可见、slot 不提前覆盖、严格有序退休、有界活性；
7. Direct/C1/C2/C3 同 trace 的全 Acc32 miter。

## 10. 证据边界

| 声明 | 当前证据 |
|---|---|
| EREP 候选、资源和门槛在 formal result 前冻结 | `[本地字节锚]` |
| 双 slot 有界调度参考器 | `[单测] 4/4 PASS` |
| 4800-group Direct/GASR 数字 | `[rtl]`，见 docs/286 |
| joint-head profile | `[运行中审计]`，非 `[prof] PASS` |
| C0--C3 正式周期 | `[待验证]` |
| EREP RTL/Acc32/反压 | `[待验证]` |
| Yosys/OpenROAD 代理 | `[待验证]` |
| DC/STA/SAIF/PTPX | `[待验证]` |

因此，EREP 当前只是一个被事前门槛约束的架构假设，**不是**
已完成的 DATE 贡献。

## 11. 独立 DATE 评审与 v1 作废

独立评审对本 v1 给出 `Major Revision`：

| 维度 | 分数 |
|---|---:|
| 证据可信度 | 3/5 |
| 方法严谨性 | 2/5 |
| 创新性 | 2/5 |
| DATE 潜力 | 2/5 |

评审确认 EREP 具有 Local5 relation 生命周期和精确顺序约束，不是
任意组合；但当前可见机制仍会被读成 cache/replay、loop interchange、
双缓冲和双 Acc context。更关键的是，v1 有四个会使正式裁决
无效的 P0：

1. C0--C3 未逐候选冻结完整资源和状态生命周期；
2. S2 改变全局 weight 请求顺序，却仍使用全局第 n 笔事务服务合同；
3. v1 参考器只是 F/E 标量递推，未建模 tile-specific 执行、slot/context
   与 drain 合法时间线；
4. 加权 speedup、cluster bootstrap 和 stage p95 多重比较未唯一冻结。

因此 v1 已由下列文件明确作废，禁止读取 formal result 后用 v1
做候选裁决：

```text
contracts/local5_erep_candidate_prereg_v1_invalidation_20260810.json
```

替代合同与修复见
`docs/288_Local5_EREP_v2评审修复与分阶段门槛_20260810.md`。
