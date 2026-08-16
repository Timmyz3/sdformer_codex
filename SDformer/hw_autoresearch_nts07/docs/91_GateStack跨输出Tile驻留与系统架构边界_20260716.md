# GateStack 跨输出 Tile 驻留与系统架构边界

## 一、结论

本轮补上了两个此前只有模型、没有 RTL 证据的机制：

1. `output-tile-stationary + head-stacked replay` 的双层循环调度；
2. descriptor 从首次 build 到最后一个输出 tile 的有界驻留和末次使用释放。

但这仍不等于完整加速器签核。新 scheduler 是必要控制，不应单独包装成架构创新。真正可写入 DATE 的架构点是它与一次构建多次重放、统一计算后端、分层生命周期和真实 H67 workload 稀疏性的组合。

当前架构状态更新为：

```text
H67 ordered event build
        |
        v
Head Slot + Depth80 Descriptor Residency
        |
        v
Window/Output-Tile Scheduler
  tile外层: 3 / 6 / 12 / 24
  head内层: 3 / 6 / 12 / 24
        |
        v
Replay PLAN ----> Atomic COMMIT（叶级RTL已完成）
        |
        +--> Resident decoder
        +--> IPD32W decoder
        +--> RAW41 exact decoder
                  |
                  v
        Shared TDR + Multicast + AccTile
                  |
                  v
      bias/requant/final + context retire
```

## 二、为什么 output tile 必须是外层

H67 每个 attention stage 的投影是多个 input head 对同一 output tile 的累加：

```text
for output_tile:
    clear AccTile once
    for input_head:
        replay compact event representation
        read current output-tile weights
        accumulate locally
    add bias once
    emit final tile
```

该顺序具有三个直接硬件作用：

- AccTile 在所有 input head 期间驻留，不需要每个 head 把 partial sum 写回片外或大 SRAM；
- 同一个 head 的 descriptor/event payload 在不同 output tile 之间复用，不需要每个 tile 重建 OBI/IPD；
- Resident/IPD32W/RAW41 只改变前端表示解码，TDR、multicast 和 accumulator 后端保持统一。

如果改成 head 外层、output tile 内层，必须同时驻留多个 AccTile，或者反复读写 partial sum。对 stage3 的 24 个 32-channel 输出 tile，这会把存储端口和 partial-sum 流量推成主瓶颈。

## 三、真实网络尺度

H67 profile 中四级 attention 的 `head_dim` 均为 32：

| stage | heads | 32-lane逻辑输出tile | 每窗口head replay数 |
|---:|---:|---:|---:|
| 0 | 3 | 3 | 9 |
| 1 | 6 | 6 | 36 |
| 2 | 12 | 12 | 144 |
| 3 | 24 | 24 | 576 |

四级合计每组 stage sweep 为 45 个逻辑 tile、765 次 head replay。本轮 RTL 已对该完整循环计数和顺序做严格回归，不再只依赖 3×3 玩具配置。

## 四、OUT_TILE 口径修正

旧完整窗口模型固定 `output_lanes=32`，但实际顶层默认 `OUT_TILE=8`。两者此前不能直接对照。

联合 DSE 结果：

| OUT_TILE | 相对32-lane周期 | 通用单元 | memory bits | `$mul` |
|---:|---:|---:|---:|---:|
| 8 | 3.452x | 2,456 | 45,664 | 8 |
| 16 | 1.732x | 2,584 | 87,136 | 16 |
| 32 | 1.000x | 2,840 | 170,080 | 32 |

这里的 memory bits 主要随 AccTile lane 数增加，通用单元只用于结构趋势。后续冻结规则：

- 架构和周期模型默认 `OUT_TILE=32`，与 H67 `head_dim=32` 一致；
- 8/16-lane 若用于物理折叠，另设 `output_subtile_id`；
- 物理折叠必须重新计算 tile 次数、bias 尾相、weight 带宽和端到端帧率；
- 未经同库 DC，不宣称 32-lane 的面积或频率可接受。

## 五、descriptor 跨 tile 驻留结果

集成测试使用 2 个 head、3 个输出 tile：

```text
fill head0 once
fill head1 once
tile0: lookup head0/head1, no release
tile1: lookup head0/head1, no release
tile2: lookup head0/head1, release after each last use
```

结果为 6/6 cache hit、0 miss、15 个 descriptor entry 读取、2 次末 tile cache release、2 次末 tile slot release。前两个 tile 的 release 为 0。

这个结果证明了资源生命周期，但还没有证明节能。论文必须继续统计以下物理数据移动：

- no-residency 每 tile 重建 descriptor 的 slot/cache 读写字节；
- Depth64/80 residency 的 fill、lookup、SRAM 活动和 bypass；
- head-major 数据流的 partial-sum spill 字节；
- output-tile-stationary 数据流的 weight 读取、AccTile 更新和 final 写出字节。

## 六、PLAN/COMMIT 单一所有者

现有 `gatestack_replay_launch_control` 会分别启动 slot replay、decoder 和 route；现有 multihead projection top 又会原子启动 route 与 backend。二者不能直接级联。

最终控制边界拆成：

### PLAN 阶段

PLAN 只读取并冻结元数据，不启动任何数据流：

```text
context_id
head_id
payload_tag
route = Resident / IPD32W / RAW41
replay_start_word
term_count
event_count
cache_owned
slot_replay_required
```

### COMMIT 阶段

COMMIT 是唯一 transaction owner。各资源先给出不依赖 commit pulse 的 `reserve_ready`，仅当下列资源同时可预约时才广播同拍 commit pulse：

```text
decoder/route frontend
shared projection backend
slot replay port（若需要）
lifecycle session
```

任何一个未 ready 时，所有 commit pulse 均保持为 0。该两相预约/提交协议避免用其他端 ready 门控标准 ready/valid 的 valid，从而避免 valid 抖动和组合 ready-valid 环。旧 launch control 后续保留为叶级回归对象，但不直接进入 full top。

本轮已实现 `gatestack_replay_atomic_commit.sv`：

| 用例 | projection | slot | lifecycle | 结果 |
|---|---:|---:|---:|---:|
| Resident 有 payload | 同拍 | 同拍 | 同拍 | PASS |
| Resident 空 payload | 同拍 | 不请求 | 同拍 | PASS |
| 非法 RAW 无 slot | 不请求 | 不请求 | 不请求 | admission reject PASS |

三类资源分别不可预约时均未发生部分提交。模块通过 Icarus、Verilator+SVA、Yosys 与 Erie 独立 lint，Verilator/Erie 均为 0 warning/error，通用综合为 48 cells。execution tag 在 commit 握手时锁存，避免 PLAN 源紧接着变化导致完成脉冲与 tag 错配。

尚未实现的是 PLAN builder 以及 atomic COMMIT 到实际 multihead decoder projection 的接口适配。

## 七、必须拆分 payload tag 与 execution tag

跨 tile 驻留后，同一 head payload 在所有 tile 之间不变，但每次 replay 的执行实例不同：

```text
payload_tag   = {window, context, head, build_epoch}
execution_tag = {window, context, output_tile, head, exec_epoch}
```

- slot/cache/decoder 输入与 decoder done 校验 `payload_tag`；
- weight request/response、TDR、accumulator、head done、tile done 和 final 校验 `execution_tag`；
- 当前 RTL 仍复用单一 tag，因此本轮只签核循环和 residency，不签核最终 full top tag 语义。

## 八、分层生命周期

最终 full top 至少存在四个不同释放边界：

| 资源 | 获取点 | 释放点 |
|---|---|---|
| decoder stream | head COMMIT | decoder done |
| projection backend | head COMMIT | backend done |
| slot/cache payload | window build commit | 该 head 最后一个 output tile 完成 |
| window context | window start | 最后 tile 的全部 final handshake 和全部 head release |

decoder done、backend done、tile done 和 context retire 不能合并成一个 done。异常路径还需要 context 级 abort/drain 和 slot/cache 清理 sweep。

## 九、DATE 架构贡献边界

可作为主贡献候选：

1. **Head-stacked output-tile-stationary replay**：用一个驻留 AccTile 顺序吸收所有 input head，消除跨 head partial-sum spill；
2. **Build-once replay-many descriptor residency**：利用 H67 descriptor 的有界 term 深度，在多个输出 tile 之间复用同一 compact representation；
3. **Representation-heterogeneous, compute-unified pipeline**：Resident/IPD32W/RAW41 三种无损表示共享 TDR、多播和 accumulator；
4. **Last-use hierarchical lifetime**：按 decoder、backend、payload 和 context 四级最后使用点释放资源，并支持双 context build/execute 重叠。

不能单列为贡献：

- output tile 计数器；
- `last_tile` 标志；
- ready/valid FSM；
- release pending bit；
- 参数从 8 改成 32；
- Yosys 单元数。

## 十、下一步 RTL 淘汰门槛

1. 完成 payload/execution 双 tag，并在故意 tag 错配时 admission 前拒绝；
2. 实现 PLAN builder，并将已验证的原子 COMMIT 接入实际三个 decoder 和 projection；
3. bias、final、tile done 增加 output tile 标识；
4. 将 scheduler、slot、cache、PLAN/COMMIT、三个 decoder、共享 TDR 和 lifecycle 接成单 context full top；
5. 再扩展双 context，统计端口冲突、FIFO 深度和 ordered trace 周期；
6. ordered trace 相对 no-residency/head-major 基线吞吐至少 `1.20x`；
7. 同目标库、含 SRAM 与 SAIF 的 EDP 至少改善 15%，否则 residency 只保留为工程优化，不作为主贡献。

## 十一、复现入口

```bash
sim_hitflow/run_gatestack_output_tile_scheduler_checks.sh
sim_hitflow/run_gatestack_output_tile_residency_checks.sh
sim_hitflow/run_gatestack_replay_atomic_commit_checks.sh
python3 scripts/sweep_gatestack_output_tile.py
```

详细结果：

- `results/gatestack_output_tile_residency_20260716/report.md`
- `results/gatestack_output_tile_dse_20260716/output_tile_dse.md`
