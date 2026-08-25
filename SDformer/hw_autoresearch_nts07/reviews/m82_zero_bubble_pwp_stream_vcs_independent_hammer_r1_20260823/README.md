# M82 Zero-Bubble PWP Stream 独立打铁 R1

日期：2026-08-23  
角色：独立审阅；不以生产 PASS 行作为结论前提  
范围：M82 RTL/SVA/TB/filelist/contract、生产 sealed run、M79 反证基线，以及 M81 bank DSE 边界  
总判定：**M82 隔离 zero-bubble stream GO，并应替代 M79 command-separated frontend；M81 bank 集成、真实 fallback、M78 shared32 性能重新准入和 DATE/PPA headline 仍 NO-GO。**

## 1. 独立复核结果

六项生产输入 SHA 与合同完全一致；原 runner 已在新的隔离 `RUN_DIR` 复跑，VCS/SVA 再次得到：

```text
PASS M82 zero-bubble regular=129 escapes=8 starts=139 ii_checks=135 stalls=1 lanes=96 protocol_attacks=3 service=3,4,4,5
```

独立 TB 不复用生产 TB 的 start counter，使用混合序列：

```text
8 -> 11 -> 9 -> 10 -> escape -> escape -> 8 -> escape -> 11 -> 10 -> 9 -> 8
```

独立 VCS 结果：

```text
M82_INDEPENDENT_II previous_width=8  observed_cycles=3
M82_INDEPENDENT_II previous_width=11 observed_cycles=5
M82_INDEPENDENT_II previous_width=9  observed_cycles=4
M82_INDEPENDENT_II previous_width=10 observed_cycles=4
M82_INDEPENDENT_II previous_width=12 observed_cycles=1
PASS M82 independent hammer normal=11 escapes=3 mixed_ii=11 stall_cycles=3 attacks=8 signed_extremes=8,9,10,11 service=3,4,4,5 escape_service=1
```

独立 TB 还验证了：

- 8/9/10/11-bit 每种宽度的 signed 最小值、最大值、`-1`、`0` 和 96-lane 重建；
- 相邻 escape、escape→regular、regular→escape 均无额外空拍；
- output stall 三周期时，下一首 beat 保持且不被接受；释放后前一 output retire 与下一 start 同拍发生；
- start-mid-transaction、premature last、missing final last、9/10/11 padding 非零、escape 缺 last、escape data 非零共 8 类攻击全部 fail closed；生产 SVA 的 protocol-fault cover 在独立 run 中命中 8 次。

机器可读结果见 `m82_independent_hammer.json`。

注：`independent_mixed_vcs`、`_r2`、`_r3` 是独立 TB 编写过程中的编译/scoreboard 诊断目录，不作为证据引用；最终可引用 run 仅为 `independent_mixed_vcs_r4`，其日志已进入 `review_artifact_sha256.json`。

## 2. M82 是否真的消除了 M79 的额外 1 cycle？

**是，针对当前 always-ready 隔离 stream 边界，已经由独立 VCS 证实。**

| 位宽 | payload | M79 最小 command II | M82 最小 start II | 改善 |
|---:|---:|---:|---:|---:|
| 8 | 96 B / 3 beat | 4 | 3 | -1 cycle |
| 9 | 108 B / 4 beat | 5 | 4 | -1 cycle |
| 10 | 120 B / 4 beat | 5 | 4 | -1 cycle |
| 11 | 132 B / 5 beat | 6 | 5 | -1 cycle |

原因也与 RTL 一致：M82 将 descriptor 携带在首 payload beat 上，并令 `beat_ready = !faulted && (!output_valid || output_ready)`；前一 output 可在下一事务首 beat 被接受的同一时钟沿退休。M79 的独立打铁中，独立 command 阶段造成的 `beat+1` II 已不再存在。

M79 的无隐藏敏感性曾把 Cap11/SHARED32 从 `1.4094065x` 拉到 `1.3115886x`，对应 58,969,374 个 regular PWP 各多 1 cycle。M82 在隔离 stream 接口上具备收回这 58,969,374 cycle 的能力，因此 M78 的 3/4/4/5 服务假设在**此接口边界**恢复为可实现。

但这不等于 M78 的 `1.4094x` 已重新准入：M82 输入的是理想 `beat_*` 流，没有产生 M81 SRAM 地址，也没有证明同步宏响应、barrel reorder、correction 仲裁和下游 backpressure 下仍能持续提供该流。

另一个小账必须保留：M79 width12 是 0 beat token，M82 为了统一 stream 把 escape 变为 1 control beat。M78 heldout 有 362 次 escape 使用；若 362 个 control cycle 全暴露，candidate 为 `790,689,547 cycles`，速度为 `1.4094059x`。数值影响几乎为零，但 integrated replay 中必须计费，不能隐去。

## 3. P0/P1

### Scoped P0

**0 个。** 在合同承认的 isolated stream 范围内，没有发现阻断性数值或协议错误；zero-bubble 修复成立。

### DATE/系统晋级 P0

1. **M81 bank 尚未接入。** M81 只证明 `8x32-bit` 独立 bank 的地址构造；75.83% packed beat 跨 base/base+1 row，需要八个独立 row address 和 barrel reorder。M82 没有 address/CE/read-response 接口，无法证明同步 SRAM latency 后仍每周期一 beat。
2. **真实 escape fallback 尚未接入。** M82 仅输出 `escape=1` 和 tag；没有 bit-sparse 权重地址、读取、correction/PWP 共享端口仲裁或 accumulator completion ordering。
3. **有限队列与真实 ordered replay 缺失。** M82 只有一个 1280-bit assembly buffer 和一个 1152-bit output register；output stall 会立即阻断下一事务，没有处理已经发往同步 SRAM 的 in-flight read。尚无冻结 8640-phase PWP/correction/362-escape 序列的 queue-depth sweep。
4. **宏/PPA/系统主张缺失。** 1280-bit buffer、1152-bit output、96-lane variable-width unpack、M81 cross-row mux/barrel 尚未做完整 DC/STA/Formality/SAIF/PTPX，也没有相同资源约束的 full-network A/B。

### P1

- zero-bubble II 目前由 procedural TB 检查；SVA 只有 `cp_zero_bubble_boundary` cover，没有“下一 start 必须在 3/4/4/5 cycle” assertion。
- SVA 没有输入 `valid && !ready` 时 descriptor/data 稳定、accepted beat count、continuation metadata、padding、start-tag→output-tag conservation、bounded liveness 等 property。
- 生产只有一个 output stall case；独立 run 补了三周期 hold/resume，但仍缺 randomized 长 backpressure 和同步 read-response skid 场景。
- 模块仅验证单 PWP stream；M81 明确未准入 multi-PWP concurrency。
- 名为 sealed 的生产目录仍为 `775`，文件为 `664/775`，没有输出 SHA manifest；runner SHA `c5676ef3...` 未被自身输入集合 pin。

## 4. 创新与性能评分

| 维度 | 分数 / 10 | 结论 |
|---|---:|---|
| 隔离数值/流功能 | 9.2 | 混合宽度、escape、极值和协议攻击均独立闭环 |
| zero-bubble 性能事实 | 9.0 | M79 的每事务 1-cycle 气泡被真实 RTL/VCS 消除 |
| 协议与有限资源完整度 | 7.2 | 单输出缓冲下正确；缺同步 bank in-flight 和随机 queue 压力 |
| 硬件创新性 | 7.0 | descriptor-first-beat + same-edge retire/start 是有效架构修复，但属于关键供数优化，不足以单独构成 DATE headline |
| 系统性能优势证据 | 6.0 | `1.4094x` 的一个主要 RTL 反证已关闭；bank/fallback/order 尚未关闭 |
| 宏/PPA/能效证据 | 2.0 | 尚无集成 top 和 Synopsys/macro 结果 |
| DATE 论文完整度 | 3.8 | isolated VCS 强，full-system/equal-resource/PPA 仍缺 |
| 综合里程碑 | 6.1 | 明确晋级，适合作为 M83 集成输入；不准 headline |

相比 M79 的 4.8/10，M82 的重要进步不是增加一个理论倍率，而是把已识别的 throughput contradiction 用 RTL 消掉。这类“发现反证—改架构—独立重测”的证据链是加分项。要达到 DATE 强稿，下一步必须把 M81 的容量创新和 M82 的流创新合成一个可综合、可回放的 bank-to-accumulator datapath。

## 5. GO/NO-GO 与下一门槛

GO：

- GO 保留 M82 作为新的 precision-elastic PWP frontend，停止用 M79 command-separated frontend 支撑性能模型。
- GO 进入 M83：M81 WORD_PACKED 8-bank + descriptor lookup + cross-row reorder + M82 stream 的集成 RTL。
- GO 在模块内部表述“always-ready isolated start II=3/4/4/5，独立 VCS 通过”。

NO-GO：

- NO-GO 宣称 M81 bank-integrated zero-bubble。
- NO-GO 宣称真实 362 escape fallback、finite-queue 或 accumulator ordering 已闭环。
- NO-GO 单靠 M82 重新准入 M78 `1.4094x` module/system speedup。
- NO-GO DATE/best-paper、PPA、能效或 full-network headline。

下一门槛必须同时包含：

1. 8 个独立同步 32-bit bank 的地址、base/base+1 row selector、barrel reorder 和 13-bit descriptor lookup RTL；
2. 至少 1-cycle SRAM response latency、in-flight request skid/credit、output randomized backpressure；
3. frozen 8640-phase ordered trace，混合 correction/PWP/362 escape，报告 II、stall、queue occupancy、bank utilization；
4. 464/512-row macro wrapper的 DC/STA/Formality/SAIF/PTPX；
5. Fixed12、bit-sparse、Phi-like fixed12、Cap11 M81+M82 在同频率/面积/带宽下的端到端 A/B。

当前最安全的论文表述是：**“M82 independently validates a zero-bubble elastic PWP stream at the isolated ready/valid boundary; bank-integrated throughput, fallback ordering, PPA, and system speedup remain unadmitted.”**
