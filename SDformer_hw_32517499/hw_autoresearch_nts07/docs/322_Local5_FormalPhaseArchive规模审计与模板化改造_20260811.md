# Local5 Formal Phase Archive 规模审计与模板化改造

## 1. 本轮裁决

正式 profile100 已经具备，但旧 v4 phase archive 不能直接启动全量生成：

```text
v4 full expansion = DENY_FULL_RUN
formal G0 = DENY
下一准入项 = 单窗口 head-template + tile-reference canary
```

本容量审计只证明旧表示把同一 input-head 的 direct/execute 结构事件骨架按
output tile 重复保存，造成不可接受的 archive 和重放内存规模；它不假设各 tile
的 identity/service-cycle patch 相同，也不对 Local5 算法或 RTL 数值正确性作判断。

可复跑入口：

```bash
python3 scripts/audit_local5_formal_phase_archive_scale.py \
  --output-dir results/local5_formal_phase_archive_scale_audit_v3_20260811
```

结果：

- `results/local5_formal_phase_archive_scale_audit_v3_20260811/phase_archive_scale_audit.json`
- `results/local5_formal_phase_archive_scale_audit_v3_20260811/phase_archive_scale_audit.md`

## 2. 正式 profile100 规模

输入逐 SHA 绑定：

- `ordered_term_manifest.json`：
  `db92881db34b62cfd0bf62eccddbf4e860c670076814bb8483f7a7b219874f52`
- `ordered_term_items.npz`：
  `05a4c4155c6e439760caeb0a5ed636d7a989d02c9be741ab0e159303d987f533`

| 指标 | 数值 | 证据 |
|---|---:|---|
| joint window | 1,200 | `[prof]` |
| input-head group | 13,800 | `[prof]` |
| phase | 462,600 | 冻结 schema |
| destination unique item | 22,848,620 | `[prof]` |
| source product term | 9,870,505 | `[prof]` |
| multiplicity 展开 delivery/update | 29,164,959 | `[prof]` |
| active epoch record | 1,634,217 | `[prof]`，`source_term_count>0` |
| 按 output tile 展开 product term | 185,785,962 | `[prof]+[模型]` |
| 按 output tile 展开 delivery/update | 542,125,785 | `[prof]+[模型]` |
| 按 output tile 展开 record | 29,223,267 | `[prof]+[模型]` |

旧 v4 最少事件数为：

```text
fill:               2 * active_record                  =     3,268,434
execute metadata:   3 * active_record * H              =    87,669,801
direct+execute Acc: 2 * multiplicity_delivery * H      = 1,084,251,570
---------------------------------------------------------------------
main expanded events（不含 prepare/drain）             = 1,175,189,805
```

这里的 Acc 事件必须按 multiplicity 展开的 delivery/update 计数，不能用较小的
destination unique item 代替。该计数还没有加入 prepare/drain、额外调试 snapshot、
状态事件或 Python 对象开销。

## 3. 旧 v4 为什么不可执行

当前 `local5_erep_archive_replay_v4.py` 每条事件固定保存：

```text
event_resource:uint8 + event_cycle:uint32 + event_identity:S64 = 69 byte
```

加 phase metadata 后，按“每个 prepare/drain phase 恰有一个语义事件”的场景，
未压缩数组约 `75.53 GiB`；若 drain 对每个 source 做一次 450 元素 vector read，
约 `75.93 GiB`；若每个 source/channel 都形成 scalar read，则约 `88.30 GiB`。准确
prepare/drain 数量必须由 RTL canary 冻结；v4 schema 本身允许空事件，所以该场景不是
严格 schema minimum。NPZ 压缩可能减小磁盘文件，但 parser 必须把
数组解压到内存，不能以磁盘压缩率掩盖运行时内存。正式 Acc32 expected/actual 另有约
`1.48 GiB` raw payload。

因此，直接跑旧 v4 不是“证据更严格”，而是在已知表示冗余下浪费数十 GiB 内存，并且
很可能产生 ZIP64/临时副本问题。

## 4. 模板化 archive

### 4.1 精确不变量

对固定 `OUT_DIM=32` 的一个 input head：

1. relation/source/term 顺序不随 output tile 改变；
2. output tile 只选择另一组 `32x32` INT8 权重；
3. 权重数值改变 product value，但预期不改变 ready/valid、term 数或 Acc bank 地址；
4. 每个 tile 使用局部 out index `0..31`，但冻结 identity service 显式包含
   `output_tile`，因此 service-cycle/identity 可能随 tile 变化；
5. 因此只能共享**参数化模板**，并为每个 tile 保存 identity/service-cycle patch，
   不能假设静态事件流逐 tile 完全相同。

以上第 3、4 项仍需一个真实 `OUT_DIM32` 多 tile RTL canary 证明，当前标为
`[待验证]`，不能只靠代码注释认定。

### 4.2 两级方案

**方案 A：参数化 head-template + tile patch。**

- 每 input head 保存一份 fill/direct/execute 模板；
- 每 output tile 保存模板引用及 identity/service-cycle patch；
- identity 从 `S64` 改为 `uint32` 结构化索引；
- base-template 事件数为 `66,501,003`，base event 复用因子 `17.67x`；
- 每 common phase 一个语义事件时，base-template 为 `0.57 GiB`；按 450-vector
  drain 为 `0.62 GiB`，按 `450x32` scalar drain 为 `2.23 GiB`；
- 上述数字均**不含 tile patch**，不能作为完整 archive 容量或端到端存储缩减。
- 若全部主事件都要保存 `uint32 cycle` patch，单 cycle 字段约 `4.38 GiB`；若 cycle
  和 identity 各一个 `uint32`，约 `8.76 GiB`，还未计 patch 索引和 offset。

**方案 B：term/source 对齐 cycle archive。**

- ordered-term/source 顺序本身就是 identity；
- term 只保存 direct/execute cycle；
- active source 只保存 relation-read、epoch-write/read、FIFO enq/deq cycle；
- identity 由正式 offsets 和索引重建；
- 排除尚未冻结的 common phase 和 tile patch，base 模型约 `0.26 GiB`。

方案 B 更小，但 parser 与 identity 重建的证明义务更高。先做方案 A 单窗口 canary，
通过后再决定是否晋级方案 B。

## 5. 不允许混淆的证据

模板去重只作用于 **phase/event archive**。以下内容不能去重或推断：

1. 每个 output tile 的真实 INT8 权重；
2. 每个 output tile 的 Acc32 expected/actual；
3. 不同 input head 的 term/source 顺序；
4. 不同 sample、stage、block、window 的 phase duration；
5. Direct 实测事件与 EREP epoch-slot/FIFO 模型事件的来源标签。

特别是第 5 项：当前 EREP C1--C4 是 `[rtl校准]+[模型]`，并非候选 RTL。新 archive
必须显式区分 `rtl_observed`、`rtl_derived` 和 `schedule_model`，不能把模型事件统称为
“RTL raw trace”。

## 6. 下一步最小验证

1. 冻结一个 H3、`OUT_DIM32`、真实 checkpoint 权重的 sample/window；
2. 对同一 input head 至少回放两个 output tile；
3. 比较 phase duration、resource、cycle、identity、事件来源和 SHA；
4. 证明参数化 template+tile patch 展开与逐 tile 原始回放逐事件一致，且覆盖至少一个
   tile-dependent service-latency 情况；
5. 生成一窗 template archive，再由独立 parser 展开为旧 `WindowCommandWork`；
6. 同时校验 candidate cycle/resource count/event-ledger SHA、Acc32 以及 prepare/drain；
7. 通过独立 DATE 风格复审后，才估算 1,200 窗全量 wall time 和分片策略。

## 7. 独立 DATE 风格复审

本轮初版规模审计得到 `3/5`：没有 P0，但有两个 P1。复审指出初版把 unique item
误当成 Acc delivery/update，且静态 head-template 忽略了 `output_tile` 对 identity
service 的影响。v2 已修正这两项；该修正只提高容量模型可信度，不把 G0 状态升级为
PASS。

修正 v2 后再次复审仍为 `3/5`，裁决为
`CONDITIONAL_PASS_TO_SINGLE_WINDOW_CANARY`。该轮关闭了 multiplicity 计数和静态模板
问题，但新发现一个 P1：`0.57/0.62/2.23 GiB` 未包含 tile patch。v3 已完成以下整改：

1. 所有模板容量字段明确命名为 `base_template_*_excluding_tile_patch`；
2. 删除把 `17.67x` 写成完整存储缩减的口径，只保留 base event 复用因子；
3. 新增 dense cycle-only `4.38 GiB` 与 cycle+identity `8.76 GiB` 容量包络；
4. sparse patch 字节数和密度保持 `null`，等待真实 canary；
5. 绑定审计脚本、archive/ledger/schedule/capacity/identity-service 六项源码 SHA；
6. 单测从 2 项增至 5 项，显式检查 patch 没有被计入 base-template；随后按第三轮
   复审补为 6 项，增加正式 profile 提取与冻结 JSON/Markdown 精确重生成。

因此当前可进入单窗模板 canary，但仍不能用 base-template 数字做全量实现预算。

在该 canary 之前，不修改 formal admission 为 PASS，也不实现 EREP candidate RTL。

## 8. 2026-08-11 H3 canary 更新

v9 首轮 canary 被独立复审判为 3.5/5 Reject：它只是把 weight valid 延后两周期，
并未形成真实 `valid=1,ready=0`。该结果保留为负结果。

修复后的 `results/local5_h3_phase_template_patch_canary_v2_20260811` 已完成单窗 H3
参数化
template + typed tile patch：

- 862,507 行真实 v10 trace 逐行无损展开，byte-stream SHA 一致；
- 结构模板 206,078 行，`event/origin` 骨架复用因子 4.185x；
- 完整 archive 已计入 cycle/identity/payload patch，为 24,734,032 byte；相对原始
  52,519,869-byte CSV 为 2.123x 文件缩减；
- 与 v8 的 cycle-free handshake/boundary/state ledger 一致；
- 与 v8 的 844,073 条 core-all cycle-free 全序 ledger 一致；
- 43,200 个 Acc32 与 v8、软件整数参考三方一致；
- 9,216 个 weight response 均实测 2-cycle held-valid，共 18,432 个
  `valid=1,ready=0` telemetry cycle；
- 九类定向 template/patch 篡改均被独立 expander 拒绝。

这关闭了本文件第 6 节的“单窗口 template+tile patch 逐事件同构”步骤，但尚未独立
展开为旧 `WindowCommandWork`，也未覆盖 H6/H12/H24 或 1,200 窗。formal G0 仍为
`DENY`。详细证据见 `docs/328_Local5_H3_PhaseTemplate与TypedTilePatch闭环_20260811.md`。
