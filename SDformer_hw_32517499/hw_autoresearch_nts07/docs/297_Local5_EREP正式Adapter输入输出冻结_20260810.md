# Local5 EREP 正式 Adapter 输入输出冻结

## 1. 当前裁决

本页只冻结正式 adapter 的输入、重组和输出合同，不实现 EREP 候选 RTL，也不
生成 admission PASS。截至 2026-08-10 后续检查，GPU producer 已完成 `25/100`，正式
`ordered_term_manifest.json` 尚不存在。因此：

```text
formal adapter = DENY
formal G0 = DENY
EREP candidate RTL = DENY
```

现有 `local5_erep_statistics_v4.py` 因 admission receipt 缺失而 fail closed，
该行为正确。

## 2. 正式输入集合

adapter 只允许读取固定目录：

```text
results/local5_fullres_bb1e4_joint_heads_profile100_20260809
```

必须逐 SHA 绑定：

1. `ordered_term_manifest.json` 与 `ordered_term_items.npz`；
2. `joint_window_selection_plan.json`；
3. `ordered_cohort.json`；
4. `joint_head_run_identity.json`；
5. `gpu_exclusivity_audit.json`；
6. checkpoint projection contract JSON/NPZ；
7. producer、profiler、attention、model、dataset 与量化脚本源文件。

任一文件缺失、SHA 不符、qualification 非真、GPU audit 非 PASS 或出现外来 GPU
进程，adapter 不得写任何 admission artifact。

## 3. Payload 形状合同

正式 manifest 必须有 `13,800` 个 group，来源为 `100 sample x 12 block x all
heads`。stage 的 head 数固定为 `3/6/12/24`，block 数固定为 `2/2/6/2`。

关键数组必须满足：

| 数组 | dtype | 正式形状/长度 |
|---|---|---|
| `group_offsets` | int64 | `13,801` |
| `group_tags` | uint64 | `13,800` |
| `descriptor_group_offsets` | int64 | `13,801` |
| `descriptor_source_id/plane/y/x` | uint16/uint8/uint16/uint16 | `6,210,000` |
| `descriptor_q_bitmap/k_bitmap` | uint64 | `6,210,000` |
| `descriptor_incoming_gates` | uint16 | `6,210,000 x 5` |
| `descriptor_valid_mask` | uint8 | `6,210,000` |
| `source_group_offsets` | int64 | `13,801` |
| source/destination frontier arrays | 固定整数 dtype | 每 group 恰好 450 source/destination |
| item term arrays | manifest 固定 dtype | 长度等于 `group_offsets[-1]` |

每个 group 的 descriptor source id 必须严格为 `0..449`，坐标、plane、K、gate、
valid mask 和 source-major term 重建必须逐项一致。不能只相信 NPZ dtype 或长度。

## 4. Canonical joint-window 重组

adapter 不依赖 manifest 行顺序，而按以下唯一 key 建表并拒绝重复/缺失：

```text
(sample, stage, block, selected_window, input_head)
```

每个 `(sample,stage,block)` 必须恰有一个预注册 window 和全部 `head=0..H-1`，
得到 `100 x 12 = 1,200` 个 joint window。窗口顺序固定为 sample-major，再按
`(stage,block)` 的 `[(0,0),(0,1),(1,0),(1,1),(2,0)..(2,5),(3,0),(3,1)]`。

### 4.1 不能漏掉的 HxH 工作

一个 stage 有 `H` 个输入 head，projection 输出通道为 `H*32`，即有 `H` 个
`OUT_DIM32` 输出 tile。因此一个 joint window 的精确投影工作是：

```text
H input heads x H output tiles
```

正式 adapter 不得把一个输入 head 只回放到一个输出 tile，也不得把 13,800 个
input-head group 当成完整 projection 工作。projection contract 中第 `o` 个输出
tile 必须使用矩阵行 `[32*o,32*(o+1))`，第 `h` 个输入 head 使用列
`[32*h,32*(h+1))`。

## 5. 相序与候选模型

Direct 校准的五段为：

```text
prepare -> relation_fill -> relation_commit -> execute -> compute_drain
```

当前 T450/OUT_DIM2 校准只证明 TB 能从 RTL 接口观测边界；不是 OUT_DIM32
性能证据。正式 adapter 必须对 OUT_DIM32 重新校准：

1. 每个 input-head relation/term execute 的真实 cycle 和命令账本；
2. 每个 output tile 的 context prepare；
3. 全部 input head 累加完成后的公共 Acc32 vector drain/serializer；
4. epoch record capture/replay 的 1RW 写、seal、读、FIFO 与 execute 资源冲突。

C0--C3 周期由 `local5_erep_command_schedule_v4.py` 对上述 RTL 校准 phase/command
账本重排得到，证据只能标为 `[rtl校准]+[模型]`。C4 在 G0 固定为 relaxed oracle
`C4O_G0`；G2 的可实现无条件 first-fit 为 `C4I_G2`，两者禁止混写。

## 6. 防止聚合自报的输出合同

正式 adapter 不能只写 1200 行 `c0..c4` scalar。最低输出应为：

1. `head_phase_ledger.json`：13,800 个 input-head 的原始 RTL phase、命令计数与
   trace SHA；每个 window 另存最终 Acc32 miter 摘要与 mismatch count；
2. `window_schedule_ledger.json`：1,200 个窗口的 HxH task、C0--C4 event/resource
   schedule、每候选 tail cycle 和冲突审计；
3. `command_ledger.json`：统计器消费的 1,200 行摘要，每行必须携带其
   `window_schedule_ledger` canonical SHA；
4. `admitted_rows.json`：只复制经统计器重算后的 C0--C4，不接受 adapter 自填
   且无底层账本的 scalar；
5. `admission_receipt.json`：绑定全部输入、adapter、runtime、底层 ledger 和输出
   SHA。

统计器必须从底层 phase/event ledger 重新执行 schedule 并重算 C0--C4，再与摘要
比较；当前只比较两个 JSON 中相同 scalar/digest 的做法不足，正式 G0 前必须
修改。

## 7. Acc32 数值闭环

正式 numeric miter 的边界为每个 `(window,output_tile,source,out)` 的 Acc32：

- 输入 head 累加顺序固定为 `0..H-1`；
- 输出 tile 顺序可按候选调度变化，但每 tile 内 head 顺序不得变化；
- C0--C4 必须消费同一 relation、term、weight 和 bias；
- 不允许剪枝、近似、溢出策略差异或候选专属 workload；
- mismatch 必须为 0；adapter 必须先按固定
  `output_tile -> source -> out` 坐标重排 expected/actual，再写 canonical archive。
  不使用单独 multiset digest，因为 multiset 会掩盖坐标交换；固定坐标字节摘要同时
  绑定数值与位置。

## 8. 启动条件

只有以下条件同时满足，才允许实现并运行正式 adapter：

1. producer 完成 100/100 并生成 manifest/payload；
2. qualification 为真且 13,800 group 全覆盖；
3. 1,200-window canonical key 集合精确；
4. projection contract 完整覆盖 12 block；
5. GPU audit、cohort、identity 和 selection plan 全部 SHA 绑定；
6. OUT_DIM32 phase、common drain 和 Acc32 miter 设施已通过独立复审。

在此之前，不创建 EREP candidate wrapper、filelist 或 RTL。

## 9. HxH 旁路 Preflight 进展

不修改运行中 producer 的前提下，已新增只读旁路检查器：

```text
scripts/local5_erep_formal_preflight_v4.py
```

它已机器化执行：

1. selection plan 必须恰有 1200 个 canonical window；
2. 1200 window 必须展开为 13800 个唯一 input-head key；
3. projection contract 必须覆盖 12 block，每个权重矩阵严格为
   `[H*32,H*32]`，NPZ 六类数组 shape/dtype 全部精确；
4. 每个窗口枚举 `H input head x H output tile`，全队列恰为 210600 个唯一任务；
5. 正式 manifest 到达后，group 行顺序可变，但 key multiset 必须精确等于上述
   13800 项，缺失、重复或非法 head 一律失败。

当前 preflight 因 manifest 缺失返回 `DENY_FORMAL_MANIFEST_ABSENT`，并明确
`admission_generated=false`。这只关闭 HxH 拓扑合同，不关闭第 6 节的底层 ledger
重放和防聚合自报 P0。

## 10. Anti-self-report 旁路实现进展

在正式 producer 尚未完成、禁止生成 admission 的阶段，已实现独立旁路重放器：

```text
scripts/local5_erep_ledger_replay_v4.py
```

统计器不再允许只比较 `admitted_rows.c0..c4` 与同源
`command_ledger.c0..c4`。正式 receipt 新增强制绑定：

```text
head_phase_ledger.json
window_schedule_ledger.json
command_ledger.json
```

head ledger 只允许保存 phase duration、逐资源相对 cycle、RTL trace SHA、Acc32
miter SHA 与 mismatch count，schema 中不存在 C0--C4。独立重放器从这些底层字段重建
`WindowCommandWork`，执行冻结 C0--C3 调度和 C4 relaxed oracle，再逐层生成 window
schedule 与 command digest。即使同步修改两份上层 scalar 并重算其 digest，也会与
phase 重放结果不同而失败。

当前该设施只有 `[synthetic-contract]+[代码审计]` 证据；正式 1200-window/
13800-head ledger 尚不存在，formal G0 仍为 DENY。详细结果和边界见
`docs/306_Local5_EREP防聚合自报三层账本_20260810.md`。

## 11. Archive 内容重放进展

已新增 `local5_erep_archive_replay_v4.py`，冻结并独立解析 trace/miter NPZ 的
成员、dtype、shape、canonical phase 次序、raw event 与 Acc32 内容。正式规模固定为
1200 window、13800 head、462600 phase 和 198720000 个 Acc32 标量；任一不符即拒绝。
统计器已在生成 C0--C4 之前调用该 parser。当前 46 项联合测试通过，但使用的仍是
synthetic fixture；producer manifest、正式 adapter、正式 RTL trace/miter 和 formal G0
仍未到达。详细合同见
`docs/310_Local5_EREP正式Archive内容重放合同_20260810.md`。
