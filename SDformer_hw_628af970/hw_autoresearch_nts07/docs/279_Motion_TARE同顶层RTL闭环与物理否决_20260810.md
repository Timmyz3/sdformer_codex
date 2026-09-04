# Motion TARE 同顶层 RTL 闭环与物理否决

> 日期：2026-08-10  
> 前序准入：`docs/278_Motion_TARE与TTB8_ZKQI独立增量去重筛选_20260810.md`。  
> 本轮唯一问题：TARE-W8/W16 在完整 TTB8-ZKQI row-top 强基线上是否仍有可发表的硬件收益。  
> 最终结论：`REJECT_TARE`。

## 1. 结论

本轮完成了此前评审要求的 A/B/C 同顶层筛选：

```text
A: TTB8-ZKQI + two parallel Direct32
B: TTB8-ZKQI + TARE-W8
C: TTB8-ZKQI + TARE-W16
```

三者共享 row store、TTB8 scanner、ZKQI 三类精确注入、RQTB/SCS directory、
Shiftmax/gated-K backend；只替换 active-score 前端。

结果：

1. `[rtl]` W8/W16 的叶级定向向量保持 raw/Q7 语义，真实 138 行最终 gated-K
   输出 bit-exact；`[待验证]` 尚未逐 pair 对比真实 138 行的 raw16/Q7；
2. `[rtl]` W8 最大周期回退 `3.9667%`，违反 1% 门槛，先行否决；
3. `[rtl]` W16 最大周期回退 `0.2522%`，通过周期门槛；
4. `[开放映射代理]` W16 相对 Direct32x2 面积增加 `51.82%`；
5. `[开放映射代理]` W16 面积归一吞吐只有 `0.6571x`，远低于预先冻结的
   `1.10x` 门槛；
6. **按事前规则，TARE-W8/W16 均 REJECT，不进入 Motion 的 DATE 贡献列表。**

该结论只否定当前 Direct32x2 强基线下的 TARE-W8/W16 任意-lane compactor
RTL，不否定 TARE 的代数恒等式、所有无 compactor 变体，也不声称 ASIC
功耗/EDP 已被证明更差。

## 2. 完成的 RTL

### 2.1 原子双 score executor

新增：`rtl_h67/h67_tare_score_pair.sv`。

接口以一个 temporal pair 为原子事务：

```text
input : {pair_id, Q0, K0, Q1, K1}
output: {pair_id, score0_q7, score1_q7, K-active-mask,
         update-count, dense-fallback}
```

关键语义：

- 一个 32-lane alpha-XNOR engine 先算 T0 anchor；
- update-count 不超过 W 时，由 W 个选中 lane 精确恢复 T1 residual；
- update-count 大于 W 时，下一拍复用同一 32-lane engine exact replay T1；
- T0/T1 同一个 packet 原子提交到 RQTB/SCS directory；
- RNE 只在每个最终 raw16 上执行一次；
- residual 使用 signed 13 bit，覆盖 W16 的 `[-1024,+1024]`；
- sparse/zero 使用组合 fall-through，只有 dense packet 占 replay 槽。

### 2.2 row-top 集成

`rtl_h67/h67_zkqi_row_shiftmax_top.sv` 新增参数：

```text
ACTIVE_SCORE_RESIDUAL_W = 0  -> Direct32x2
ACTIVE_SCORE_RESIDUAL_W = 8  -> TARE-W8
ACTIVE_SCORE_RESIDUAL_W = 16 -> TARE-W16
```

row store 和下游 directory 接口没有改变。新增性能计数器
`perf_tare_dense_fallbacks`，逐 row 与 TB 重新计算的 update mask 对照。

## 3. 验证中发现并修复的问题

### 3.1 每 row drain bubble

第一版把所有 TARE packet 都寄存一级。138 行无反压 W16 周期为 `102,042`，
比模型多 84 拍；84 正好等于含 active pair 的 head-row 数。

修复：sparse/zero 组合 fall-through，只让 dense replay 入槽。修复后为
`101,955`，消除了每 row 一拍的 drain bubble。

### 3.2 valid 被 descriptor enable 门控

Icarus 最终输出检查没有发现问题，但 Verilator/SVA 在 stall mode 1/3 报告：
上游事务尚未接受时，`in_valid` 可能随 `descriptor_issue_enable` 撤销。

修复：

- `in_valid` 只表达上游 payload 存在；
- 新增 `in_enable` 作为 executor 的显式准入条件；
- `in_enable=0` 时强制 `in_ready=0`；
- dense packet 也不能在 enable=0 时偷收；
- SVA 锁定 input/output stall payload 稳定。

修复后 W16 四种模式 Verilator+SVA 全部 PASS，且周期与 Icarus 一致。

## 4. 边界覆盖

叶级 TB：`tb_h67/tb_h67_tare_score_pair.sv`。

| 覆盖项 | 结果 |
|---|---|
| update-count 0..32 | PASS |
| W8 8/9 分界 | PASS |
| W16 16/17 分界 | PASS |
| W16 delta +1024 | PASS |
| W16 delta -1024 | PASS |
| 随机 output backpressure | PASS |
| Icarus | PASS |
| Verilator+SVA | PASS |

真实 138 行中：

- W8 每种 stall mode 精确执行 `3,321` 次 dense replay；
- W16 每种 stall mode 精确执行 `251` 次 dense replay；
- replay 数与 Python raw-lane profile 完全一致；
- 最终 gated-K 输出零丢失、零重复、零 gate mismatch、零协议错误。

## 5. 同顶层周期

| stall mode | Direct32x2 | TARE-W8 | W8回退 | TARE-W16 | W16回退 |
|---:|---:|---:|---:|---:|---:|
| 0 | 101,707 | 105,020 | 3.2574% | 101,955 | 0.2438% |
| 1 | 109,692 | 112,747 | 2.7851% | 109,908 | 0.1969% |
| 2 | 111,091 | 114,408 | 2.9858% | 111,341 | 0.2250% |
| 3 | 167,317 | 173,954 | 3.9667% | 167,739 | 0.2522% |

解释：W16 的 251 次 fallback 并非全部暴露到端到端关键路径；无反压时只增加
248 拍，少量 replay 被既有扫描/后端空隙隐藏。W8 的 fallback 太多，四种模式
均违反 1% 周期回退上限。

## 6. 开放面积筛选

映射条件：

- 同一完整 row-top；
- 同一 RTL 源集合与参数边界；
- Yosys `abc -fast`；
- Nangate45 typical liberty；
- `memory -nomap`，两边相同 SRAM 行为数组均不计面积；
- 不含 DC、STA、SAIF、PTPX，不称 ASIC PPA。

| 候选 | cells | area | 无反压吞吐比 | 面积归一吞吐 |
|---|---:|---:|---:|---:|
| Direct32x2 | 20,380 | 24,058.370 | 1.0000x | 1.0000x |
| TARE-W16 | 33,802 | 36,524.726 | 0.9976x | 0.6571x |

W16 面积比强基线大 `51.82%`。默认 ABC 对 Direct32x2 约两分钟完成；W16
运行超过 12 分钟仍停在约 30k-gate 组合网络映射，本轮手动取消。该时长只作
复杂度诊断，定量否决仍使用两边同流的 `abc -fast` 面积。

## 7. 为什么 lane-work 模型失效

profile 的 lane-only 模型把 W16 近似为：

```text
32-lane anchor + 16-lane residual = 48 lane
```

当前 RTL 把 32-bit 任意 update mask 压成最多 16 个 lane-id，并从
Q0/K0/Q1/K1 中按 lane-id 选择操作数：

```text
32-bit update mask
       |
       v
16-way priority extraction
       |
       v
16 x (5-bit lane-id + four 32:1 operand selects)
       |
       v
signed residual accumulation
```

alpha-XNOR 单 lane 本身只是在 `{64,1,0}` 中选择。结合 RTL 结构推断，任意
lane compactor/mux 是候选面积变大的主要来源；但 `[开放映射代理]` 只证明完整
候选逻辑变大，尚无层次面积或 compactor-off RTL 消融，因此根因仍标为
`[待验证]`。可以确定的是，`40.8631%` score-lane work 降低没有转化成当前
实现的面积收益。

## 8. 无 compactor 替代的快速否决

为避免误判成“只需优化 priority encoder”，本轮又用同一 raw trace 检查固定
上下 16-lane 半片：

- active pair：14,554；
- update=0：27；
- 只命中低半片：1,713；
- 只命中高半片：2,017；
- 同时命中两半片：10,797，即 `74.1858%`；
- 一个 16-lane fixed-slice residual 需对双半片项追加一拍；
- lane-only 吞吐约 `0.5741x`；
- lane-only 面积归一吞吐约 `0.7655x`。

固定半片虽然删除了任意 lane compactor，却因 update 位置分散而在模型中失败。
因此 fixed-slice 候选没有通过进入 RTL 的准入门槛；这不是对所有无 compactor
TARE 变体的广义否定。

## 9. DATE 贡献边界

不能声称：

- TARE 是 Motion 的硬件创新点；
- lane-work 减少等于面积/能耗减少；
- `abc -fast` 是 ASIC PPA；
- W16 相对单 Direct32 的旧加速仍适用于当前强基线。

可以声称：

- `[rtl]` ZKQI 与 active-lane residual 是两个语义正交、bit-exact 的执行层；
- `[负结果]` 在 H67 fullres T450 上，任意 lane compaction 的控制/选择代价超过
  arithmetic reuse 收益；
- `[方法]` workload profile、强基线、同顶层 miter 和开放面积门槛成功阻止了
  一个仅在 lane-work 模型上看似优秀的机制进入论文贡献。

负结果提高论文可信度，但**不增加创新点数量**。

## 10. 对 Motion 主线的影响

Motion 保留：

1. TTB8-ZKQI 的 token-time metadata 分层扫描；
2. three-class exact zero-K 注入；
3. RQTB temporal score 合并；
4. SCS/Shiftmax 与 gated-K backend；
5. 已有同宏/开放物理代理与跨 100 sample ordered profile。

Motion 删除或降级：

- TARE-W8/W16：降为负基线；
- fixed-slice TARE：模型否决，不写 RTL；
- “TARE 带来面积/能耗优势”：删除。

下一次 Motion 新机制必须先回答“是否避免任意 lane compaction”，并在 profile
阶段同时计入选择网络或固定拓扑服务周期。否则不准进入 RTL。

## 11. 产物与复现

- RTL：`rtl_h67/h67_tare_score_pair.sv`；
- row-top：`rtl_h67/h67_zkqi_row_shiftmax_top.sv`；
- SVA：`verif_h67/h67_tare_score_pair_assertions.sv`；
- 叶 TB：`tb_h67/tb_h67_tare_score_pair.sv`；
- row miter：`tb_h67/tb_h67_zkqi_row_miter.sv`；
- 汇总脚本：`scripts/report_h67_tare_zkqi_row_rtl.py`；
- 单测：`tests/test_report_h67_tare_zkqi_row_rtl.py`；
- runner：`sim_h67/run_h67_tare_zkqi_row_checks.sh`；
- 结果：`results/h67_tare_zkqi_row_rtl_20260810/`。

```bash
./sim_h67/run_h67_tare_zkqi_row_checks.sh
```

自动化末行：

```text
PASS REJECT_TARE
PASS Motion TARE/ZKQI row RTL screening: REJECT_TARE
```

## 12. 独立 DATE 评审

评审结论：`Weak Reject / Major Revision`，无 P0；本包 `3.6/5`、Motion 整体
`3.3/5`、架构创新性 `3.0/5`、DATE 就绪度 `2.9/5`。

评审确认：

1. Direct32x2 是同 row-top 强基线，没有偷换边界；
2. 当前 `REJECT_TARE` 成立，但范围必须限定为 W8/W16 任意-lane compactor RTL；
3. `memory -nomap`/`abc -fast` 足以作当前实现的 fail-fast 淘汰代理，不是 PPA；
4. 负结果增加方法可信度，不增加 DATE 创新点；
5. 下一轮唯一优先级应转为 Local5 legal-1RW TCFM5。

本轮已整改两项 P1：

- 将真实 138 行 raw16/Q7 从 `[rtl]` 收窄为 `[待验证]`，只保留最终 gated-K
  bit-exact 声明；
- 报告器改为解析叶级 PASS，并绑定全部日志、向量、liberty、RTL/TB/SVA、
  runner 哈希与工具版本；W16 决策同时检查周期和面积门槛。

保留的 P2 边界：反压仍按 wall-clock 而非 transaction-indexed；没有 Acc32、
SAIF/EDP 或 compactor 层次面积消融。由于无反压和开放面积已足以否决当前实现，
这些不再为 TARE 补做，避免把资源投入已失败候选。
