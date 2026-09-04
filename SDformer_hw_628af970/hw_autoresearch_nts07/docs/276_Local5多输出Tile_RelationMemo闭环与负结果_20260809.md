# Local5 多输出 Tile Relation Memo 闭环与负结果

日期：2026-08-09

## 1. 本轮结论

本轮只推进 `docs/275` 指定的 Local5 最高优先级缺口：把 relation memo 接入真实
output-tile scheduler、三输入头 T450/OUT32 数值路径和跨头 Acc32 累加器，并与每个
output tile 都重新计算 relation 的基线做同条件对照。

- `[rtl]` relation memo 路径与 recompute 路径均完成三输入头、三输出 tile、T450、
  OUT32、随机 Token/weight 延迟和 final-result 反压下的整数零失配；
- `[rtl]` memo 将 Q/K Token 请求从 4050 降至 2250，减少 44.44%；
- `[rtl]` 三个服务 seed 的平均周期仅从 696609.0 降至 686677.3，平均加速
  1.0145x；
- `[rtl]` 四种 partial-stream 故障均 fail-closed，并修复了协议错误后子引擎仍可能
  泄漏 Token/weight 请求的服务隔离缺陷；
- `[rtl]` 修复 projection weight context 无法按 output tile 释放的问题，避免后续
  tile 使用陈旧权重；
- `[待验证]` full-resolution Local5 真实 trace 的 memo resident/hit 分布尚未拿到，
  本轮定向稀疏/稠密组合只验证机制，不代表部署收益。

最重要的结论是负结果：**relation memo 的 exact 复用成立，但当前标量 Acc32 接口使其
几乎不能转化为周期收益。** FCSR 不应作为 Local5 的独立主性能贡献；下一轮必须直接
减少 partial 交接与读回工作，而不是继续扩展 memo 控制。

## 2. 本轮闭合的系统断点

此前已有三个局部证据：

1. relation memo 能保存和 replay Local5 的 relation record；
2. tagged T450 引擎能完成单输入头、单输出 tile 的数值作业；
3. cross-head executor 能完成三输入头 OUT32 的 Acc32 归约。

但它们尚未回答以下系统问题：

- relation 是否能跨多个 output tile 保持驻留；
- 每个 output tile 的 projection weight 是否具有独立生命周期；
- memo miss 是否能无损回退到完整 score/Shiftmax5 重算；
- replay、重算和跨头 Acc32 是否使用同一结果合同；
- partial stream 在 duplicate、reorder、wrong-last 或提前结束时是否污染 Acc；
- memo 流量收益在公平 RTL 周期下是否仍显著。

本轮把这些问题放入同一 scheduler-to-result transaction 中验证。

## 3. RTL 数据流

```text
gatestack_output_tile_scheduler
  -> output tile 0, head 0..2
       -> job-local projection weight load
       -> real Q7 score + integer Shiftmax5
       -> relation transpose / source-major relation build
       -> resident decision
       -> TCFM5 projection
       -> 450 x OUT32 scalar partial Acc32
  -> output tile 1..2, head 0..2
       -> job-local projection weight load
       -> resident head: exact relation replay
       -> nonresident head: full score/Shiftmax5 fallback
       -> same TCFM5 projection and Acc32 path
  -> cross-head exactly-once Acc32 reduction
  -> 450 x OUT32 final drain
```

实现上由 `qfit_local5_memo_tagged_t450_job_engine` 管理持久 relation 与作业级权重：

- `job_decode_required=1` 仅允许首 output tile 建立 relation；
- 后续 tile 先发起 replay，命中后不请求 Q/K Token；
- replay miss 不跳过计算，而是回到完整 Token 请求和 score/Shiftmax5 路径；
- 每个 head/output-tile 作业完成后显式释放 weight context；
- relation 物理存储在下一 window 开始时统一回收，`cache_release` 在本轮只作为逻辑
  生命周期意图计数，不能解释成每个 head 都执行物理清空。

## 4. 权重与 Relation 生命周期修复

原 `qfit_fcsr_relation_memo_projection_top` 把
`weight_context_release` 固定为 0。单 tile 验证不会暴露该问题，但多 output tile 使用
不同权重时，后续作业无法可靠地建立新权重上下文。

本轮增加显式握手：

```text
weight_context_release
weight_context_release_ready
```

并把它接入 TCFM5 projection backend。relation memo 与 projection weight 因而具有
不同生命周期：

- relation：按 window/input-head 驻留，可跨 output tile 复用；
- weight：按 input-head/output-tile 作业驻留，每个作业结束后释放。

这是 exact 多 tile 集成所必需的正确性修复，不是 DATE 架构贡献。

## 5. 独立 Oracle 与输入合同

`scripts/generate_local5_memo_multitile_oracle.py` 独立产生三输入头、三输出 tile 的
整数金参考：

| 输入头 | relation 特征 | active record | term | 预期行为 |
|---:|---|---:|---:|---|
| 0 | 稀疏 | 6 | 7 | 首访建立，后续两 tile 命中 |
| 1 | 稠密 | 450 | 21920 | 不驻留，后续两 tile 完整回退 |
| 2 | 稀疏 | 8 | 10 | 首访建立，后续两 tile 命中 |

权重函数同时依赖 input head、output tile、lane 和 output channel，因此陈旧权重上下文会
直接造成 Acc32 失配。runtime invalid-candidate mask 只禁止候选参与 score/gate，不清除
几何有效位置的 `K_self` payload；该合同与 relation transpose 后的 gated-K 投影一致。

## 6. 公平周期与流量结果

一键入口：

```bash
sim_qfit/run_qfit_local5_memo_multitile_checks.sh
```

正式产物：

```text
results/qfit_local5_memo_multitile_20260809/report.md
results/qfit_local5_memo_multitile_20260809/summary.json
results/qfit_local5_memo_multitile_20260809/status.tsv
```

| 仿真器 | seed | recompute 周期 | memo 周期 | 加速 | Token 减少 |
|---|---:|---:|---:|---:|---:|
| Icarus | 17717 | 696143 | 686260 | 1.0144x | 44.44% |
| Icarus | 44257 | 696862 | 686838 | 1.0146x | 44.44% |
| Icarus | 48879 | 696822 | 686934 | 1.0144x | 44.44% |
| Verilator/SVA | 17717 | 696143 | 686260 | 1.0144x | 44.44% |
| Verilator/SVA | 44257 | 696862 | 686838 | 1.0146x | 44.44% |
| Verilator/SVA | 48879 | 696822 | 686934 | 1.0144x | 44.44% |

每次运行均完成：

| 账本 | 数量 |
|---|---:|
| memo Q/K Token request | 2250 |
| recompute Q/K Token request | 4050 |
| memo hit / fallback | 4 / 2 |
| replay record | 28 |
| partial Acc32 | 129600 |
| final Acc32 | 43200 |
| oracle mismatch | 0 |

Icarus 与 Verilator/SVA 对每个 seed 周期完全一致。以上数字是合成前 RTL 周期和事务
统计，不是 SRAM macro 时序、目标频率、整帧 FPS 或 ASIC PPA。

## 7. 为什么 44.44% 流量削减只有 1.0145x 加速

每个 input-head/output-tile 作业无论 relation 是 replay 还是重算，当前接口仍需串行
传送：

```text
450 token x 32 output = 14400 scalar partial Acc32
```

三输入头、三输出 tile 因而固定产生 129600 个 scalar partial，之后还要执行跨头
1RW read-modify-write 和 43200 个 final scalar readout。relation score 重算只占总周期
的一小段，所以减少 Q/K 请求并未消除主导工作。

这直接否定以下主张：

- “Local5 relation memo 本身带来显著端到端加速”；
- “memo hit rate 高即可推出高吞吐”；
- “继续增加 memo 容量或替换策略是当前最高优先级”。

memo 仍可能降低前端 SRAM 动态访问能量，但在没有 SAIF/PTPX 和同宏功耗结果前，只能
记为 `[待验证]`，不能用 Token 请求数代替功耗结论。

## 8. Partial 故障与服务防火墙

本轮加入四种运行时 partial-stream 故障：

1. duplicate：重复首个 partial；
2. reorder：先发送 index 1；
3. wrong-last：首个 partial 提前置 `last`；
4. early-done/drop：一个合法 partial 后提前结束。

四种故障在 Icarus 中均满足：

- 坏 beat 不产生 Acc memory command；
- duplicate/drop 只保留故障前已经提交的一次合法写；
- reorder/wrong-last 不产生写；
- protocol error 粘滞；
- 错误后外部 Token、weight 和 result 服务请求被关闭。

测试过程中发现，原 cross-head executor 虽不再写 Acc，但已经启动的子引擎仍可能在
错误后继续请求 Token/weight。修复后增加 child-service firewall，仅在
`TX_RUN_HEAD && !protocol_error` 时开放请求和响应。该修复属于 fail-closed 正确性，
不计入性能创新。

## 9. SVA 与开放综合代理

新增或扩展的检查包括：

- replay 路径不得产生 Token 请求；
- Token/weight/result ready/valid stall 稳定；
- clean job 的请求、响应和结果账本完整；
- partial 身份错误不得产生 memory command；
- error 粘滞且停止外部工作；
- Token 几何范围与 result 完成顺序。

完整回归状态：

| 项目 | 状态 |
|---|---|
| 三 seed memo/recompute Icarus bit-exact 对照 | PASS |
| 三 seed memo/recompute Verilator/SVA 对照 | PASS |
| 四种 partial 故障 fail-closed | PASS |
| memo executor lint 与 Yosys 开放映射 | PASS |
| 原 cross-head OUT32 全回归 | PASS |

Yosys 对展开后的 cross-head memo executor 报告 6090 个 generic cell 和 31 个
`$mem_v2`。这些数字只证明 RTL 可被开放工具读取和映射；没有工艺库面积、SRAM macro、
SDC、STA 或活动功耗，禁止称为 PPA。

## 10. 对 DATE 创新性的影响

本轮提高了 Local5 的系统完整度与证据可信度，但没有把 FCSR 变成强架构贡献。准确
定位如下：

| 机制 | 当前证据 | DATE 定位 |
|---|---|---|
| exact relation memo/replay | bit-exact，Token -44.44% | 可选前端流量机制 |
| miss 完整回退 | bit-exact | 正确性保障 |
| weight/relation 双生命周期 | RTL 闭合 | 系统集成合同 |
| dual context、bank、FIFO | 已实现 | 常规微结构，不单列贡献 |
| 当前 scalar Acc32 数据流 | 1.0145x | 已证明是瓶颈，需重构 |

Local5 后续若要形成 DATE 级架构主张，应围绕“relation 确定后如何改变投影与累加的物理
工作粒度”展开，而不是围绕缓存命中本身包装。可评估的下一轮候选是：

1. **OUT32 向量 Acc 原位跨头驻留**：不把 1024-bit partial 搬到新的宽存储，而是
   让 head 0--N 直接在现有 TCFM5 向量 Acc 字中连续累加，只在最后 head 后读出；
2. **多 output-tile supertile**：relation 驻留时，一次 source-major 遍历同时服务
   多个 output tile，使 memo 复用与 projection weight streaming 在同一循环内兑现；
3. **vector/supertile 联合数据流**：只有前两项分别完成收益归因后才允许联合，避免
   同时改变向量宽度和 output-tile 遍历而无法解释收益来源。

这些候选目前均为 `[待验证]`，尚未实现，不得写成已完成贡献。

## 11. 双线边界与 Motion 后续

Motion 线没有停止，也不被冻结为只维护回归。当前策略是：

- Local5 先完成本轮系统断点和评审；
- Motion 保持已有 SCS/NMF/DCTF/RQTB2S 的可回归状态；
- 下一轮机制选择同时允许 Local5 和 Motion 提案；
- 新机制必须先证明它消除真实 dominant work，再进入 RTL；
- 两条线分别给出公平强基线和独立 DATE 评分，不能把两边弱机制拼成一个贡献数。

Motion 候选仍可包括基于真实 ordered trace 的 TTB/STT 重排、Gate-code/term 数据流、
跨行或跨 output-tile 复用，以及此前被否决但在新 full-resolution 分布下可能重新成立的
机制。没有新 profile 或同约束收益时，不因论文命名好听而晋级。

## 12. 下一轮最高优先级

先由独立 DATE 审稿人评审本轮证据包，重点回答：

1. 1.0145x 是否足以保留 FCSR 为正文机制，还是应降到附录/流量消融；
2. OUT32 vector、multi-output-tile supertile 或二者联合，哪一个最可能直接消除当前
   129600 个 scalar partial 的主瓶颈；
3. Local5 在 full-resolution trace 未完成前，下一轮可做到什么证据等级；
4. Motion 下一轮应补强已有机制，还是选择一个新 architecture principle。

评审后每轮仍只实现一个最高优先级缺口。Local5 完整度优先不等于 Motion 停止；当
Local5 当前断点闭合或等待 fullres profile 时，立即切回 Motion 的可审计新机制迭代。

## 13. 独立 DATE 评审

独立只读审稿人检查了本文、正式结果、`docs/275` 和关键 RTL，没有修改文件。评分为：

| 范围 | 评分 | Recommendation |
|---|---:|---|
| 本轮 Local5 证据包 | 4.2/5 | 可作为高质量 RTL 签核包 |
| 整个 Local5 硬件线 | 3.0/5 | Weak Reject / Major Revision |
| 架构创新性 | 2.5/5 | Weak Reject |
| 当前 DATE 投稿总体 | 3.0/5 | Weak Reject |

评分没有因 RTL 工作量而抬高。评审确认本文没有实质过度声明，但指出五项关键缺口：

1. FCSR 没有消除实际主瓶颈，不能列为摘要贡献；
2. 定向三 head 测试不能替代 fullres same-window all-head 分布；
3. TCFM5 内部本来就是 OUT32 向量 Acc，当前读接口和独立跨头 Acc 人为将其标量化；
4. 缺逐阶段 cycle ledger，当前只有定性 Amdahl 判断；
5. 缺同 SRAM macro、时序闭合、活动功耗和 best-legal 存储组织。

FCSR 的裁决是“降附录，不是否决”：保留 exact replay、miss fallback、生命周期、
流量和故障证据；只有 fullres 稳定命中且同宏动态能量显著下降，或它成为后续数据流的
必要 enabling mechanism，才允许恢复为正文机制。

## 14. 评审后唯一优先级与公平合同

下一轮唯一实现项冻结为：

> **基于现有 TCFM5 OUT32 向量 Acc 字的原位跨头驻留累加。**

目标数据流是：

```text
head 0 terms -> TCFM5 vector Acc 初始化并累加
head 1..N terms -> 同一组 vector Acc 原位继续累加
last head -> 唯一一次 final drain
```

它必须真正删除独立 `14400x32b` 跨头 Acc 与中间 scalar partial 搬运。若实现退化为
远距离 1024-bit partial streaming，或没有减少物理 Acc 访问，则停止候选。

公平对照冻结为：

| 候选 | 跨头 Acc | Relation |
|---|---|---|
| B0 | scalar 1RW RMW | recompute |
| B1 | scalar 1RW RMW | memo |
| B2 | vector-resident in-place | recompute |
| B3 | vector-resident in-place | memo |

必须报告逐阶段 cycle、总周期、stall、所有 transaction/bit、有效/分配存储 bit、bank
和端口、容量浪费、宏读写/激活 bit、向量加法与总线切换、WNS 闭合后的开放物理代理。
SRAM 同时给两种公平口径：

1. `iso-macro`：使用相同宏集合，隔离 scalar/vector 调度；
2. `best-legal`：双方在同一 compiler/library 下选择各自最优合法组织。

1024-bit 行为级逻辑字必须明确拆成多少个 32/64/128/256-bit 宏；不得把 `$mem`、宽
总线或 32-lane 加法器视为免费资源。multi-output-tile supertile 暂不进入本轮，因为
它不能单独消除 129600 个 scalar partial，联合方案也会破坏收益归因。

Motion 在 Local5 本轮实现后恢复新机制筛选。独立评审只推荐一个方向：在
`TTB8-ZKQI + RQTB2S` 的相同边界内评估 exact temporal-residual score fusion。必须
先扣除 zero-K 重叠，再统计 active pair 的 delta lane work、fallback、状态 SRAM、
cycle、toggle 和面积；独立增量不足 10% 或状态成本后面积归一吞吐/EDP 不改善即否决。
