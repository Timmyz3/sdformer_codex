# Local5 formal G0 准入检查表（joint-head 已完成，formal 继续推进）

> 依据：`docs/297_Local5_EREP正式Adapter输入输出冻结_20260810.md` adapter 合同、`docs/311` archive 裁决、`docs/312` 看板  
> 当前状态：Gate A/B 已通过；Gate C 已完成 100/100 canonical 数值 shard 和单窗
> cross-head canary；Gate D 的完整 phase ledger/admission 尚未完成，内部 formal G0
> 仍为 **DENY**。论文中统一称“全 profile 定点一致性准入门”，不称数学形式证明。

## Gate A — producer 产物

- [x] `ordered_term_manifest.json` 存在（13800 groups）
- [x] `ordered_term_items.npz` 存在且 SHA 可绑定
- [x] 100/100 sample 完成；GPU exclusivity audit PASS
- [x] `joint_head_run_identity.json` / selection plan / projection contract SHA 一致

## Gate B — preflight

```bash
cd hw_autoresearch_nts07
bash sim_qfit/run_local5_erep_formal_preflight_v4.sh
# 或 positive fixture：run_local5_erep_formal_preflight_v4_positive_fixture.sh
```

- [x] 无 manifest 时必须 `DENY_FORMAL_MANIFEST_ABSENT`（fail-closed）
- [x] 有 manifest 时 HxH 拓扑 / key multiset 通过

正式结果：`results/local5_erep_formal_preflight_v4_formal_20260811`，状态为
`PREFLIGHT_PASS_NOT_G0`，覆盖 1200 window、13800 input-head group 和 210600 个
H×H 投影任务。

## Gate C — 三段 adapter（仅 Gate A+B 后）

1. software-expected 金参考生成器 + SHA  
2. DUT RTL filelist + 仿真命令 + 原始输出 SHA  
3. read-only merge → miter archive（禁止同一 adapter 自证两边）

当前已完成单窗来源隔离、集成 cross-head canary 和四个真实 stage 的分层
smoke：

```text
sample0 / stage0 / block0 / window94
3 input head x 3 output tile x OUT_DIM32
```

partial canary 位于
`results/local5_erep_formal_canary_v2_reviewfix_20260811`：Icarus 与
Verilator/SVA 各导出 129600 个 DUT partial Acc32，只读 merge 后各有 43200 个
final Acc32，均零失配。

集成 canary 位于
`results/local5_erep_integrated_cross_head_canary_v4_final_20260811`：真实
Q/K、score/Shiftmax5、relation、term、checkpoint INT8 projection 与 DUT 内跨 head
累加得到 43200 个 final Acc32，两个模拟器均零失配。两项只证明单窗链路可执行，
不等于 1200-window formal adapter 已完成。

早期分层 smoke 已被 sample2 的 v5 密封 release 全拓扑数值分片覆盖；下表区分
H3 的双模拟器证据和 H6/H12/H24 的单模拟器正式 provenance：

| stage | head 数 | final Acc32 | Verilator 与软件金参考 |
|---:|---:|---:|---:|
| 0/H3 | 3 | 43,200 | 0 mismatch（Icarus + Verilator，完整 provenance） |
| 1/H6 | 6 | 86,400 | 0 mismatch（sample2，Verilator，v5 完整 provenance） |
| 2/H12 | 12 | 172,800 | 0 mismatch（sample2，Verilator，v5 完整 provenance） |
| 3/H24 | 24 | 345,600 | 0 mismatch（sample2，Verilator，v5 完整 provenance） |

H6/H12/H24 已具备 v5 source bundle、工具/命令绑定和真实窗口 Acc32 零失配。
其中 H24 又完成 identity-service 逐事件消费、hold2 精确反压、独立 cycle-free state
结构 oracle 和 source-only Phase Array Store；仍缺 Icarus 交叉复验和 phase-ledger formal。
当前已独立审计 numeric 正式进度为 100/100；这只关闭 Gate C 的 canonical 数值分片，
不关闭 Gate D 的 phase/admission。sample3-6 详见
`docs/334`；
sample7-14 又用同一 sealed v5 release 完成 96 个 block-window、15,897,600 个 Acc32
零失配，详见 `docs/336`。sample0/1 为早期 provenance，故同一 v5 release 的覆盖为
sample2-14 共 13/100。

sample31-46 后续用同一 v5 release 完成 192 个 canonical block-window 和
31,795,200 个 Acc32 零失配，独立审阅为 `4.5/5 Conditional Accept`，详见
`docs/339`。sample15-30 随后完成同规模扩跑，并通过全量 NPZ、随机 memh、共同
release 和来源链独立复核，同为 `4.5/5 Conditional Accept`，详见 `docs/341`。
因此正式累计更新为连续 `sample0-46 = 47/100`、564 个 canonical block-window、
93,398,400 个 Acc32 零失配；同一 sealed v5 release 的严格范围为
`sample2-46 = 45/100`。旧批次未封存 live launcher 的 P2 继续保留，不能倒签。

sample47-62 随后使用同一 sealed v5 release 完成 192 个 canonical block-window、
31,795,200 个 Acc32 零失配。本批冻结 batch runner、单测和实际 shard launcher，且
每个 sample 执行前后均 fail-closed 校验 launcher SHA。新增只读累计审计器直接复核
sample0-62 的 63 份 NPZ、拓扑、offset、逐元素 expected/actual 和 provenance 链，
累计为 756 个 canonical block-window、125,193,600 个 Acc32 零失配；同一 sealed v5
的严格范围为 `sample2-62 = 61/100`、732 个窗口、121,219,200 个 Acc32。独立复审
为 `4.5/5 Accept`，但仅限 workspace-bound 数值 RTL 证据包，详见 `docs/342`。

sample63-99 已继续补齐。最终累计包
`results/local5_numeric_coverage_audit_sample0_99_v4_final_20260813` 只读复核 100 个
sample、1,200 个 canonical block-window 和 198,720,000 个 pre-bias/pre-BN/
pre-requant/pre-residual Acc32，mismatch=0；同一 sealed v5 的严格范围为 sample2-99。
执行来源链包 `results/local5_numeric_execution_chain_sample0_99_v3_final_20260813`
可证 91 个 `RUN` 和 2 个 `RESUME`，并保留 3 个 pre-batch 来源与 4 个 legacy receipt
缺口，不倒签历史。

另有独立的 post-score 性能/正确性强基线包：

```text
results/local5_joint_ep29_tcfm5_linear5_realw_sample100_population_rtl_v5_final_20260813
```

它在 100 个 sample 各选一组 qualified T450 group，绑定真实 checkpoint INT8 权重，
比较相同同步 relation SRAM、frontier、term builder、五个单写 Acc bank 与 readback
合同下的 TCFM5 和 Linear5。L1/L2 memory latency 下分别为 1.496x/1.466x；四配置
合计 360,000 个 Acc32 比较零失配，四配置均通过随机 gap SVA，并封存 actual Acc32、
RTL/TB/SVA/source SHA。另有四 stage 各一个最大 term group 的完整 32-channel output
tile 回放，四配置合计 230,400 次 Acc32 比较零失配。上述包是
`[rtl]+[profile-qualified-trace]` 的组件级证据，不含完整
score/Shiftmax5，不提供 462,600 phase ledger，也不改变 Gate D 或 formal G0。

审稿后 TB 已新增真实 `STAGE_ID/BLOCK_ID/WINDOW_ID` 参数与范围检查，actual receipt
也必须与 task plan 坐标一致。sample0 已用同一冻结版本重跑全部 12 个 block：

```text
results/local5_erep_numeric_sample0_shard_v1_reviewfix_20260811
12/12 window
1,987,200 pre-bias/pre-requant Acc32
mismatch=0
formal G0=DENY
```

该 shard 首次关闭单个 sample 的真实 stage/block/window 数值覆盖和窗口级可恢复
adapter。随后建立共享 sealed RTL release，并用同一份 H3/H6/H12/H24 可执行文件完成
sample1：

```text
release: results/local5_erep_numeric_rtl_release_v2_20260811
release manifest SHA256:
ee1bf0d6001dd963284680faf67be4103f96e8910144def48350ca8136293676

sample1: results/local5_erep_numeric_sample1_shard_v2_release_20260811
12/12 window
1,987,200 pre-bias/pre-BN/pre-requant/pre-residual Acc32
118,036,260 regression cycles
mismatch=0
formal G0=DENY
```

此前数值分片为 3/100。共享 v5 release 先通过单 H3 真实消费后验封和独立复审
（4.2/5 Conditional Accept），随后 sample2 的 12-window H3/H6/H12/H24 扩跑
全部通过：1,987,200 个 Acc32 零失配，跑后 release 再验封 PASS。H24 identity/phase
单窗闭环后，sample3-14 已用同一 v5 release 扩跑通过，累计 numeric coverage 为
15/100。

## Gate D — archive / G0

```bash
bash sim_qfit/run_local5_erep_ledger_replay_v4_checks.sh
# formal 规模：1200 window / 13800 head / 462600 phase / 198720000 Acc32
```

- [x] sample0 至 sample99 canonical numeric shard mismatch=0  
- [x] 100/100 sample shard 独立 expected/actual/miter 完成（1,200 canonical window）  
- [ ] 462,600 条 phase ledger 覆盖且无缺失/重复  
- [x] H3/H6/H12/H24 参数化 identity-service + typed phase 单窗 RTL canary 逐事件同构；
  H12 已用 Phase Array Store 完成 12,244,663 行 legacy/source-only 双重放、10/10 负例和
  RSS 收口；H24 已完成 47,941,735 行真实 source-only 重放、345,600 个 Acc32 零失配、
  43,522,611 条 state 解析全序、精确 hold2 和 10/10 负例，详见 `docs/333`  
- [ ] `admission_receipt.json` 绑定全部输入/输出 SHA  
- [ ] 通过后才允许 EREP candidate RTL  

### Compact telemetry 前置层

sample3-6 的 48 个窗口已完成 compact telemetry 健康检查，见 `docs/335`。该层绑定
`window_complete -> actual receipt/log/memh -> Acc32 archive slice`，并独立重算 H
闭式计数和 transaction delay，最终复审为 4.3/5 Accept（仅健康检查）。它不含
462,600-phase 相序/resource/command ledger，不能勾选 Gate D 或改变 formal G0。

独立 DATE 审阅进一步确认：numeric v5 和现有 Phase Array Store 都不能直接生成正式
resource ledger。后续冻结为“全量被动 phase telemetry + 四种 H/18-cluster 完整 trace
anchor + 参数化流式证明 + 独立只读重放”，详见 `docs/337`。该方案的方法学评分为
`4.1/5 Conditional Accept`，但 pilot 尚未完成，因此 Gate D 仍不勾选。

H3 pilot 后续已在 canonical window249 上闭环，独立复审为 `4.2/5 Conditional
Accept for H3 telemetry pilot`，详见 `docs/338`。该晋级只允许进入 H24/有限多窗口
pilot；52 条局部 semantic phase 不等于 formal 462,600 条，因此 Gate D 仍不勾选。

sample15-78 与 sample79-99 已生成 compact telemetry 前置层；后者覆盖 21 个 sample、
252 个窗口，并绑定 env-sealed v2 parent batch。所有 `cycles/frontend_cycles` 都是同源
验证回归遥测，不是部署周期；compact 层仍不包含 462,600 条 phase/resource/command
ledger，故不改变 Gate D 或 formal G0。

## 明确禁止

- 用 synthetic Accept 冒充 formal Accept  
- admission 前扩 EREP 架构 RTL  
- 把 formal 正确性合同单独写成 DATE 主创新
