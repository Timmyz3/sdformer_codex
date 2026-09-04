# Local5 EREP G0 Direct/TCFM5 RTL 校准 V-plan v4

## 1. 范围与裁决

本设施只校准现有 Direct online 与 TCFM5 1RW 数据路径，不实现 EREP C1--C3，
也不改变 DUT。当前证据只能标为 `[synthetic-RTL]+[rtl校准]`，不能标为
`[formal-T450]`、EREP `[rtl]` 或 ASIC PPA。

前五次独立评审均未接受整包。第六次评审给出 `4/5 Weak Accept`，接受
final-stream、固定 fixture 的 snapshot fire/state 合同及 synthetic calibration
整包，P0/P1 为 0。snapshot valid/ready 波形没有独立事件证明，只能作为
`fire=valid&ready` 的辅助字段。正式
`13800 head-group / 1200 joint-window` adapter 仍不存在，formal G0 继续 `DENY`。

## 2. 接入边界

三个 monitor 均由校准 TB 显式实例化：

- `qfit_local5_erep_direct_monitor_v4`；
- `qfit_local5_erep_tcfm5_monitor_v4`；
- `qfit_local5_erep_serializer_monitor_v4`。

runner 不再定义 `QFIT_EREP_BIND_V4`，该宏会主动触发编译错误；
`QFIT_EREP_LEGACY_BIND_V4` 的全局 bind 代码已删除。正式 adapter 禁止使用全局
serializer bind 或 generate 层次自动附着。monitor 有本地观测状态和
`$display/$fatal`，它是动态校准器，不是无状态 formal checker。

## 3. Strict raw schema

每行必须同时满足候选和事件对应的精确字段集合，以及固定的
`resource/kind/phase/scope` 语义；未知事件、未知字段、缺字段、重复字段、错误
bank-resource 对应和错误 phase-kind 对应均失败。文件必须严格按以下顺序结束：

1. `EREP_V4` 结构化行；
2. 唯一、计数完全匹配且位于全部 trace 后的
   `PASS Local5 EREP calibration v4 ...`；
3. 唯一的 Icarus 或 Verilator 已知 `$finish` 行，之后不得再有输出。

任何额外 warning/fatal/error/文本均失败。CLI 只保留 `--trace` 和 `--output`；
不存在 `--evidence formal_profile`、`--seed` 或可改 sample/stage/block/head/tile
的参数。因此 synthetic 工具无法通过换标签伪造 formal 证据。

## 4. 数据与数值闭环

校验链为：

```text
relation accept 完整 payload
  -> relation read 顺序
  -> FIFO2 enqueue/dequeue 有序 payload
  -> term (gate, lane, destination mask)
  -> gate * weight 的 packed delta
  -> 五色 destination bank/address
  -> logical update
  -> first-touch write 或 read-modify-write
  -> Acc32 final/vector/serializer scalar
```

具体检查：

- relation 覆盖与 source identity 唯一；
- FIFO2 严格保持 relation-read 顺序，不再只比较 multiset；
- enqueue 的 K、gate、mask、坐标逐字段等于 accepted relation；
- 解析器按 term-builder 的“lane 升序、gate 首现去重、相同 gate 合并 mask”规则，
  从 FIFO payload 独立重建全部 Direct term，并逐项比较 source/coordinate/lane/
  gate/mask/last/source-last；
- TCFM5 接受的每条 term 都必须 `commit=1`，窗口 last 必须唯一且位于末项；
- 每个 term 的 delta 等于 `gate * weight`；
- destination mask 展开后的 logical update 多重集完全一致；
- 1RW 首触直写、复访 read/writeback 的地址和数据逐命令重放；
- Direct scalar final、TCFM5 vector response 和 serializer scalar 均由 term ledger
  独立重算，Acc32 mismatch 必须为 0。

## 5. 测得 execute-tail 后的周期规则重建

Direct 的 fill 与 readout 规则可独立检查，但 execute 结束仍取自
`last_term_accept_cycle`。因此本设施不再称其为“独立周期预测”，而明确称为
“测得 execute-tail + 冻结 drain/readout 规则重建”：

```text
PREPARE_BEGIN = 0
FILL_BEGIN = 2
EXECUTE_BEGIN = 2 + 2*relation_records + 1
DRAIN_BEGIN = last_term_accept_cycle + 2
COMPUTE_DONE = DRAIN_BEGIN + 2
read_accept[i] = COMPUTE_DONE + 3*i
read_response[i] = read_accept[i] + 2
full_cycles = COMPUTE_DONE + 3*scalar_count
```

规则重建边界必须逐项等于 monitor 边界，重建总周期必须等于测得总周期。当前
phase 分解为 prepare/fill/execute/compute-drain/readout = `2/19/40/2/54`，规则
重建和 RTL 均为 `117 cycles`。该相等不能作为独立性能模型证据。

## 6. Backpressure 覆盖

stall 时稳定性现在覆盖：

- Direct descriptor 的 source/plane/y/x/K/gates/mask/last；
- Direct term 的 source/plane/y/x/lane/gate/mask/last/source_last；
- TCFM5 term 的 source/plane/y/x/lane/gate/mask/commit/window_last；
- serializer 的 source/坐标/out/data/last。

固定 synthetic fixture 的 stall 账本必须精确为 Direct term `4` 拍、TCFM5 term
`1` 拍、serializer `18` 拍，不能只要求非零。每个连续 stall run 必须在下一拍
以完全相同 payload 被接受；TCFM5 stall 时 `commit=0`、接受时 `commit=1`。
serializer 的 9 个双拍 run 触发 monitor 连续稳定性检查；Direct 和 TCFM5 term
是单拍 stall，只能声称“stall-to-accept identity 已检查”，不能声称连续两拍
稳定性已覆盖。新增 fail-closed 条件为：

- `(candidate,resource,cycle)` 必须唯一；
- 一个接受事务最多只能被一个 stall run 消费；
- run 形状必须精确为 Direct `4x1`、TCFM5 term `1x1`、serializer `9x2`；
- 结果保存全部 stalled cycle 及其唯一接受事务 payload；
- 每个 monitor 内 `time = epoch + 2000*cycle`，stall 与下一拍接受的时间连续；
- candidate/mask/K/gates/lane/gate/address/delta/vector/scalar 等字段逐一检查 synthetic
  RTL 实例的位宽和值域。

## 7. 当前回归结果

结果目录：

```text
results/local5_erep_calibration_v4_snapshotledgerfix_20260810
```

| 指标 | Direct online | TCFM5 1RW |
|---|---:|---:|
| compute cycles | 64 | 10 |
| full readout/serializer cycles | 117 | 69 |
| logical Acc updates | 20 | 11 |
| compute physical commands | 31 | 16 |
| 全部 physical 1RW commands | 49 | 22 |
| Acc32 mismatch | 0 | 0 |
| RMW writeback mismatch | 0 | 0 |
| same-cycle 1RW collision | 0 | 0 |

Icarus 12.0 与 Verilator 5.020 的 phase、数值、RMW、stall、identity 和 C0 ledger
逐字段一致。C0 event ledger SHA-256 为：

```text
c7d4a6564eb95ee883cb8a0c725bd87990370e05d8488826af9b0db0513c5639
```

单测/负测：schedule `14/14`、identity `15/15`、parser/mutation 共 `10/10`。
除旧攻击外，新增重复 stall cycle、同一接受事务重复消费、serializer run 形状
破坏、event time 回退/跳变、字段越界、coherent valid/fire 篡改和 coherent
state/boundary 位移，均被拒绝。
Icarus/Verilator 新生成日志逐字段一致，结果哈希清单全部通过。

## 8. V-plan 状态

| ID | 优先级 | 检查项 | 状态 |
|---|---:|---|---|
| G0-V4-01 | P0 | raw schema、字段语义、位宽、时间与终止顺序 | `[synthetic-RTL] 六审ACCEPT` |
| G0-V4-02 | P0 | relation/FIFO2→term 完整重建 | `[synthetic-RTL] 六审ACCEPT` |
| G0-V4-03 | P0 | term→logical update→RMW | `[synthetic-RTL] 六审ACCEPT` |
| G0-V4-04 | P0 | term ledger→Acc32 final | `[synthetic-RTL] 六审ACCEPT` |
| G0-V4-05 | P0 | measured-tail 周期规则重建 | `[synthetic-RTL] 本地PASS/非独立预测` |
| G0-V4-06 | P0 | stall 唯一账本、固定 run 形状与 accept identity | `[synthetic-RTL] 六审ACCEPT` |
| G0-V4-07 | P0 | 双模拟器逐字段一致 | `[cross-sim] 本地PASS` |
| G0-V4-08 | P0 | synthetic 无 formal 标签/bind 入口 | `[代码审计]+[负测] 本地PASS` |
| G0-V4-09 | P0 | hash-bound formal T450 adapter | `[待验证] DENY` |
| G0-V4-10 | P0 | 1200-window command ledger | `[待验证] DENY` |
| G0-V4-11 | P1 | 多窗口/reset/macro backend | `[待验证]` |

## 9. Formal adapter 最低合同

正式 adapter 只能读取固定路径下通过 qualification 和 GPU audit 的
`ordered_term_manifest.json`、payload、selection plan、cohort、run identity 和
projection contract，并逐 SHA 绑定。它必须：

1. 覆盖恰好 `13800` 个 head group、`1200` 个 joint window；
2. 每行携带固定 sample/sequence/stage/block/window/head/tile identity；
3. 使用 full T450/OUT_DIM32 RTL replay 产生 Direct phase/cycle 与 Acc32 miter；
4. 用实际 RTL command/phase ledger 重建 C0，不接受调用者输入 scalar cycle；
5. 对 C1--C4 只生成 `[rtl校准]+[模型]` 资源调度，不伪装为候选 RTL；
6. 固定 producer revision、dirty-state receipt、工具和 adapter SHA；
7. 任一缺失、重复、unknown、hash mismatch 或 coverage mismatch 均不写 PASS。

在这两项 formal P0 关闭前，`local5_erep_statistics_v4.py` 的生产入口应继续因
缺少 admission artifacts 而 fail closed，EREP 候选 RTL 禁止启动。

## 10. 历史：第四次复审整改增量

第四次独立复审给出 `3/5、Synthetic DENY、Formal DENY`。其 P1 指出原解析器
只验证最终 Acc32 数值集合，没有冻结 Direct/TCFM5 的最终流顺序、读请求与响应
身份配对，以及 serializer `last` 的唯一位置。P2 指出 snapshot packed fire 与
控制状态可被合法位宽内篡改，结果清单也没有绑定构建日志、二进制和 complete。

整改后的 synthetic 合同新增：

1. Direct drain accept/response 必须严格按 source-major/out-major 排列，响应保持
   同一 `(source,out)`，且响应 cycle 等于 accept cycle 加 2；accept 的
   `(plane,y,x)` 必须由 source 唯一重建。Direct 接口没有 `last`，结果明确标记
   `final_last_marker_applicable=false`，不得过度声明。
2. TCFM5 vector accept、vector response、serializer input 必须按 source `0..8`
   排列；serializer output 必须按 source-major/out-major 排列。serializer input
   与 output 均只允许最后一项 `last=1`。
3. 每个 cycle snapshot 必须满足 packed `fire=valid&ready`；Direct 的 phase、
   busy/done、relation active/done，以及 TCFM5 的 phase/state、busy/done 都按冻结
   边界逐 cycle 检查。
4. mutation 测试新增最终 `last` 清零、最终 payload 对调、Direct response payload
   对调、packed fire 翻转、busy 翻转及 candidate-valid 合法子集篡改。
5. `result_sha256.txt` 覆盖 Icarus/Verilator 构建日志、归一化日志与可执行二进制；
   `receipt_sha256.txt` 再绑定 result 清单与 `complete.json`。这些哈希用于本地
   防误配，不构成外部信任根。

该轮历史结果目录：

```text
results/local5_erep_calibration_v4_finalstreamfix4_20260810
```

Icarus/Verilator 跨仿真逐字段一致，schedule `14/14`、identity `15/15`、
parser/mutation `10/10`；Direct 与 TCFM5 的 Acc32 mismatch 均为 0。当前仅可写
`[synthetic-RTL]+[rtl校准] 本地PASS/待第五次独立复审`。该状态已过期，不得作为
当前结论。正式 adapter、G0 与候选 EREP RTL 仍为 `[待验证] DENY`。

## 11. 第五次复审整改增量

第五次评审给出 `3/5`，接受 final-stream 子合同，但对 synthetic 整包继续
`DENY`。两个 P1 均已按固定 fixture 的独立 golden 整改：

1. Direct boundary 固定为 `0/2/21/61/63`，TCFM5 固定为 `0/1/6/9`；state、
   busy/done 与 phase 只使用这些预注册 cycle，不再使用 monitor 自报 boundary
   推导期望。
2. Direct packed fire 从同周期 `relation_accept/fifo_enqueue/term_accept/
   drain_read_accept` 重建，close 位固定为 drain begin 前一拍；TCFM5 packed fire
   从 `term_accept/vector_read_accept` 重建，未使用的 scalar-read/close 位必须为 0。
3. mutation 增加 coherent `valid+fire` 改写与 coherent `state+boundary` 位移，并在
   helper 入口统一断言 mutated trace 不得等于原 trace。

当前结果目录为：

```text
results/local5_erep_calibration_v4_snapshotledgerfix_20260810
```

19 项 result 哈希及 complete receipt 均核验通过，但没有外部信任根。第六次
复审已将本固定 fixture 的 `[synthetic-RTL]+[rtl校准]` 整包裁决为 `ACCEPT`；
formal adapter 继续 `[待验证] DENY`。

## 12. 第六次复审最终裁决

第六次评审复现了 coherent valid/fire 和 coherent state/boundary 两项攻击，均被
拒绝；独立重放 Direct 186 个、TCFM5 69 个 snapshot，fire mismatch 为 0。裁决：

- Final-stream：`ACCEPT`；
- Snapshot contract：`ACCEPT`，仅限固定 fixture 的 fire/state；
- Synthetic calibration 整包：`ACCEPT`；
- Formal adapter：`DENY`。

残余 P2：snapshot valid/ready 未分别绑定独立事件，不能声称完整握手波形证明；
相关源文件当前未纳入可复现 Git revision，本地 SHA 不能替代提交版本或签名。
