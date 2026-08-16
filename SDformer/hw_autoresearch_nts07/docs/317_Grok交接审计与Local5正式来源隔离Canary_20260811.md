# Grok 交接审计与 Local5 正式来源隔离 Canary

## 1. 本轮裁决

Grok 交接中的两项核心状态经重新执行后成立：

1. Local5 joint-head profile100 确实完成，正式 manifest 含 13800 个 input-head group、
   22848620 个 term item 和 6210000 个 source descriptor；
2. H×H formal preflight 通过，状态为 `PREFLIGHT_PASS_NOT_G0`。

但 formal G0 仍为 `DENY`。本轮新增的是一条可执行的三段来源隔离 canary，尚未生成
1200-window 正式 archive、三层 phase ledger 或 `admission_receipt.json`。

## 2. 对 Grok 报告的审计修正

### 2.1 成立部分

| 项目 | 审计结果 | 证据 |
|---|---|---|
| profile100 完成 | 成立 `[prof]` | manifest qualification=true、100 sample、12 block、13800 group |
| GPU exclusivity | 成立 `[prof]` | `gpu_exclusivity_audit.json` status=PASS、foreign PID 为空 |
| Motion 1.17× | 数字可复现 `[rtl校准模型]` | 不等于多样本真实 RTL |
| formal G0 | 仍为 DENY | 无全量 archive/admission |

### 2.2 Motion 模型的方法学边界

`docs/314` 中的 sample50--99 “held-out”只表示在 100-sample profile 上做后半集合外推，
不是模型训练留出集。周期模型只在 sample0/window0 的 138 个 RTL head-row 上拟合。
此外，带截距最小二乘会让校准集预测总和等于真实总和，因此校准集
`speedup_model == speedup_true` 不能作为独立泛化证据。可保留的声明是：

> `[rtl校准模型]` 在冻结线性模型下，100-sample profile 的估计总加速约 1.17×；
> 它仍需多样本真实 RTL 或解析 FSM 周期模型升级证据等级。

## 3. Formal preflight

结果目录：

```text
results/local5_erep_formal_preflight_v4_formal_20260811
```

通过项：

| 合同 | 结果 |
|---|---:|
| canonical joint window | 1200 |
| input-head group | 13800 |
| H×H projection task | 210600 |
| projection block | 12 |
| manifest/payload/contract SHA | 全部匹配 |

preflight 不生成 admission，`formal_g0_status=DENY` 是正确行为。

## 4. 三段来源隔离 Canary

### 4.1 冻结范围

```text
sample=0, stage=0, block=0, window=94
3 input head x 3 output tile x OUT_DIM32 = 9 task
```

该坐标直接来自正式 selection plan 和 joint-head manifest，不是 synthetic fixture。

### 4.2 数据流与来源隔离

```text
正式 descriptor + theta-folded checkpoint INT8 weight
       |                              |
       |                              +-> software-expected 生成器
       |                                  -> 3 x 450 x 32 final Acc32
       |
       +-> 显式 task plan -> DUT input vector
                            -> Icarus DUT actual
                            -> Verilator/SVA DUT actual
                                      |
                                      v
                    read-only merge：按 output tile 跨 3 head 整数求和
                                      |
                                      v
                           与 software expected 逐坐标比较
```

actual 仿真使用 `+NO_ACC_CHECK`，TB 不读取 `expected_acc.memh`；因此 DUT 原始输出不是
由 expected 文件驱动的自证。actual receipt 分别绑定模拟器命令、原始日志、DUT
TB/RTL/SVA filelist 及 SHA。merge 只读取 expected、actual、task plan 和 receipt，
不读取原始 profile。

### 4.3 结果

结果目录：

```text
results/local5_erep_formal_canary_v1_20260811
```

| 项目 | Icarus | Verilator/SVA |
|---|---:|---:|
| task | 9 | 9 |
| DUT partial Acc32 | 129600 | 129600 |
| merge 后 final Acc32 | 43200 | 43200 |
| 周期 | 15975 | 15975 |
| 软件金参考 mismatch | 0 | 0 |
| 跨模拟器 raw actual | 完全一致 | 完全一致 |

证据等级为 `[rtl]+[软件整数金参考]`，范围仅限上述一个正式 stage0 同窗。周期包含
单 head 独立任务重启和读回，不是候选 EREP 周期、full encoder 周期或 ASIC PPA。

## 5. 联合回归修复

联合运行 preflight、archive replay、ledger replay 和 canary 时，旧脚本会把同一个
`PhaseTrace` 源码以顶层模块和 `scripts.*` 两个名字加载，导致 `isinstance` 假失败。
本轮改成：包加载使用相对导入，命令行直接执行使用同目录导入；没有放宽类型检查。

联合结果：31/31 PASS。canary 的 result/receipt SHA 全部复核通过。

## 6. 不能写进论文的结论

- 不能把单窗 canary 写成 Local5 formal G0；
- 不能把 15975 cycle 写成 EREP 加速、整窗部署延迟或 full encoder FPS；
- 不能把 Icarus/Verilator 一致写成形式证明；
- 不能在全量 admission 前扩 EREP candidate RTL；
- 不能把 Yosys/OpenROAD 代理写成 ASIC PPA。

## 7. 下一步最小闭环

1. 把 canary 从单 head DUT 逐任务回放升级为集成 cross-head DUT 原始 final Acc32；
2. 选择 stage0/1/2/3 各一个正式同窗做四 stage canary，先验证最大 H=24 的控制、
   存储和运行时间；
3. 根据四 stage 实测决定 formal archive 分片粒度和 CPU 预算；
4. 再生成 1200-window phase/Acc32 分片 archive、三层 ledger 和 admission receipt；
5. 正式 G0 通过前保持 EREP candidate RTL 禁入。
