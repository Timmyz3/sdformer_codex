# M519 R5 channel-local fault VCS receipt-blind 独立打铁 r1

日期：2026-08-27  
状态：`PASS_VCS_RECEIPT_BLIND__DC_RUNNER_REPAIR_AND_NEW_STATIC_ADMISSION_ALLOWED__DC_NOT_AUTHORIZED`  
评分：**98/100**；P0/P1/P2 = **0/0/2**。

## 技术摘要

R5 的三阶段 Synopsys VCS campaign 可准入为局部功能证据。本评审不依赖作者 receipt 自报数字，而是先从合同和封存静态评审重建预期，再从原始 `compile.rc`/`sim.rc`/`sim.log`/`assert.report` 逐项重算。结果为：

1. attack / primary / equal-bandwidth 三阶段 compile/sim RC 全为 0，三条终端 PASS 各恰好一次，无 assertion/fatal/watchdog/error 签名；
2. 12 类 attack/recovery、10 个 runner-gated SVA cover、合法 response 与非法 request 同拍退役 2 次、sticky quiescence 10 次均成立；
3. K1/K1×8 和 K8/K1×8 两条 full-workload regression 的十行周期与冻结 r2 逐行一致，numeric/tuple/weight/same-edge mismatch 均为 0，要求的 stall/attack 均非零；
4. 合同 19/19 exact files、启动身份、静态评审封存、结果 inner manifest 与 outer seal 均校验通过。

因此，允许进入 **DC runner 修复 → 新的独立 static admission**。当前仍不允许启动 DC，也不得宣称 `TIM-209=0`、combinational-loop-free、PPA 或系统加速。

## 身份、范围与可复核基线

- 数据集：`results/m519_r5_channel_local_fault_vcs_r1_20260827`；粒度为三个独立 VCS phase，不是 DC/PPA 或全网帧级测量。
- 合同 SHA256：`779180ed7ca889a92c83273476f6d70a970ed5f8a713e235fd18c4600919160a`。
- VCS runner SHA256：`e6d7160b47b4f49827dcf7c65ef7036bb9139911b64de2992a0daec350897dc0`。
- 静态评审 outer-seal-file SHA256：`61ac10d46be82989aca702bae079510f872b50badad99926dd68e0972a68a8e9`。
- 结果 outer-seal-file SHA256：`6c180a8a5c97d5f05042a0534e68e179899c57e2e025db14ecf72eebced77286`。
- `docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`，本评审未修改该文件。

独立重算结果：合同 exact files **19/19 OK**；`preflight_identity_check.txt` 中的合同、评审、源码、filelist、runner 和 `docs/359` 全部为 OK；结果 `SHA256SUMS` 全量通过，`SHA256SUMS.seal.sha256` 通过。

## 三阶段执行与负向签名审计

| phase | compile.rc | sim.rc | 终端 PASS | assertion/fatal/watchdog/error |
|---|---:|---:|---:|---:|
| adapter attack unit | 0 | 0 | 1 | 0 |
| K1 vs K1×8 full regression | 0 | 0 | 1 | 0 |
| K8 vs K1×8 equal-bandwidth regression | 0 | 0 | 1 | 0 |

VCS 工具身份为 `V-2023.12-SP1_Full64`。三份 compile log 均显式解析合同指定的 top module，未检出 compile warning/error/fatal。三份 sim/assert 证据中未出现 `failed at`/`Offending`/`Fatal`/`watchdog` 或 assertion failure。

## channel-local fault 同拍语义已被定向触发

attack 终端行为：

`attack_classes=12 reset_cases=11 legal_response_on_request_fault=2 sticky_quiescent_checks=10 normal_requests=4 normal_responses=3 request_side_effect_violations=0 response_side_effect_violations=0`。

runner-gated 的 10 个 cover 全部非零：

| cover | match |
|---|---:|
| legal response + illegal request, same cycle | 2 |
| source-count/mask mismatch | 2 |
| zero mask | 5 |
| channel/bank mismatch | 1 |
| illegal response + legal request | 1 |
| pending drain + illegal request | 1 |
| response backpressure then attack | 1 |
| held response + attack + retirement | 1 |
| cut-through response + attack + retirement | 1 |
| sticky fault quiescence | 30 |

TB 还以显式 oracle 定向检查了 source-count-out-of-range 和 slice-out-of-range；两者各增加一次 `attack_classes`，并经 `request_only_attack_and_check` 确认请求不接收、不发 bank 事务、ledger 不变、sticky fault 后静默。因此功能性覆盖成立；其独立命名 cover 缺失仅影响可诊断性，列为 P2。

## 冻结 r2 周期与数值逐行保真

### K1 与 K1×8

| blocks | events | K1 cycles | K1×8 cycles | K1×8/K1 speedup |
|---:|---:|---:|---:|---:|
| 1 | 20 | 259 | 53 | 4.886792× |
| 2 | 41 | 737 | 133 | 5.541353× |
| 4 | 90 | 3153 | 499 | 6.318637× |
| 8 | 110 | 7569 | 1246 | 6.074639× |
| 1 | 0 | 14 | 14 | 1.000000× |

正工作负载四行聚合为 K1 `11718` cycles 与 K1×8 `1931` cycles，即 `6.068358×`。这是 **8 倍峰值带宽扩展** 的局部周期对照，不是稀疏机制同资源收益，更不是系统加速。

primary 终端证据：`clean_cases=10 reset_cases=2 protocol_attacks=4 numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 same_edge_release_violations=0`；`request_stalls=1363`、`response_injection_stalls=3143`、`result_stalls=47`、`raw_stalls=4509`，均非零。

### K8 与等带宽 K1×8

| blocks | events | K8 cycles | K1×8 cycles | K8 speedup |
|---:|---:|---:|---:|---:|
| 1 | 20 | 51 | 53 | 1.039216× |
| 2 | 41 | 131 | 133 | 1.015267× |
| 4 | 90 | 486 | 499 | 1.026749× |
| 8 | 110 | 1231 | 1246 | 1.012185× |
| 1 | 0 | 14 | 14 | 1.000000× |

正工作负载四行聚合为 K8 `1899` cycles 与 K1×8 `1931` cycles，即同带宽 `1.016851×`。这一数字只说明 R5 修复未破坏原有周期语义；未经面积/能量归一前，不支持 throughput/mm² 或能效 claim。

equal-bandwidth 终端证据：`clean_cases=10 reset_cases=2 protocol_attacks=4 numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0`；`request_stalls=375`、`result_stalls=45`、`raw_stalls=1165`，均非零。

## P0/P1/P2 发现

### P0 = 0

未发现身份漂移、seal 破损、RC/PASS 矛盾、周期行漂移、数值错误、协议副作用或 claim-boundary 越界。

### P1 = 0

对本次 VCS 局部功能准入未发现高风险缺口。

### P2 = 2

1. source-count-out-of-range 有 TB 内部定向 oracle，但无独立命名 SVA cover/runner 门；
2. slice-out-of-range 有 TB 内部定向 oracle，但无独立命名 SVA cover/runner 门。

两者均不会造成假 PASS，因为内部 `expect_true` 失败会阻止终端 PASS，且实际 `attack_classes=12`。它们仅使日志无法一眼区分 A5/A6，建议在下次非消耗性维护中增加命名 cover，不需要重跑本次 VCS 才能进入 DC 静态准入。

## DC 放行裁决

**允许**：修复当前 DC runner 已知的 raw-log 假阴性，然后为新 runner/Tcl/contract/VCS/review 身份创建新的独立 static admission。

**不允许**：直接启动当前 DC runner。其第 265 行对整份 `dc.log` 搜索任意 `TIM-209|OPT-150`，而 dc_shell 会回显 Tcl 中的 regexp token，使真实 0/0 成功路径也会返回 44。该问题是假阴性，不会误准入有环设计，但会使 canonical PASS 不可达。

新 DC static admission 至少必须重新绑定：

1. 修复后 DC runner/Tcl/filelist 精确 SHA；
2. 本 VCS result outer seal 和本独立 review outer seal；
3. K1/K8/K1×8 三点各自的 precompile `TIM-209=0`/`OPT-150=0` 门；
4. compile 只能从显式 PASS branch 可达，失败/中断树必须内外封存后 quarantine；
5. 新 static review P0=0 后才可单次 DC campaign。

## Claim boundary

准入：R5 channel-local fault 定向功能、sticky containment/reset recovery、r2 数值/协议/周期回归保真，以及进入 DC runner 修复和新 static admission 的资格。

未准入：`TIM-209=0`、combinational-loop-free、DC/STA、area、power、energy、throughput/mm²、完整 FC2、system speedup 或 DATE headline。

## 方法、限制与下一步

本评审只做只读证据审计，未运行 EDA，未修改 result/source/contract。VCS 不能证明综合后组合环不存在；该命题必须由三个 ARCH_MODE 各自的 DC precompile gate 证明。最小后续是先修复 DC runner 的第 265 行证据门，再交新 static hammer，不得跳过静态准入直接运行 DC。
