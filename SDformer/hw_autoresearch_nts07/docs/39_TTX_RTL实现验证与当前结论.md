# TTX RTL 实现验证与当前结论

> **2026-07-11 结论补充**：下述 PASS 证明历史 TTX RTL 与其软件公式一致，不证明 strict symmetric ATLIF 或 no-K-carrier。后续 H63-H65 统一 attention 候选均因 DSEC 精度/活动率失败而未进入 RTL；完整判定见 `40_H63对称ATLIF无GateK注意力探索.md`。

**日期**：2026-07-10  
**对象**：`rtl_ttx/` 完整模块级原型

---

## 1. 新增实现

```text
rtl_ttx/
  ttx_tx_score_q7.sv
  ttx_zero_k_class_score_q7.sv
  ttx_exp2_lut_q8.sv
  ttx_ceil_log2_u32.sv
  ttx_row_engine.sv
  ttx_late_gate_accum.sv
  ttx_descriptor_scheduler.sv
  ttx_attention_top.sv
  filelist.f

tb_ttx/
  tb_ttx_row_engine.sv
  tb_ttx_scheduler.sv

sim_ttx/
  run_iverilog.sh
  run_verilator_lint.sh
  run_yosys.sh
  run_all_checks.sh

scripts/
  ttx_zaf_reference.py
```

---

## 2. 验证结果

一键命令：

```bash
./sim_ttx/run_all_checks.sh
```

### 2.1 Directed simulation

结果：

```text
PASS: TTX row engine, ZAF folding, and FGK late-gate tests passed
PASS: TTX 12-block descriptor scheduler issued 6720 rows
```

覆盖：

1. TX score；
2. dense row Shiftmax；
3. ZAF K-zero class folding；
4. folded 与 dense 有效 token gate 一致；
5. K-zero token sparse output；
6. 全 K-zero row 无输出并正常 done；
7. output backpressure；
8. threshold metadata；
9. FGK late-gate 代数等价；
10. 12 blocks / 6720 rows descriptor schedule。

Icarus 输出的 `constant selects` 和 `unique ignored` 是 simulator capability 提示，不是 DUT failure。

### 2.2 Verilator lint

`ttx_attention_top` 全 filelist：

```text
PASS，无 warning
```

testbench lint 对 timescale、clock blocking assignment 等仿真风格 warning 做了局部豁免，不影响 DUT lint。

### 2.3 随机等价参考

`scripts/ttx_zaf_reference.py`：

| 项 | 结果 |
|---|---:|
| random rows | 2000 |
| tokens/row | 162 |
| dense vs folded mismatch | 0 |
| FGK mismatch | 0 |
| 随机模型 K-zero ratio | 55.27% |
| exp transaction reduction | 54.15% |

随机概率模型只是验证与保守 sanity check，论文收益应使用真实 TTX profiling。

### 2.4 Yosys generic synthesis

所有 top：

```text
Found and reported 0 problems
PASS: Yosys synthesis/check completed for TTX tops
```

主要 generic cell 数：

| Top | cells |
|---|---:|
| `ttx_tx_score_q7` | 131 |
| `ttx_late_gate_accum` | 65 |
| `ttx_descriptor_scheduler` | 75 |
| `ttx_row_engine` hierarchy | 3016 |
| `ttx_attention_top` hierarchy | 3096 |

这些是 Yosys generic cells，不是 gates、um² 或 DC area。

数组在当前脚本中被映成寄存器，因此：

```text
Number of memories: 0
```

不能拿该数字报最终 SRAM/ASIC 面积。

---

## 3. TTX 实测 ZAF profiling

对象：

```text
date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8
checkpoint_epoch2.pth
valid40
```

输出：

```text
neuron_experiments/H9_bipolar_self_attention/results/
nts11_hardware_p0_profiles/ttx_ep2_valid40_zaf_20260710/
```

加载审计：

| 项 | 结果 |
|---|---:|
| ATLIF installed | 105 |
| Shiftmax/TTX attention | 12 |
| checkpoint missing/unexpected | 0 / 0 |
| samples | 40 |
| attention records | 480 |

分 stage：

| Stage | K-zero token | active entries/row | fold classes/row | TTB2 empty |
|---:|---:|---:|---:|---:|
| S0 | 0.7815 | 35.40 | 2.99 | 0.2837 |
| S1 | 0.9741 | 4.20 | 1.39 | 0.7435 |
| S2 | 0.9391 | 9.87 | 2.47 | 0.6316 |
| S3 | 0.8945 | 17.09 | 2.25 | 0.6145 |

按真实 6720 rows/frame 加权：

| 指标 | 结果 |
|---|---:|
| K-zero token ratio | **88.15%** |
| active entries/row | **19.20 / 162** |
| fold classes/row | **2.43 / 33** |
| dense exp transactions/row | 324 |
| ZAF exp transactions/row | **40.82** |
| exp transaction reduction | **87.40%** |
| sparse output entry reduction | **88.15%** |

该统计直接支持 ZAF，而不是沿用 H60/NTS 的 proxy。

---

## 4. 已发现并解决的问题

### 4.1 Testbench handshake race

最初 `cfg_start/start_frame` 在 posedge 用 blocking drive，DUT 与 TB 存在调度 race，仿真死等。改为 negedge drive、跨 posedge 保持后，两套 TB 正常结束。

### 4.2 非 2 次幂 memory depth

162-entry 和 33-entry 数组使用 8/6-bit 地址时，Yosys memory expansion 为不可达地址生成 undriven mux，`check` 报 5264 problems。

修复：

```text
active store physical depth = 256
class histogram physical depth = 64
```

修复后所有 top 为 0 problems。这也符合 SRAM compiler 常见的 2 次幂深度组织。

---

## 5. 当前还不能声称的内容

1. 还没有 TTX PyTorch row golden vector 对 RTL 的 bit-accurate checker；
2. 还没有 DC、SDC、工艺库、WNS/TNS；
3. 还没有 SRAM macro/CACTI；
4. 还没有 SAIF/PT-PX 功耗；
5. 还没有 FGK 接入真实 projection RTL；
6. 还没有完整 encoder sparse-MAC、downsample、skip、decoder RTL；
7. 还不能声称完整 optical-flow ASIC；
8. ZAF novelty 仍需更广泛 patent/IEEE/ACM 检索。

---

## 6. 下一步优先级

### P0：论文功能可信度

1. 导出 TTX 多 stage/head 的 Q/K/score/gate golden rows；
2. 加 Python/RTL file-driven checker；
3. 验证 raw-Q7 center cancellation 与 PyTorch float-center-then-INT8 的误差；
4. 把 FGK 接到一版真实 projection lane；
5. 增加随机 ready/valid、reset-mid-row、early-last、边界配置测试。

### P1：硬件主表

1. active store 改 SRAM wrapper + synchronous read pipeline；
2. nonempty-class bitmap walk，取消固定 33-cycle scan；
3. dense/ZAF 两个配置做同约束 DC 对比；
4. SRAM compiler/CACTI 计入面积功耗；
5. TTX valid workload 生成 SAIF；
6. DC + PT-PX + LEC。

### P2：系统扩展

1. event SRAM/NoC；
2. ATLIF temporal mixer cluster；
3. sparse projection/MLP；
4. residual/downsample/skip controller；
5. decoder 边界或外部接口。

---

## 7. 当前结论

TTX 已从“旧 H60 在 mu=0 时可兼容”的状态，推进到独立的模块级 RTL 主线：

```text
TTX-only score
+ exact ZAF-Shiftmax
+ factorized gated-K
+ 12-block descriptor scheduling
+ directed/random/synthesis verification
+ TTX valid40 workload evidence
```

它已经足够支撑下一步 golden/DC 工作，但仍不能直接作为最终 ASIC PPA 结果投稿。
