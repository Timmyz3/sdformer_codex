# 3xIndependent32 等并行度结构基线 RTL 闭环

## 1. 目的

本轮不再用 `3 x HATF32 area` 算术推导，而是实现真实三实例 wrapper，与 HATF96-Central 在相同 96 个 product lane 下比较 replay/decoder/control 的复制代价。

比较边界为 decoder + projection backend，不复制上游 Builder：

~~~text
3xIndependent32
  engine0: replay/decoder -> 32-lane weight/product -> 2-bank Acc
  engine1: replay/decoder -> 32-lane weight/product -> 2-bank Acc
  engine2: replay/decoder -> 32-lane weight/product -> 2-bank Acc
~~~

三套引擎只共享 clock/reset。tile/head、resident/IPD/RAW payload、cache fill、weight、bias、final、done、error 和计数接口全部独立 packed 暴露，没有 payload 广播、decoded-term 广播或中央 weight join。

## 2. RTL 与测试

新增：

- `rtl_hitflow/gatestack_three_independent32_projection_top.sv`；
- `tb_hitflow/tb_gatestack_three_independent32_projection_top.sv`；
- `sim_hitflow/run_gatestack_three_independent32_projection_checks.sh`；
- `dc_handoff/scripts/run_gatestack_three_independent32_nangate45_mapping.sh`；
- `scripts/summarize_gatestack_equal96_mapping.py` 及单元测试。

小规模 TB 同时运行三个不同 tag/tile/channel/gate/weight/bias 的 resident 事务，并给 engine1/2 施加不同 weight 与 final 反压。结果：

| 项目 | 结果 |
|---|---:|
| 独立事务 | 3 |
| product lane | 96 |
| 逐元素比较 | 384 |
| mismatch | 0 |
| cross-talk | 0 |
| Icarus | PASS |
| Verilator + projection SVA bind | PASS |
| Yosys check | PASS |
| Erie | 0 error / 0 warning |

Icarus/Verilator 周期分别为 48/49。该周期只用于小规模结构验证，不代表 H67 真实 S0-S3 wall time。

## 3. 同库开放逻辑映射

两种设计使用同一 19-file RTL 集合、同一 `NangateOpenCellLibrary_typical.lib` 和相同 Yosys 映射流程：

| 结构 | decoder实例 | product lane | logic area | cells | `$mem_v2` |
|---|---:|---:|---:|---:|---:|
| HATF96-Central | 1 | 96 | 203921.452 | 148134 | 3 |
| 3xIndependent32 | 3 | 96 | 270665.640 | 190696 | 9 |

在该开放逻辑代理下，Central96 相对真实三路独立 wrapper：

- logic area 降低 `24.659%`；
- mapped cell 数降低 `22.319%`。

这比旧的算术 `3 x HATF32` 更可信，支持“共享一次 replay/decoder/control”具有逻辑面积价值。

## 4. 不能越界的结论

当前结果仍不能证明 HATF96 已成为 DATE 主架构贡献：

1. `$mem_v2=3/9` 不计入 logic area，且每个 memory 的宽度和用途不同，不能写成存储面积降低三倍；
2. Central 的行为 Acc 是 96-wide，正式物理比较必须拆为与 Independent 相同的六个32-lane SRAM macro；
3. Independent 需要三个 slot replay read client，当前 wrapper完整暴露三路接口，但未将真实 typed-slot 三读口面积和能量接入；
4. 当前只跑小规模三个 resident 事务，未运行 H67 S0-S3 真实 payload和wall time；
5. 无 SDC、STA、SAIF、SRAM macro、布局布线或绝对功耗；
6. Central 的中央 768-bit weight与宽product互连物理代价仍未反映。

因此当前可写的唯一架构证据是：

> 在固定96个product lane的同边界开放逻辑映射中，decode-once Central 结构相对三套独立 decoder 减少约24.7%的映射逻辑面积；存储、时序、功耗和真实trace性能仍待同约束闭环。

## 5. 对 DCTF 的指导

`3xIndependent32` 现在成为 DCTF 的必须对照。DCTF 应保留 Central 的单 decoder，同时把三路 weight/product/Acc 做成 bank-local：

- 如果 DCTF 与 Central 周期接近，并保留相对 Independent 的共享面积，同时改善中央宽网时序/能量，才形成架构贡献；
- 如果 DCTF 只比 Central 多一个 command FIFO，没有物理或能量收益，则淘汰；
- 如果真实三读口 slot 代价抵消 24.7% logic 节省，则 HATF/DCTF 降级为参数选择。

晋级门槛保持：相对 `3xIndependent32` projection EDP 改善至少 `15%`，或总能量和总面积各改善至少 `10%`。

## 6. 复现

~~~bash
bash sim_hitflow/run_gatestack_three_independent32_projection_checks.sh
bash dc_handoff/scripts/run_gatestack_three_independent32_nangate45_mapping.sh
PYTHONPATH=scripts python3 -m unittest \
  scripts.test_summarize_gatestack_equal96_mapping
~~~

机器可读结果位于 `results/gatestack_equal96_mapping_20260720/report.json`，中文表位于同目录 `report.md`。
