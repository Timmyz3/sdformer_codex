# M475 独立 DC hammer（2026-08-26）

## 裁定

**88/100，CONDITIONAL GO，无 P0。只允许进入 macro feasibility、Formality，以及带保守 macro/I/O/RC 模型的 PrimeTime。**

生产 receipt 没有作为数值来源。独立审计器从封存 report、mapped netlist 和 SDC 重新解析并交叉核对，共执行 **241 checks，0 mismatch**。M475 满足冻结的 logic-only DC pass gate，但绝不构成 physical 3 ns、macro PPA、M473 performance、全网或系统 headline admission。

## 独立复算结果

| 项目 | 独立结果 | 裁定 |
|---|---:|---|
| DC / technology | V-2023.12-SP3 / TSMC28 HPC+ | 身份通过 |
| setup / hold corner | ssg0p9v125c / ffg1p05vm40c min library | 通过 |
| clock | 3.000 ns，setup uncertainty 0.200 ns，hold 0.090 ns | 合同一致 |
| setup WNS | **0.0000 ns** | 仅合同边界通过，无物理余量 |
| hold WNS | 0.0100 ns | prelayout 通过 |
| timing paths | setup 100 / hold 100，0 violated | 通过 |
| constraints | max/min delay、cap、transition、fanout 共 5 类，0 violated | 通过 |
| logic levels | **70** | P1 |
| cells | 35,333 = 30,824 comb + 4,509 seq | netlist 逐实例复算一致 |
| cell area | 37,316.285232 um2 | 仅 cell area |
| macros / black boxes | **0 / 0** | 两个存储 cut 均未计入 |
| post `check_timing` | 检查 unconstrained endpoints，未发 warning/error | 通过 |

100 条 setup path 全部只在 `0.0000–0.0044 ns` 内；最差路径是 `issue_parent_id[4] -> scratch_reads_q_reg[21]`。它终止于 debug scratch-read counter，但 100 条近临界路径还包含 50 条 `parent_data_q` 路径，因此不能靠“只是 debug”消除物理风险。

面积报告明确给出 `Net Interconnect area: undefined`，timing path 使用 `ZeroWireload` 和 ideal clock。`clk_core` 有 4,509 loads，DC 四次发出 TIM-134，并用 fanout=1000 做延迟计算；没有 CTS、placement 或 extracted RC。因此 0.0000 ns 只能叫 **pre-macro logic-only contract pass**。

## P1

1. **零 setup 余量 + 70 logic levels。** 在 ZeroWireload/ideal clock 下已经顶到 0.0000 ns。PrimeTime 必须加入非零 interconnect/clock envelope 和真实或保守 macro timing；任何负 slack 都推翻 fused timing assumption，不能暗中放宽时钟或增加 issue bubble。
2. **“144-byte scratch”口径必须纠正。** `96×12=1152b=144B` 是每个 word 的宽度；`ROW_BITS=6` 是 64 words，所以总容量是 **9 KiB**。resident psum 是 `96×19=1824b=228B/word`，若同为 64 行则是 **14.25 KiB**。超宽 1R1W 的 banking、mux、decode、tCQ/setup、面积和能量均未进入 M475。
3. **4 个 VER-318。** RTL 102/104/220/221 的 signed-to-unsigned part selection 是 final/prefix truncation 点。VCS directed 测试和数值界不能替代 RTL↔gate 全状态等价；必须跑 exact-SHA Formality。
4. **PPA 代表性。** 37,316 um2 包含 debug counters 和宽 pipeline registers，却不含约 9 KiB scratch、潜在 14.25 KiB psum、布线与时钟树。它既不是完整物理 PPA，也不是纯生产数据通路面积。若删除 instrumentation，必须建立新 RTL 身份并重跑 VCS/DC/Formality。

另有一个 P2 证据问题：`input_sha256.txt` 只记录 9 个直接输入，没列 nested VCS/hammer seals、docs/359、dc_shell binary；当前 runner 确实在运行前检查它们，本审计也全部复验通过，但 runner 自身没有进入冻结合同或结果 seal。下轮应补全。

## Warning 与约束审计

`dc.log` 有 0 Error/Fatal、12 Warning：4×VER-318、4×UISN-40、4×TIM-134。UISN-40 是 DesignWare synthetic library 加载；TIM-134 全部指向 postcompile `clk_core` 4,509-load ideal-clock net。precompile `check_design` 的 feedthrough/shorted-output/unused-cell lint 经优化后，postcompile `check_design` 干净。

mapped SDC 有 1 clock、2 uncertainty、4,153 input-delay、3,166 output-delay、3,166 output-load、1 reset false-path；`1 + 4153 + 3166 = 7320`，与报告总端口数一致。postcompile `check_timing` 执行了 unconstrained-endpoint 检查且没有 warning，但这不等价于 macro/physical timing 完整。

## 严格 admission boundary

以下仍全部为 false：

- physical 3 ns timing
- scratch / resident-psum macro area, timing, power
- M473 full-controller RTL 与 performance admission
- power、energy、paper-ready PPA
- full-network、system speedup、DATE headline

下一关顺序：Formality；冻结两类 memory 的容量与 banking 并做 macro feasibility；再以真实或保守 macro/route/clock 模型跑 PrimeTime。即使三项都过，仍需新的独立裁定，不能自动把 M473 的 `PASS_M473_CPU_DSE_NO_GO` 改成性能 GO。

## 复核入口

```bash
python3 results/m475_independent_hammer_review_r1_20260826/audit_m475_independent.py \
  --root .
```

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
