# T24：FPGA（Kria KV260）综合/时序——TCAS-II FPGA 口径一期（2026-09-16）

> 背景：T23 评估方法学定案（FireFly 系口径 = Vivado 报告，板级实测非 TCAS-II
> 硬性要求）。本机 Vivado 2025.1 Standard 版不支持 xczu9eg（ZCU102），改用
> **Kria KV260（xczu5ev-sfvc784-2-e）= FireFly-S/T 同器件**——对比锚点更好。
> 口径：out-of-context 综合 + 250 MHz（4 ns）目标时钟 + report_timing_summary。
> RTL 与 T15 相同（cert_gate_bitl_synth / fx_gate_synth / plane_ser，常熟烘焙、
> thr 640b 外置行口）。脚本 `t24_fpga/synth_zcu102.tcl`。

## 1. 结果总表（KV260 xczu5ev-sfvc784-2-e，OOC 综合）

| 核 | LUT | FF | DSP | WNS @4ns | 估计 Fmax | 拍/组 | 吞吐（组/s@Fmax） |
|---|---:|---:|---:|---:|---:|---:|---:|
| **cert_gate_bitl（C1 门核）** | 5,905（5.04%） | 1,936 | **0** | +0.995 ns | **~333 MHz** | 4.15–4.24 | ~79 M |
| fx_gate（FX 字串行基线） | 316 | 509 | **10** | +0.302 ns | ~270 MHz | 11 | ~24.5 M |
| plane_ser（生产者） | 496 | 264 | 0 | +2.613 ns | ~721 MHz | — | — |
| **C1 链（producer+consumer）** | 6,401 | 2,200 | **0** | — | ~333 MHz（链瓶颈在门核） | 4.2 | ~79 M |

**读法**：
1. **吞吐**：C1 链 4.2 拍/组 @333 MHz ≈ 79M 组/s，vs FX 11 拍/组 @270 MHz ≈
   24.5M 组/s → **同器件 3.2× 吞吐**；位传输（24b×11 vs 10b×4.2）少 6.3×。
2. **DSP**：C1 全链 **0 DSP**（FireFly-S 同款卖点）；FX 基线 20 个 48b 乘法
   被 Vivado 打包进 10 个 DSP48E2。
3. **LUT↔DSP 交换（诚实披露）**：C1 的门核算术（dot10 子集和 + 48b 区间数据
   通路）全部落 LUT，而 FX 的 10 个 16×24 乘法器骑 DSP 几乎不占 LUT——与
   Nangate45 口径（C1 37,168 < FX 41,482 µm²）方向相反。
   **（T25 实测修正）** 原稿把 5,905 LUT 归因于"12.8kb ROM 落 distributed RAM"，
   实测**证伪**：ROM 版 `LUT as Memory = 0`——子集和表根本没进分布式 RAM，
   而是被综合成 32:1 选通逻辑叠加 48b 加法/移位通路。LUT 的去处是算术逻辑
   本身，不是存储映射。详见 §3 与 T25_REPORT。
4. 绝对规模：5.9k LUT = KV260 的 5%（FireFly-S 用 84k LUT）——门核是模块级
   资产，不是整网。

## 2. 过程记录

- plane_ser 首版 `in_ywords[gi*24 +: 24][23]`（part-select 再索引）Verilator
  接受、Vivado 拒绝（Synth 8-2599）；改为中间 wire 后 **Verilator 回放
  4 traces 仍 EXACT**（verify_plane.py，功能等价零差）。
- Vivado Standard 版 part 范围：最高 xczu7ev（ZCU102 xczu9eg 不可用）；
  ZCU102 口径可在 eventflow_zcu102 侧远端 Vivado 补做（同 RTL 同脚本）。

## 3. k=4 掩码缩 FPGA 数据通路——**T25 已实测（原预测被证伪，见 T25_REPORT）**

原假设：k=4 时 dot10 退化为 40 项选位加法树，可**大幅**压低 LUT（"数百 LUT
量级"）。**T25 实测否定了这个量级**（同器件同口径，4 ns 目标）：

| 门核变体 | LUT | FF | DSP | WNS@4ns | Fmax |
|---|---:|---:|---:|---:|---:|
| ROM 版（t15 综合版） | 5,905 | 1,936 | 0 | +0.995 | ~333 MHz |
| 加法树，稠密 A（k=10） | 6,451 | 1,938 | 0 | +1.394 | ~384 MHz |
| **加法树，k=4 掩码 A** | **5,030** | 1,756 | 0 | +1.053 | ~339 MHz |

1. **ROM 不是瓶颈**：稠密加法树比 ROM 版还多 546 LUT（+9.2%）——因为子集和
   表本来就没进分布式 RAM，早已被综合成逻辑。
2. **k=4 是真实但温和的收益**：vs ROM 版 **−875 LUT（−14.8%）**，vs **同结构**
   稠密树 **−1,421 LUT（−22.0%）**，FF 亦降 1936→1756，且时序更快
   （WNS +0.995→+1.053）、0 DSP 不变。
3. 诚实结论：证书感知稀疏化在 FPGA 资源轴上**不是**"消掉 C1 唯一劣势"的
   银弹，而是 **~15% 的 LUT 节省**，代价 +1.0% 粗头 AEE（T21f）——与它在
   供数轴上 −18.0% 的收益并列，是一个**联合（训练+RTL）旋钮**。

## 4. 二期进度

1. **功耗**：VCD 已由 Verilator --trace 按真实激励生成（20,000 组，4 ns 时基），
   但 **Vivado 的 VCD/SAIF 活动标注无法用于 OOC 综合网表**——`read_vcd`
   实测 0% 网表匹配（三种 strip_path 全试过；Verilator 的 RTL 名与综合后
   网表名——如 `nmsk` vs `nmsk_reg[0][47]`——不对应），且本机 Vivado 2025.1
   的 XSim **不支持 `$toggle_report` 等 SAIF 任务**，无 VCD→SAIF 转换工具。
   故 `report_power` 报的是**向量无关（vector-less）默认活动模型**结果，
   仅作参考（见 §5），**不作为能耗结论**。
2. **k=4 加法树数据通路**：已完成（§3、T25_REPORT）。
3. （可选，加分项）上板：KV260/ZCU102 回放 capture + 片上周期计数器对拍
   Verilator（ESTU 级即够，无需逻辑分析仪）。

## 5. 功耗（向量无关口径，仅参考）

| 核 | 总功耗 | 动态 | 静态 | 拍/组@250MHz | 动态能量/组 |
|---|---:|---:|---:|---:|---:|
| cert_gate_bitl_synth | 0.583 W | 0.241 W | 0.342 W | 4.211 | 4.06 nJ |
| fx_gate_synth | 0.379 W | 0.039 W | 0.341 W | 11 | 1.72 nJ |

**重要限定**：动态功耗的 93%（C1）/ 93%（FX）差异来自**门数**，而向量无关模型
对每个非时钟网默认按固定翻转率估算，**忽略本设计的使能门控**（寄存器仅在
sop/平面有效拍更新，实测组均仅 4.2 拍工作）。因此这张表**偏向门数少的 FX**，
不能读作"C1 更费能"。**C1 的真实优势轴是位传输**（每组件 24b×11=264b vs
10b×4.2=42b，少 6.3×，见 §1 读法 1）——链路能耗才是系统级主项，
本轮**未测**，不做结论。

