# M1760 independent source hammer: M1753 C2 three-axis mapped energy

结论：**PASS，98/100，P0/P1/P2 = 0/0/1。允许作者创建精确绑定本 review 的 M1761 一次性 release；本评审未启动也不直接授权 EDA。**

## 三轴与负载边界

M1753 绑定 K1、K8 和等带宽 K1×8 三个 mapped netlist，每轴都重放同五组 public-port directed case，接收 source 为 20/41/90/110/0，合计 261，时钟为 3 ns。runner 要求 3 次独立编译、15 次 simv、15 个 DUT-only SAIF 全部通过后，才可启动 15 次 PTPX；任何部分轴都不可引用。这些负载仅是 `DIRECTED_COMPONENT_NOT_PRODUCTION`，不是 trace、system 或 production energy。

## 功耗口径

SAIF 只覆盖 `tb_m1684_c2_m1609_fresh_mapped_production_energy.core.dut`，PrimeTime 必须对 whole mapped component 报告 cell internal、net switching、cell leakage 和 total power，禁止 cell-collection 减法或 add-back。结果是 logic-only pre-macro，明确不包含 weight SRAM、testbench memory model、IO PHY、CTS 和 post-layout parasitics。

## 强制联合披露

K8 相对等带宽 K1×8 的固定周期比是 **1.0167276529×**，固定吞吐/mm² 比是 **4.5627200965×**。两者必须在同一表格和同一句中出现；K8 相对单 K1 的数字不得作为 headline。

## 独立打铁

源自检通过，作者单测 7/7 通过。独立 hammer 完成 50 项检查，15/15 个负向 mutation 均被拒绝，包括 K1 cycle/axis/source 替换、重复 PASS、缺失 binary-clean PASS、三轴 Cartesian 缺失/重复、cycle/source 分母漂移、功耗分解错误、功耗字段缺失/重复以及重复/非有限 JSON。审阅过程 VCS/simv/SAIF/PTPX/EDA 均为 0，且没有网络访问。

## 非阻断建议

复用的 M979 TB 内部 `expected_cycle()` 没有 AXIS_ID=0 的 K1 分支，因此 TB 自身不检查 K1 周期。这不构成发布缺口：每次 sim 后立即运行的 exact-SHA M1753 checker 必须看到唯一且精确匹配 K1 axis/case/events/cycles 的 PASS 行，然后才增加 SAIF 计数；15 个 checked SAIF 不齐则 PTPX 不会开始。建议以后的 successor TB 补上 K1 内部锚点，但不应改写已审的 M1753 身份。
