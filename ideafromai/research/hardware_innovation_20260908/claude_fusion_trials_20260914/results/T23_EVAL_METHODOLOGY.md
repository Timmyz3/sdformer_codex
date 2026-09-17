# T23：基准性能评估方法学 + TCAS-II 口径调研（2026-09-16）

> 任务（用户指令）：学 Prosperity/Phi/FireFly/GustavSNN 等标杆论文的性能评估方法；
> 找 TCAS-II 对标论文学评估口径；找"纸面更优"的合法 trick；考虑 TCAS-II 的
> FPGA 验证要求。
>
> 方法：四路并行精读（GustavSNN HPCA26 / FireFly-S TCAS-I'24 + FireFly-T /
> Bishop+Phi+Prosperity ISCA-HPCA'25 / TCAS-II 2022–2026 对标检索），
> 本文档为综合定案。四份原始报告见本文件生成记录（代理会话）。

## 1. 六篇标杆 + TCAS-II 对标：评估口径总表

| 论文 | 载体 | 指标 | 硬件数字来源 | 功耗来源 | FPGA 上板 | 精度口径 |
|---|---|---|---|---|---|---|
| GustavSNN (HPCA26) | ASIC 28nm FDSOI | **有效 GOPS/W**（跳过后操作数/PE+buffer 功耗）、延迟、Dr/Dw | DC 综合 @1GHz + ICC2 P&R（1.34mm²）+ CACTI 7.0 | PrimePower（仿真翻转率驱动） | 无 | "a little drop" 未量化 |
| FireFly-S (**TCAS-I**'24) | FPGA KV260 | **FPS/W**、LUT/BRAM（零 DSP 卖点） | Vivado 综合+实现 @333MHz | **Vivado 报告**（非实测） | 无实测 | 分级表 0.86pp |
| FireFly-T | FPGA KV260 | **GOP/s（密集等效）**、GOP/s/W、GOP/s/DSP | Vivado @300MHz | Vivado 报告 | 无实测 | 允许 ~1pp |
| Bishop (ISCA'25) | ASIC 28nm | mJ/图、EDP、逐层归一化加速 | DC 综合 + **基线同资源复现综合** + STONNE 仿真 + CACTI | 解析（CACTI+综合） | 无 | ECP 反升/−0.13pp |
| Phi (ISCA'25) | ASIC 28nm | GOP/s、GOP/J、GOP/s/mm²（**同频同工艺全表**） | DC 综合（0.662mm²）+ CACTI + DRAMsim3 | 解析 | 无 | PAFT "轻微下降" |
| Prosperity (HPCA'25) | ASIC 28nm | 同 Phi 三件套 | DC 综合（0.529mm²）+ CACTI + DRAMsim3；A100 实测 | 解析 | 无 | 无损 iso-accuracy |
| **ESTU (TCAS-II'25)** | FPGA iCE40UP5K | 资源、µJ/推理 | **OSS CAD Suite（Yosys+nextpnr）@21MHz** | **示波器+分流电阻实测**（3.76mW） | 整 SoC 上板 | FP 87.21→量化 86.97 |
| DeltaTrack (TCAS-II'25) | ASIC 28nm | fps、mJ/frame、归一化 2.26–4.66× | **仅 28nm post-layout 估计** | 解析 | **无，照样中稿** |
| MEA AxC (TCAS-II'26) | — | 动态功耗降 86.6% | **纯综合** | 综合报告 | 无 | — |

## 2. TCAS-II 门槛 norm（定案）

1. **FPGA 板级实测不是硬性要求**。DeltaTrack 只有 post-layout 估计、MEA AxC
   只有综合、FireFly-S（TCAS-I 正刊）功耗取 Vivado 报告——**纯"综合+实现+
   零差仿真"口径可投**。ESTU 的示波器实测是加分项非门槛。
2. **开源流程可发**：ESTU 全链 OSS CAD Suite（Yosys+nextpnr）发 TCAS-II——
   我们的 yowasp-yosys 流程有直接先例；但**本机有 Vivado 2025.1**（/opt/vivado），
   用 Vivado 口径更主流（FireFly 系范式）。
3. **短文评估节典型结构**（ESTU 模板，5 页容 6 表 4 图）：资源表 → 吞吐/时延
   （解析模型可代实测）→ 功耗 → 一张 SoTA 对比大表（平台/精度/功率/能效）。
4. **时延可用解析模型**（ESTU 式(3)：T=Σ稀疏度×数据量/吞吐）——我们的 T16
   解析模型（拍比=f(ρ,msb)）正是同款动作，且有 480 traces 实测+松弛 1.22 背书。
5. **光流任务口径直接参照 SpiDR**（65nm 芯片，DSEC + AEE）——我们已有
   valid825 AEE 全套。

## 3. "纸面更优"的合法技巧清单（标杆实证，标注我们是否采用）

| # | 技巧 | 出处（受益数字） | 我们采用？ |
|---|---|---|---|
| 1 | **密集等效吞吐计数**：分子按稠密算子数，跳过的操作白送 | FireFly-T（3029 GOP/s，实际只执行 ~1/4） | **用**：FX 基线 24 拍/组 vs C1 4.2 拍 → 等效吞吐 5.7×（同端口）；k=4 再 ×1.22。诚实脚注注明 executed vs dense-equivalent 双口径 |
| 2 | **Vivado 报告功耗代实测**（分母不含板级静态/DRAM） | FireFly-S/T（全部能效数字） | **用**：Vivado implementation 报告（ZCU102），脚注声明口径 |
| 3 | **相对倍数做标题、区间下限取最弱基线** | GustavSNN 11.8×(naive GP)/1.43×(SOTA)；FireFly ×202 | **部分用**：对 FX 基线（同端口同状态同背压，公平）报倍数；对文献只划界不虚比 |
| 4 | **解析时延模型代实测** | ESTU 式(3) | **已有**：T16 + T17/T19 全网验证（比 ESTU 强：有 480 traces 零差背书） |
| 5 | **"满足实时约束"重构叙事**（峰值低但唯一 mW 级） | ESTU 3.76mW | **可仿**：事件相机前端低功耗叙事（位传输 6.3× 少 → 供数侧能耗主导下降） |
| 6 | **基线 handicap / 同资源复现** | GustavSNN 强制 ex-situ；Bishop 等面积复现 PTB | **不用 handicap**；用 Bishop 式公平（FX 基线同端口同位宽复现，T5 已做 RTL 级） |
| 7 | **零 DSP / 0 乘法器叙事** | FireFly-S | **用**：T10 后门核 0 乘法器（LUT+移加），与 FireFly 同款卖点 |
| 8 | **消融阶梯拆算法/架构收益** | Bishop（BSA/ECP 分离）；Prosperity（2.28→2.16→1.49） | **用**：静态掩码→微调→RTL 数据通路三级阶梯（T21 表已现成） |
| 9 | **开销-收益盈亏平衡不等式** | Prosperity（浮点加法=45 次 TCAM 操作，ΔS=13.35% vs 平衡点 4.4%） | **可仿**：证书判定 1 拍/组的开销 vs 省 ~20 拍/组（5% 开销换 83% 供数削减） |
| 10 | **跨器件混比 + 几何平均** | FireFly 系、GustavSNN | **不用**（诚实口径，同节点同频表内对比 Phi 式） |
| 11 | **等精度声明：无损** | Prosperity iso-accuracy | **更强**：逐判决零差（Verilator 320 万判决）+ f14 部署整数链基准（比所有人的"无损"口径都硬） |

## 4. 我们的评估装配方案（论文评估节设计）

**指标三件套**（对标 Phi/Prosperity/FireFly-T）：
1. **吞吐/时延**：拍/组（4.15–4.24 实测 RTL vs FX 11–24）；密集等效吞吐
   =dense 算子/时间；解析模型供设计曲线。
2. **能效**：分母 = Vivado 功耗报告（FPGA 口径）或 CACTI+开关活动（若 ASIC
   口径）；分子 = 密集等效操作。供数位传输 6.3× 少 → 能耗主要来自供数侧的
   论证链（Sparsity Tax 教训：同端口计费）。
3. **资源**：LUT/FF/BRAM/DSP（预期 0 DSP）+ Nangate45 面积交叉
   （37,168 vs 41,482 µm²）。

**精度口径**（最强项）：valid825 AEE（粗头 0.9982@k=4 vs NB0 门 1.4454、
原基线 1.0814）+ 逐判决零差 + Pareto 表（k=4/5 vs 对照）。

**对比表**：同频同工艺/同器件表内对比（Phi 式纪律）：C1 vs FX 基线（自有、
RTL 级复现）；文献侧按五档相关工作划界（AEU26/EITCE26/TVLSI26/MINT/ECHO/
BitSET/BitL/FlexSpIM），标注"域/终止类型/精度口径"三列，不做跨器件倍数虚比。

## 5. 缺口清单 → 行动项

| 缺口 | 严重度 | 行动 |
|---|---|---|
| **FPGA 综合资源/Fmax**（现在只有 Nangate45 面积，无时序收敛数字） | 高（TCAS-II 惯例表的第一行） | **T24**：Vivado 2025.1 综合 cert_gate_bitl+plane_ser 上 ZCU102（xczu9eg），出 LUT/FF/BRAM/Fmax + 功耗报告 |
| FPGA 上板（可选加分） | 中 | T24 二期：capture 回放 + 片上周期计数器对拍 Verilator；ESTU 级即可（无需逻辑分析仪） |
| 能量模型参数表（pJ/bit） | 中 | Vivado 报告口径可免；若走 ASIC 口径补 CACTI 45nm 参数表 |
| FX nomap 公平口径综合 | 低 | 在跑（T15） |
| 延迟数字 | 低 | Vivado 时序报告直接给 Fmax；拍数轴已有 |

## 6. 一句话结论

TCAS-II 对 FPGA 的真实要求 = **"综合+实现+零差仿真"可投，上板是加分**；
我们的最强差异化不是堆倍数，而是**别人没有的"逐判决零差 + AEE 全口径 +
训练旋钮"三合一**，再叠 FireFly 式密集等效吞吐与零乘法器叙事即可。
