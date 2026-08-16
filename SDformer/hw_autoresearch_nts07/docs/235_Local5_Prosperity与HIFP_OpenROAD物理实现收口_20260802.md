# Local5 Prosperity 与 HIFP OpenROAD 物理实现收口

> 日期：2026-08-02  
> 范围：Motion/Local5 双线、Prosperity 正式基线、HIFP 投影子系统 OpenROAD 分块物理实现  
> 证据等级：`[prof]`、`[rtl]`、`[open-pnr][代理]`、`[模型]`  
> 结论边界：本报告不是目标工艺 DC/PrimeTime/PTPX 签核，也不是芯片 PPA。

## 1. 直接结论

1. **本机能完成的 OpenROAD 阶段已经跑通。** 四个 HIFP 叶级候选在同一
   Nangate45、5 ns、45% 初始利用率和固定随机种子下均完成综合、布局、CTS、
   详细布线和寄生后 STA；最终 detailed-route DRC 均为 0，setup/hold 违例均为 0。
2. **当前仍不能叫物理 signoff。** 四个块分别残留 7、11、3、1 个 max-cap
   违例；未使用目标 SRAM 宏、目标 PDK、真实 SAIF、MMMC、EM/IR 和 LVS。
3. **PPDI+IBF 的物理趋势仍为正，但比旧 Yosys 结论弱。** 真实 Motion 回放周期
   从 53910 降到 45735，提升 1.179x；分块组合面积增加 5.28%，面积归一吞吐为
   1.120x。旧开放逻辑映射得到的 1.163x 因低估 PPDI 布线和控制代价而偏乐观。
4. **IBF 是当前更稳的硬件机制。** 单独启用 IBF 时周期提升 1.072x，组合面积反而
   降低约 0.48%，面积归一吞吐为 1.077x。PPDI 单独只有 1.033x 的面积归一吞吐，
   必须依靠多 trace 稳定性和目标工艺功耗结果才能晋级强贡献。
5. **Motion 仍是硬件主线，Local5 保留算法竞争线。** Local5 的 FCSR 相对强
   Stripe 基线仅约 1.017x，XBF-T8 为负结果；Prosperity bit-plane 总周期是
   product-sparsity 的 1.791x。当前没有证据支持为 Local5 新造一套投影 RTL。

## 2. Local5 正式 profile 与回放结论

### 2.1 数据完整性

Local5 post-G0 profile100 已完成，ordered trace 使用 v2 schema。回放器已增加：

- v1/v2 schema 兼容；
- v2 dataset index、hash 和记录数严格核验；
- schema 错配与 provenance 错配单元测试。

本轮回放覆盖 4800 个 sampled window-head group。它是 fullres 网络产生的真实
post-G0 trace，但每个 block/sample 仅抽取 4 个 ordered group，不能冒充完整帧周期。

### 2.2 Local5 数据流候选

| 配置 | ready100 平均周期 | 相对强基线判断 |
|---|---:|---:|
| `4xW1+Stripe` | 2172.20 | 强基线 |
| `global QFSA+Stripe` | 2149.19 | 约 1.011x |
| `XBF-T8+Stripe` | 2245.67 | 负收益 |
| `4xW1+FCSR fifo16` | 2085.59 | 约 1.017x |
| `global QFSA+FCSR fifo16` | 2076.73 | 约 1.017x |
| `XBF-T8+FCSR fifo16` | 2135.05 | 未超过最佳 FCSR |

FCSR 的收益在 ready90/ready75 下仍约 1.017x，说明不是偶然的 ready100 特例；但
收益幅度不足以承担独立 DATE 架构贡献。XBF-T8 的跨 bank 交换成本超过其均衡收益，
应作为有价值的负结果保留，不继续 RTL 化。

### 2.3 Prosperity 官方模拟器结果

Local5 将每个 `gate x multiplicity` 按 destination/lane 精确重建，再把多位 gated
product 拆成 bit-plane，逐 plane 调用 Prosperity 官方 `Simulator.run_fc` CPU 路径。

| Stage | product-sparsity cycles | bit-sparsity cycles | bit/product |
|---|---:|---:|---:|
| S0 | 898940 | 1719390 | 1.913x |
| S1 | 790836 | 404482 | 0.511x |
| S2 | 10957852 | 18607040 | 1.698x |
| S3 | 11419520 | 22374526 | 1.959x |
| 总计 | 24067148 | 43105438 | 1.791x |

只有 S1 适合 bit-sparsity；S0/S2/S3 的 bit-plane 展开开销明显更大。上述 1.791x
尚未计 36876150 周期的跨 plane merge 下界，因此不能把 Prosperity bit-plane
作为 Local5 主执行方式。更合理的借鉴是保留 product-sparse 调度器，并仅把
per-stage 表示选择作为后续可选 DSE，而不是复制 Prosperity 整体数据流。

## 3. OpenROAD 环境与可复现约束

| 项目 | 固定值 |
|---|---|
| OpenROAD | commit `547465ccf8979379098216194f5837c413c7e2e9` |
| OpenROAD-flow-scripts | commit `3a0a1efd1d8d7891de1c4961487eaf6288adf7df` |
| Yosys | 0.33，revision `2584903a060` |
| 开放工艺 | Nangate45 |
| 时钟 | 5 ns，200 MHz |
| I/O delay | 输入/输出均 0.5 ns |
| 初始利用率 | 45% |
| 物理随机种子 | 42 |
| 参数 | TOKENS=6，OUT_TILE=32，96 个 product lane |

版本、SDC 和运行入口分别冻结在：

- `openroad_hifp/ORFS_VERSION.lock`；
- `openroad_hifp/constraint.sdc`；
- `openroad_hifp/run_openroad_hifp_blocks.sh`；
- `scripts/summarize_openroad_hifp_blocks.py`。

## 4. 为什么不直接用完整顶层面积

完整 `gatestack_dctf96_banklocal_projection_top` 首次试跑可以完成详细布线，最终
DRC 为 0，但该 RTL 边界把 SRAM-facing 宽向量全部当成芯片顶层引脚：

- 顶层 IO：11493；
- 含 filler 的 component：约 53.4 万；
- 线长：约 4.965 mm 的千倍量级，即 4964794 um；
- via：1388415；
- 寄生后仍有 99 个 max-cap 违例。

这证明流程和结构可展开，也暴露了宽接口、复位树和层次边界问题；但其 pad/IO
假设不符合真实芯片，不能放进论文 PPA 主表。当前采用“一个 96-lane datapath 加
三份 32-lane accumulator”的分块组合代理，后续目标工艺必须换成 SRAM macro
wrapper 和真实块间通道。

## 5. 四个分块候选的物理结果

| 块 | 标准单元面积 (um2) | 单元数 | 关键路径 (ns) | setup/hold | max-cap | DRC | 线长 (um) | via |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DCTF96 Scalar | 86622 | 48752 | 1.867 | 0/0 | 7 | 0 | 676889 | 308763 |
| DCTF96 PPDI | 110690 | 58095 | 2.497 | 0/0 | 11 | 0 | 865069 | 380406 |
| Acc RMW | 110523 | 54351 | 2.535 | 0/0 | 3 | 0 | 794553 | 357923 |
| Acc IBF | 109854 | 55337 | 2.728 | 0/0 | 1 | 0 | 886709 | 376701 |

33 个 datapath 未约束端点均已逐项核验：3 个 `acc_update_token_ids` 固定 LSB 和
30 个 `fabric_max_occupancy[31:2]` 常量高位。IBF 的一个未约束端点是固定常量
`final_token_ids[0]`。没有发现漏时钟寄存器路径；正式 DC 仍应通过 wrapper 或
case analysis 消除这些报告噪声。

### 5.1 PPDI 的真实物理代价

相对 scalar datapath，PPDI：

- 标准单元面积增加 27.79%；
- 单元数增加 19.16%；
- 关键路径增加 33.70%；
- 线长增加 27.80%；
- via 增加 23.20%。

这说明 PPDI 的“30.27% command-work 降低”不会免费转化为 PPA。它增加了 pairing、
双 destination 控制和分发网络；旧 Yosys 单元面积不足以反映线网代价。PPDI 仍有
正收益，但现在应被定位为**条件晋级机制**，而不是无条件主创新。

### 5.2 IBF 的真实物理代价

相对 RMW accumulator，IBF：

- 标准单元面积降低 0.61%；
- 单元数增加 1.81%；
- 关键路径增加 7.64%；
- 线长增加 11.60%；
- via 增加 5.25%。

IBF 把 7290 次 bias 请求降为 45 次，且组合面积没有增长。线长和关键路径上升来自
final drain 与输出选择网络，但在 5 ns 下余量仍充足，因此它比 PPDI 更接近可稳定
写入主文的硬件贡献。

## 6. 真实周期与物理面积合并判断

组合口径是一份 datapath 加三份 accumulator，不计块间通道和 SRAM 宏。

| 模式 | Motion 周期 | 周期加速 | 组合面积比 | 面积归一吞吐 | 当前判断 |
|---|---:|---:|---:|---:|---|
| Scalar+RMW | 53910 | 1.000x | 1.000x | 1.000x | 公平基线 |
| PPDI+RMW | 49350 | 1.092x | 1.058x | 1.033x | 弱正收益，待功耗验证 |
| Scalar+IBF | 50295 | 1.072x | 0.995x | 1.077x | 稳定晋级 |
| PPDI+IBF | 45735 | 1.179x | 1.053x | 1.120x | 当前最佳组合 |

关键解释：

1. 两项机制组合后周期减少 15.16%，不是减少 17.9%；1.179x 是加速比。
2. IBF 单项的面积效率贡献比 PPDI 更确定。
3. PPDI+IBF 的 1.120x 仍为开放库分块代理；没有 SRAM、功耗和块间线，不能写成
   目标 ASIC 的 FPS/mm2。
4. 四种组合均 detailed-route DRC=0、5 ns setup/hold=0，但都还有 max-cap，故
   `DRV clean=否`。

## 7. 对 Motion/Local5 双线的架构决策

### 7.1 Motion 为何继续做硬件主线

Motion 当前同时具备：

- `[prof]` K-zero 约 88.7%、pair-empty 约 74%、final-gate work 减少约 82.5%；
- `[rtl]` PPDI+IBF 四阶段真实回放 1.179x、233280 个 INT32 输出零失配；
- `[open-pnr][代理]` 组合面积归一吞吐仍有 1.120x 正收益；
- SCS-Shiftmax、gate-term 和 HIFP 从 attention 到 projection 的连续数据流叙事。

这条证据链已经从“数据稀疏”走到“RTL 正确”再走到“物理趋势为正”。

### 7.2 Local5 为什么不能删除但暂不做新后端

Local5 仍可能在最终 fullres AEE/AAE 或 attention 算法质量上胜出，而且 HIFP 的
term consumer 可以复用；但当前硬件证据显示：

- FCSR 对强基线只有约 1.017x；
- XBF-T8 为负结果；
- Prosperity bit-plane 整体比 product-sparse 慢 1.791x；
- 尚无与 Motion 同边界的 Local5 完整 RTL 周期和物理收益。

因此采用“双算法前端、单投影后端”策略：Local5 继续完成 RTL-exact 和 fullres
质量验证，优先复用 gate-term/HIFP 接口；只有它在质量上明显领先且真实 trace
显示新的瓶颈，才修改 producer 或 descriptor，不复制第二套 accelerator。

## 8. 当前完成度

| 层级 | 当前状态 | 是否可进论文主表 |
|---|---|---|
| Motion workload profile | profile100 与关键稀疏统计已完成 | 是，标 `[prof]` |
| Local5 workload profile | profile100、ordered replay、Prosperity 已完成 | 是，注明 sampled scope |
| 投影 RTL 正确性 | 四模式真实回放、整数逐元素等价 | 是，标 `[rtl]` |
| 开放库物理实现 | 四叶块 P&R/STA 完成 | 可作趋势/附表 |
| T450/fullres Motion RTL-exact | 等待软件侧完整 hardware-order 结果及 T450 回归 | 否 |
| Local5 RTL-exact | 等待软件侧 fullres exact 结果 | 否 |
| 多样本、多窗口 p95/p99 | 仍不足 | 否 |
| SRAM latency、宏面积与能量 | 未闭环 | 否 |
| 目标工艺 DC/STA | 未做 | 否 |
| 真实 SAIF/PTPX 功耗 | 未做 | 否 |
| full-encoder FPS/energy/frame | 仍是预算模型 | 否 |

按 DATE 预硅软硬件协同论文口径，核心投影子系统可认为约 85% 完成；整篇硬件证据
约 65%。OpenROAD 把“仅 Yosys 面积代理”推进到了“可路由的开放库物理趋势”，但
没有替代系统和功耗闭环。

## 9. 下一阶段固定顺序

### 9.1 当前机器继续完成

1. 用多个 sample/window 生成 Motion 与 Local5 真实 ordered trace，报告 cycles、
   stall、bank imbalance、PPDI pairing 的 mean/p95/p99；
2. 给 HIFP simulator 接入 1/2-cycle SRAM latency、随机 final backpressure 和连续窗口；
3. 生成四模式同一批真实 trace 的 VCD/SAIF，冻结信号映射和采样长度；
4. 建立 SRAM/DRAM 访问分账：weight、descriptor、Acc、bias、output 分开；
5. 等待并接入 Motion T450 与 Local5 fullres hardware-order exact 结果；
6. 不再横向发明 RTL 模块，除非上述压力测试证明当前机制收益消失。

### 9.2 搬到 DC/PTPX 服务器后

1. 使用同一 file list、参数、SDC 和 memory wrapper 综合四个公平候选；
2. DC 报告 hierarchy area、cell count、buffer/inverter、WNS/TNS、DRV 和
   unconstrained paths；
3. 用 Formality/LEC 证明 RTL 与网表等价；没有 LEC 时补 SDF 门级回放，但要明确
   它不能完全替代形式等价；
4. PrimeTime 做目标 PVT 的 setup/hold，至少补 slow/setup 与 fast/hold；
5. PrimeTime PX 使用同一批真实 SAIF，分 internal、switching、leakage；
6. SRAM 使用同工艺 memory compiler 宏；若暂时没有，则用 CACTI 并明确标 `[模型]`；
7. 汇总 logic+SRAM+DRAM 的 mJ/frame、FPS、EDP 和 FPS/mm2；
8. 若目标工艺可供 LEF/techfile，再做含宏 P&R、SPEF 回标和 post-route PTPX。

## 10. 最终论文口径

目前最可辩护的硬件主张不是“PPDI 一定大幅提升 PPA”，而是：

> all-binary attention 产生的低基数 gate-term 允许跨 attention/projection 保持精确
> 稀疏语义；HIFP 分层外提 term 内产品和 tile 内偏置不变量。IBF 提供稳定的周期与
> 面积效率收益，PPDI 在真实 Motion trace 上进一步提升吞吐，并在开放库分块 P&R
> 后保持正的面积归一收益。

目标工艺功耗出来前，SCS-Shiftmax、gate-term/HIFP 和 IBF 可作为核心贡献；PPDI
作为有条件的增强机制。Local5 是复用同一后端的竞争算法前端，而不是第二篇未经
验证的硬件故事。

## 11. 结果路径

- `results/openroad_hifp_blocks_20260802/report.md`；
- `results/openroad_hifp_blocks_20260802/report.json`；
- `results/local5_fullres_postg0_qfsa_replay_20260730/report.md`；
- `results/prosperity_local5_gated_bitplane_20260802/report.md`；
- `results/ppdi_ibf_real_trace_20260801/report.json`。
