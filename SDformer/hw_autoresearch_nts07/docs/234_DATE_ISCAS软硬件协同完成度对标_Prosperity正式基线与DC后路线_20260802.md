# DATE/ISCAS 软硬件协同完成度对标、Prosperity 正式基线与 DC 后路线

> 日期：2026-08-02  
> 范围：Motion/Local5 双线，attention-to-projection 子系统与 DATE 投稿证据  
> 当前机器：无 DC、PrimeTime、OpenSTA、OpenROAD、Vivado；有 Yosys、Icarus、
> Verilator 和 Nangate45 Liberty。  
> 约束：本文不把开放库逻辑映射写成 ASIC PPA，不把 CICC 流片实测要求误作
> DATE 的最低门槛。

> **2026-08-02 状态更新：** 本机随后已安装并锁定 OpenROAD/ORFS，完成四个 HIFP
> 分块候选的同约束 P&R/STA；Local5 profile100 与 Prosperity 官方评估也已完成。
> 最新结果和对本文过时状态的修正统一见 `docs/235_Local5_Prosperity与HIFP_OpenROAD物理实现收口_20260802.md`。

## 1. 直接结论

1. **当前不是只差 DC。** 核心投影子系统 RTL 已较完整，但论文还缺多样本外推、
   memory-aware 周期与能量、T450/fullres 回放、Local5 正式结果和 full-encoder
   边界预算。
2. **没有 DC 不妨碍当前推进。** 先完成真实 trace、强基线、SRAM/DRAM 访问分账、
   随机反压和多窗口连续执行；这些工作不会因以后换服务器做 DC 而作废。
3. **DC 不是终点。** DC 后至少还要做时序、门级等价/仿真、真实 trace 活动率
   功耗、SRAM 宏面积与能量、端到端 FPS/energy/frame，以及同约束基线表。
4. **暂时冻结新增 RTL 创新。** 当前主张先固定为 SCS-Shiftmax、gate-term 流重塑
   和 HIFP。Local5 profile 完成前继续发明模块，容易对单样本过拟合，反而削弱
   论文主线。
5. **Prosperity 已升级为真实官方代码基线。** Motion gated-K 已按精确 bit-plane
   分解并逐 plane 调用官方 `Simulator.run_fc`；Local5 在 post-G0 profile 和逐元素
   导出完成后复用同一流程。

## 2. 同类论文通常做到哪一步

### 2.1 DATE 完整应用型软硬件协同

DATE 2024 的 NVCA 是最接近本项目的参照之一。它同时给出：

- 新网络和稀疏算法，并报告视频压缩任务质量；
- 完整硬件架构、片上 buffer 和异构 layer-chaining dataflow；
- cycle-accurate simulator，并用 RTL 校准；
- Synopsys DC + TSMC 28nm 的逻辑面积/能量；
- 频率、gate count、片上存储、功耗、GOPS、GOPS/W；
- 片外流量和算法/架构消融。

来源：[NVCA, DATE 2024](https://past.date-conference.com/proceedings-archive/2024/DATA/258_pdf_upload.pdf)。

这类论文**没有要求必须流片**，但要求算法、架构、周期模型、RTL 校准和 PPA 形成
一条闭环。

### 2.2 DATE 架构模拟器主导型

DATE 2024 Best Paper Candidate FusionArch 的完成形态是：

- 处理近乎完整的点云网络流水，而非一个孤立算子；
- 5 个网络、4 个应用、多种点规模；
- cycle-accurate simulator；
- DRAMSim3 建模片外存储，CACTI 建模片上 SRAM；
- 固定 28nm、频率、MAC 数、SRAM 和 DRAM bandwidth；
- server/edge 两个规模，并与 CPU、GPU、已有加速器比较；
- 同时报告精度、速度、能量和系统资源。

来源：[FusionArch, DATE 2024](https://past.date-conference.com/proceedings-archive/2024/DATA/196_pdf_upload.pdf)。

它说明 DATE 可以接受“系统级 simulator + 组件物理模型”为主的架构论文，但前提是
工作负载覆盖、存储模型和公平基线足够强。只用单窗口 RTL 周期不够。

### 2.3 DATE/ISCAS FPGA 部署型

DATE 2022 的 SNN accelerator 直接用 Verilog，经 Vivado synthesis/implementation
后部署到物理 FPGA，报告：

- 多个网络与数据集的准确率；
- LUT、FF、BRAM；
- 实现后频率；
- latency、FPS、板级功耗；
- 大模型 VGG-11 的外部 DRAM 和片上 BRAM 需求。

来源：[Resource-efficient SNN Accelerator, DATE 2022](https://past.date-conference.com/proceedings-archive/2022/pdf/0881.pdf)。

ISCAS 2024 的 PEFSL 也采用端到端 FPGA SoC 展示，报告 MiniImageNet 上任务结果、
30ms latency 和 PYNQ-Z1 的 6.2W 功耗。来源：
[PEFSL, ISCAS 2024](https://arxiv.org/abs/2404.19354)。

ISCAS 2025 的 DVS/SNN ADAS 工作在真实 FPGA 上达到 92.08 FPS，并报告
DVS-Gesture 94.3% 准确率。来源：
[ISCAS 2025 官方论文页](https://epapers2.org/iscas2025/ESR/paper_details.php?paper_id=1394)。

因此，若以后没有商业 ASIC 流，也可以走 FPGA 完整部署路线；但当前机器没有
Vivado，且本项目目标是 DATE 预硅 ASIC 架构，优先级仍是 DC/STA/SAIF。

### 2.4 高完成度系统与电路论文

DATE 2019 的 VIO 工作同时给出 28nm、600MHz、1.3mm2、560KB SRAM、2.2mW，
并在 FPGA 上用 EuRoC 数据集验证完整 VIO 任务。来源：
[VIO Hardware-Software Co-design, DATE 2019](https://past.date-conference.com/proceedings-archive/2019/html/0445.html)。

本地 CICC 2026 光流芯片则属于更高的流片实测档位：

- 28nm test chip 和 die photo；
- 0.625-1.2V、12-744MHz 实测；
- 0.21-75.48mW 实测；
- MVSEC 四个子数据集；
- AEE、mJ/inference、ms/inference；
- on-chip 与含 LPDDR3 外存能效分开报告。

本地文件：`docs/Zhang 等 - 2026 - A 28-nm Optical Flow Estimation Accelerator
with Redundancy Speculation, Bit-Width-Aware Compression.pdf`。

这类 CICC/ISSCC 论文通常需要流片和测量。我们不流片，就不应把其完成度设成
最低要求；但应学习其**系统能量必须含外存、任务质量必须与硬件优化联动**的口径。

## 3. 本项目应该用什么指标评价硬件

### 3.1 一级指标：决定论文是否成立

| 类别 | 主指标 | 原因 |
|---|---|---|
| 算法质量 | valid825 AEE、AAE，hardware-order 与 float 差值 | 先证明加速的是可用网络 |
| 正确性 | RTL/整数金参考逐元素 mismatch=0 | exact 架构的底线 |
| 性能 | fullres latency/frame、FPS、mean/p95/p99 cycles | 光流是实时任务，不能只报平均算子周期 |
| 能量 | logic+SRAM+DRAM energy/frame，EDP | 稀疏执行最终要回答是否真省电 |
| 面积 | logic、SRAM、总面积，FPS/mm2 | 防止靠堆 lane 换速度 |
| 存储 | SRAM 容量、读写次数、片外 bit/frame、峰值带宽 | 本网络与门码流的主要系统约束 |

不建议把 TOPS/W 作为唯一主指标。二值事件、跳零和 term 复用会改变“operation”的
定义，容易把数字做大却不能说明一帧光流真正消耗多少能量。

### 3.2 二级指标：解释为什么好

| 数据流位置 | 应报告指标 |
|---|---|
| attention | K-zero、pair-empty、score class occupancy、SCS 类扫描周期 |
| gate/term | gate cardinality、term 数、fanout、metadata bits、continuation/escape |
| PPDI | scalar delivery、paired commands、pairing ratio、偶奇失衡 p95/p99 |
| DCTF | producer/consumer stall、fabric occupancy、bank imbalance、retire stall |
| IBF | bias read/write bit、finalizer stall、Acc RMW 降低 |
| memory | weight/activation/Acc/bias SRAM 分项访问，DRAM traffic |
| 系统 | attention/projection/ATLIF/skip/IO 各阶段周期和 Amdahl 占比 |

### 3.3 当前已有与缺失

| 证据 | 当前状态 |
|---|---|
| Motion workload 动机 | profile100 已有 K-zero、pair-empty、term 等统计 |
| 投影 RTL 正确性 | 四种模式共 233280 个 INT32 输出零失配 |
| Motion 单窗口完整投影周期 | PPDI+IBF 为 45735，标量RMW为53910，1.179x |
| 偏置访问 | 7290 降至45，降低99.383% |
| PPDI command-work | sample0/window0 降低30.270% |
| 开放逻辑面积代理 | 组合结构面积倍率1.014，面积归一吞吐代理1.163x |
| Prosperity 官方强基线 | Motion gated-K bit-plane 已完成，见第4节 |
| Local5 post-G0 profile100 | 运行中，本文写作时48/100 |
| 多样本 HIFP mean/p95/p99 | 缺失 |
| T450/fullres RTL replay | 缺失 |
| SRAM latency/反压/连续窗口 | 缺失 |
| DC/STA/SAIF/SRAM macro | 缺失 |
| full encoder RTL 与端到端 FPS | 仅预算模型，未完整闭环 |

## 4. Prosperity 到底做了什么

### 4.1 已完成的三层证据

1. **官方工具链探针：** 已真实调用官方 `Simulator.run_fc` CPU 路径验证环境；
2. **Motion K support 探针：** 已把真实二值 K 输入官方 simulator，但该口径没有
   表达 Q1.7 gate，只能作为机制探针；
3. **Motion gated-K 正式强基线：** 本轮将真实 `gate_code × K` 精确分解为二值
   bit-plane，每个非零 plane 调用官方 CPU 路径。

官方仓库：[Prosperity](https://github.com/dubcyfor3/Prosperity)。

### 4.2 精确 gated-K 结果

有利于 Prosperity 的设置：

- 跳过全零 plane；
- 按 bit-plane 密度从高到低执行；
- 首个 plane 后权重保持驻留；
- 不把 plane 间移位累加成本加入官方周期；
- S1 全零 gated-K 按零 FC 周期处理，不计偏置和最终输出。

| Stage | active plane | Prosperity product | Prosperity bit-sparse | product/bit内部收益 |
|---|---:|---:|---:|---:|
| S0 | 6 | 900 | 728 | 0.809x |
| S1 | 0 | 0 | 0 | N/A |
| S2 | 6 | 9646 | 4362 | 0.452x |
| S3 | 6 | 51250 | 61606 | 1.202x |
| 总计 | - | 61796 | 66696 | 1.079x |

未计的最小 bit-plane merge 周期下界为8100。结果说明 Prosperity 的 product-sparsity
预处理只在 S3 获得净收益，S0/S2 反而受预处理开销拖累。

HIFP 同一 Motion sample0/window0 四阶段为45735周期，数值上低于 Prosperity 的
61796周期；但当前不能写成正式1.351x加速，因为：

- Prosperity 是128输出 lane，本设计是96个产品 lane；
- Prosperity 未计 merge、bias 和 final output；
- 两者存储端口、SRAM 容量与频率尚未物理对齐；
- 都只有一个 sample/window。

正式稿需要报告“原始周期趋势”和“同面积/同带宽归一结果”两列，不能只挑有利数字。

产物：

- `results/prosperity_motion_gated_bitplane_20260802/report.md`；
- `scripts/run_prosperity_motion_gated_bitplane.py`；
- `tests/test_run_prosperity_motion_gated_bitplane.py`。

### 4.3 PHI 与 Prosperity 不能混称

- Prosperity 有官方开源 simulator，本项目已实际调用；
- PHI 未找到官方公开 simulator；现有 PHI-like 结果是机制复刻与命中率敏感性
  扫描，不是官方复现；
- 正式论文中只能写“PHI-like analytical baseline”，不能写“PHI simulator result”。

### 4.4 Local5 自动接续

已新增 `scripts/run_prosperity_local5_gated_bitplane.py`。它从 post-G0 ordered
trace 的 `gate/lane/multiplicity/destination` 精确重建 Local5 多重集投影激活，按
`stage/block/head` 保持权重片边界，再进行 exact bit-plane 分解和官方
`run_fc` 评估。

当前 watcher PID 为`1849043`。它只轮询
`ordered_term_manifest.json`和`ordered_term_items.npz`，两者完整落盘后自动运行；
等待阶段不占 GPU。正式结果将写入：

- `results/prosperity_local5_gated_bitplane_20260802/report.md`；
- `results/prosperity_local5_gated_bitplane_20260802/report.json`；
- `results/prosperity_local5_gated_bitplane_20260802/watcher.log`。

该结果仍是每个block/sample抽4个ordered group的 sampled scope，不冒充full-frame。

## 5. DC 到底做到哪里才算完成

### 5.1 DC 前必须冻结

1. 同一投影边界的四个候选：scalar+RMW、PPDI+RMW、scalar+IBF、PPDI+IBF；
2. 同一参数、同一 reset/clock、同一 I/O delay、同一 SRAM macro 规则；
3. 完整 top file list、时钟/复位/false path/multicycle path 合同；
4. memory wrapper，禁止一版把 SRAM 展平为 FF、另一版使用 macro；
5. Motion/Local5 的真实 SAIF/VCD trace 集合；
6. 无 latch、无组合环、无多驱动、无 unconstrained endpoint。

### 5.2 DC 本身要交付

| 产物 | 必须内容 |
|---|---|
| synthesis netlist | 每个公平基线独立可复现 |
| area report | combinational、sequential、buffer/inverter、hierarchy breakdown |
| timing report | WNS/TNS、关键路径、目标频率、违例端点 |
| QoR report | 高扇出、max transition/cap/fanout、未映射单元、black box |
| constraints report | 时钟、IO、path group、unconstrained path=0 |

只拿到“compile成功”和总 cell area 仍不够。

### 5.3 DC 后的 DATE 最小闭环

1. **逻辑等价：** RTL 与门级网表做 LEC；无 LEC 时至少做关键真实 trace 的
   SDF/门级仿真，但后者不能完全替代 LEC；
2. **STA：** 至少目标 corner 的 setup/hold；有条件补 PVT corners；
3. **动态功耗：** 用多样本真实 VCD/SAIF 驱动 PrimeTime PX 或等价工具，报告
   internal/switching/leakage；
4. **SRAM：** 使用 memory compiler 或同工艺 macro 数据；没有 compiler 时用
   CACTI 作为明确标注的模型，面积与能量不能漏掉；
5. **片外存储：** 按每帧真实 bit traffic 和目标 DRAM 能耗/带宽计算，不把其混入
   on-chip TOPS/W；
6. **周期模型：** SRAM latency、DMA burst、最终反压、多窗口连续执行；
7. **系统表：** MHz、mm2、KB、mW、ms/frame、FPS、mJ/frame、EDP、AEE/AAE；
8. **公平消融：** 四种 HIFP 配置和 Prosperity/Direct 强基线使用相同 workload。

### 5.4 是否必须 P&R

- 对 DATE 架构论文，DC+STA+SAIF+SRAM/DRAM 模型和 RTL 校准的周期模拟器可以构成
  可投稿最低包；
- 若关键贡献是“去宽总线、bank-local 布线、时钟/互连节能”，只做 DC 很容易被
  质疑，最好补 P&R 后 wire delay、拥塞、post-route area/power；
- HIFP 的一部分主张涉及三 bank 本地执行和宽总线消除，因此 **P&R 是强烈建议，
  不是当前可跳过的长期目标**；
- CICC/ISSCC 若以芯片论文投稿，则需要流片实测。本项目当前不走这条线。

## 6. 现在是否继续迭代 RTL 创新

当前结论是：**不继续横向增加新模块，等待 Local5 profile；继续纵向补证据。**

现有投稿级贡献候选收敛为三条：

1. **SCS-Shiftmax：** 在离散 score class 和 zero-K 精确语义下的单次归一化后端；
2. **Gate-term 流重塑：** 不物化 token-major gated-K，以低基数 gate equivalence
   形成 destination term，连接 attention 与 projection；
3. **HIFP：** 对 term 内产品不变量和 tile 内偏置不变量分层外提，PPDI 映射偶奇
   Acc 端口，IBF 将 bias RMW 推迟到 final retirement。

新增 RTL 的触发条件只保留以下四类：

- Local5 profile 显示现有 term schema 明显不适配，metadata 或 escape 超标；
- 多样本 p95/p99 暴露 bank imbalance、fabric head-of-line blocking；
- 1/2-cycle SRAM 和随机反压使当前收益消失；
- 以后 DC/STA 暴露真实关键路径，必须改 pipeline 或存储端口。

除此之外，新的 FIFO、context、bank、cache 或调度器只会增加论文复杂度，不自动
增加新颖性。

## 7. Local5 profile 完成后的固定动作

1. 核验 profile provenance、100/100 完整性和 post-G0/hardware-order 合同；
2. 输出 gate cardinality、source/destination term、fanout、PPDI 配对率、delta escape、
   metadata 占比的 mean/p95/p99；
3. 将 Local5 gated projection 逐元素导出为精确 bit-plane，调用同一 Prosperity 官方
   runner；
4. Motion 与 Local5 使用同一指标表比较，不把 T162 外推冒充 T450 实测；
5. 选择主线：优先考虑 fullres AEE/AAE、HIFP收益稳定性、Prosperity差分和存储成本，
   不是只看单个 sparsity 数字；
6. 再决定是否生成新的 RTL。默认决策仍是复用 HIFP 后端，只替换 attention producer。

## 8. 推荐论文目标线

本项目最现实的目标不是“达到 CICC 流片论文”，而是：

> **DATE 预硅软硬件协同架构论文：真实 fullres 网络质量 + 多样本 workload +
> bit-exact RTL 关键子系统 + cycle-accurate memory-aware simulator +
> DC/STA/SAIF/SRAM/DRAM 全成本 + 强基线。**

达到该目标后，full encoder 不一定每个 ATLIF 都实例化成完整 RTL，但必须给出可审计
执行图、存储容量、调度周期和系统 FPS/energy/frame。若只停在当前单窗口投影 RTL，
完成度仍是高质量原型，不是完整 DATE 论文。

## 9. 当前阶段判定

| 维度 | 完成度判断 |
|---|---|
| 核心投影架构与 RTL | 约85% |
| 单样本精确回放 | 约85% |
| 多样本 workload 与尾延迟 | 约45% |
| 强基线 | 约65%，Prosperity Motion 已正式接入，Local5待补 |
| memory-aware 系统模型 | 约40% |
| ASIC PPA | 约15%，仅开放逻辑代理 |
| full encoder 系统闭环 | 约35% |
| DATE 硬件论文总证据 | 约60%，当前仍是 Borderline Reject |

下一阶段的收益主要来自**证据完整度和系统口径**，不是继续增加 RTL 数量。
