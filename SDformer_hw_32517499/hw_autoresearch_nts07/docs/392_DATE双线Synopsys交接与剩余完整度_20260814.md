# DATE双线Synopsys交接与剩余完整度（2026-08-14）

## 1. 本轮裁决

双线架构机制在本轮冻结，不继续为提高创新分扩展第四拓扑、第三条稀疏路径或新编码。当前工作转为论文完整度闭环：把冻结的Motion公平双槽切片和Local5生产切片变成可在Synopsys服务器上直接执行、可追溯、失败关闭的综合与功耗交付包。

该交付包的完成不等于已经获得ASIC PPA。本机没有`dc_shell`、`fm_shell`、`pt_shell`、目标`.db`、SRAM/RF宏和SPEF，因此当前证据仍是`[rtl]`、开放综合可读性检查和`[待DC/PTPX]`。

## 2. 冻结综合对象

| 顶层 | 固定参数 | 边界 |
|---|---|---|
| `h67_fixed2s_mssb5_dc_top` | MSSB5 score-front、Fixed2S、32-entry FIFO、flop K store | Motion T450 attention-row slice，不是encoder |
| `h67_rqtb2s_mssb5_dc_top` | 与Fixed2S完全相同，仅`QUOTIENT_ENABLE=1` | Motion T450 attention-row slice，不是encoder |
| `local5_unified_out2_dc_top` | Q-silent、ident-K、保序overlap、FCSR mode 0、TCFM5、`OUT_DIM=2` | score→Acc32 tile，不是encoder |

Motion的两个DC顶层使用相同端口、score-front、FIFO、K存储、SCS/Shiftmax和输出边界，后续面积、时序和功耗差异可用于隔离时间商逻辑。MSSB5在两边都打开，只是主数据流的score-front支撑，不单列贡献。

Local5新增了逐层显式参数`RELATION_SCHED_MODE`、`ACC_BACKEND_KIND`和`ACC_MEMORY_IMPL`，wrapper明确固定为`0/0/0`。这只消除编译宏污染和默认参数歧义，不改变生产执行语义。

## 3. 当前可复核结果

### 3.1 本机预检

- 三个冻结wrapper均通过当前源码Verilator lint。
- 三个冻结wrapper均通过Yosys elaboration、`proc/opt`和`check -assert`，均为`0 problems`。
- Yosys数字只用于证明可综合展开，不是门级面积、时序或ASIC PPA。
- handoff静态审计检查top、filelist、参数、SDC、脚本与SHA，当前为PASS。
- 本机运行DC、Formality、PrimeTime STA和PrimeTime PX均以“工具不存在”明确退出，不生成伪报告。
- 旧Formality脚本只生成验证报告、未将`verify`失败传递为shell失败；本轮已改为写`formality_status.txt`并在非PASS时返回非零。

### 3.2 Local5参数冻结回归

使用joint ep29真实100-group向量、当前源码、Verilator `--assert`重新编译并回放：

- 范围：score/Shiftmax5→relation/FCSR→term→TCFM5→Acc32；
- `OUT_DIM=2 tile`，不是encoder；
- `groups=100`；
- `total_cycles=155791`；
- Acc32、score/gate与协议断言全部通过。

第一次回放因使用不存在的默认向量目录而触发`$readmem`缺失和首行score mismatch，该运行已按失败处理，未作为证据。使用manifest对应的真实向量目录后才得到上述PASS。

### 3.3 主表纪律

本轮没有修改`docs/359_DATE终局冻结_20260813.md`，其SHA仍为：

```text
dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
```

双线主表报告器仍通过。Motion封存公平锚点和Local5十二窗`OUT_DIM=2 tile`封存列均未被本轮旁路结果覆盖。

## 4. Synopsys服务器执行顺序

### G0：身份与输入

1. 运行`audit_date_dual_handoff.py`并归档JSON及所有SHA。
2. 冻结标准单元`.db`、PVT、operating condition、时钟周期和约束SHA。
3. 冻结SRAM/RF宏的`.db/.lib`、端口、读写延迟、read-during-write语义和面积。
4. 冻结真实trace、checkpoint/config、向量manifest、仿真器版本和SAIF层次前缀。

### G1：DC与等价

1. 在完全相同库和SDC下分别综合三个顶层。
2. 检查`check_design`、`check_timing`、unconstrained path、WNS/TNS、max fanout/cap/transition、面积层次和memory inference。
3. 对每个顶层运行Formality，所有compare point必须通过。
4. Motion必须成对报告Fixed2S与RQTB2S；Local5只报告`OUT_DIM=2 score→Acc32 tile`，不能写成encoder PPA。

### G2：STA、布局布线与功耗

1. 先用PrimeTime做DC网表pre-layout STA。
2. 用OpenROAD或商业P&R获得布局布线网表和SPEF后，按相同PVT做post-route STA。
3. 用真实trace产生SAIF；PTPX必须报告未注释节点，不能以默认翻转率替代。
4. 报告dynamic、leakage、memory和控制层次功耗；将功耗乘以同一工作负载周期，换算`energy/head-row`或`energy/window`。
5. OpenROAD代理、DC pre-layout和PTPX post-route三类结果必须分列，禁止混称ASIC签核。

## 5. 不只是DC：仍缺的论文完整度

### Motion

1. 当前公平锚点仍需扩展为多样本真实Fixed2S/RQTB2S RTL，而不是只依赖RTL校准模型。
2. sample0/window0的138行已用checkpoint INT8权重闭合两个输出通道、1104个Acc32标量零失配；仍缺在同一公平包中覆盖全部输出通道，以及bias/BN/requant/residual后的block输出。
3. 需要给出score、slot/FIFO、SCS/Shiftmax、K read、projection和stall的周期/能量分解。
4. 需要12-block/full-encoder operator share和Amdahl结果，避免把局部`1.1865x`写成整网加速。

### Local5

1. 当前100-group生产闭环是`OUT_DIM=2 tile`；已有四个stage定向高负载组的`OUT_DIM=32`、57600个Acc32值零失配，但这是功能压力而非无偏群体性能，仍需更高OUT_DIM的周期/带宽/容量分布。
2. 需要跨window/跨sample的真实trace周期分布和最差组说明。
3. 需要12-block时间复用、权重切换、buffer容量、读回和连续窗口调度的系统壳或校准模型。
4. FCSR relation ring、weight row和五个Acc bank必须接真实SRAM/RF宏；当前inferred array可能被DC映射为flop/mux，不能作为最终面积能量。

### 双线共同项

1. DC/STA/Formality/SAIF/PTPX与macro-aware PPA。
2. post-route DRC/DRV、max-cap/max-transition和关键路径归因。
3. SRAM/DRAM访问量、带宽峰值、吞吐/面积、能量/flow和尾延迟。
4. 与强基线在相同位宽、端口、存储容量、时钟和工艺下比较。
5. 从算法checkpoint到profile、RTL向量、SAIF和报告的完整身份链。

因此当前不是“只差DC”。RTL组件与验证工程已较完整，但论文实验完整度仍取决于多样本、真实存储、功耗活动和full-encoder组合证据。

### 现有系统工件的复用边界

- `results/hit_flow_full_encoder_budget_ordered.json`仍以PCCC开关和旧H67 attention cycle代理建模，且文件自身说明空间负载是活动率加权MAC代理。它不能直接替换为当前Motion/Local5周期后作为论文端到端结果。
- `results/qfit_local5_encoder_job_scheduler_20260809/report.json`可复用为Local5系统控制壳：`[rtl]`闭合1320个window group、6720个output-tile request和54000个input-head/replay request；但它明确不含T450 token数据路、真实SRAM延迟、weight reload、跨头归约和最终输出。
- 下一版full-encoder模型必须把冻结切片接入该作业壳，并从同一真实trace获得每个job的服务周期与存储事务；在此之前不生成新的encoder speedup。

## 6. 本轮DATE复评

两位独立评审对创新分存在分歧：严格评审约`3.1/5`，另一评审对Local5给到`3.7/5`。共同意见一致：

- 新机制继续堆叠的边际收益已经低于证据风险；
- 生成式退休合同可作为Local5统一数据流的规格来源，不是第四贡献；
- 当前最低项仍是实验/系统完整度，而不是RTL代码量；
- 真正能提高接收概率的是强基线、公平宏模型、多样本和DC/PTPX闭环，而不是重新命名已有机制。

本轮之后按该共同裁决冻结新机制，Motion和Local5继续双线补证据，不放弃任一条线。

## 7. 主要工件

- `dc_handoff/rtl/date_motion_dc_tops.sv`
- `dc_handoff/rtl/date_local5_dc_top.sv`
- `dc_handoff/filelists/date_motion_2s.f`
- `dc_handoff/filelists/date_local5_out2.f`
- `dc_handoff/constraints/date_dual_core.sdc`
- `dc_handoff/run_dc.sh`
- `dc_handoff/run_formality.sh`
- `dc_handoff/run_ptsta.sh`
- `dc_handoff/run_ptpx.sh`
- `dc_handoff/scripts/audit_date_dual_handoff.py`
- `dc_handoff/scripts/audit_synopsys_postrun.py`
- `dc_handoff/runs/date_dual_handoff_audit_20260814.json`
- `dc_handoff/runs/local5_prod_paramfreeze_g100.log`
