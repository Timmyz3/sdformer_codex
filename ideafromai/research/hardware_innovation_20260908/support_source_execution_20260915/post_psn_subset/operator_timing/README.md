# 单判决 lane 的两点逻辑映射

**本次固定 3ns 的两个 DC 点都满足 setup；规范化式在此 slice 中 cell area 少 5.2678%，最坏数据到达时间少 0.17175ns。** 这支持把阈值移位移出 prefix 串联支路的局部可行性，不证明整 H8 核达到 3ns，也不证明整链加速比。

| 同约束映射结果 | 原式 NORMALIZED=0 | 规范化 NORMALIZED=1 |
|---|---:|---:|
| Total cell area，库面积单位 | 5293.259944 | 5014.421946 |
| Combinational area | 2773.889963 | 2491.271965 |
| Sequential area | 2519.369981 | 2523.149981 |
| Sequential cells | 1333 | 1335 |
| 全部 cells | 5207 | 5076 |
| 最坏数据到达时间，含 clk-to-Q，ns | 2.66545 | 2.49370 |
| 该路径 required time，ns | 2.77861 | 2.77937 |
| 最坏 setup slack，ns | +0.11316 | +0.28567 |
| 关键 startpoint | hi_r_reg[2] | lo_r_reg[3] |
| 关键 endpoint | gate_out_reg | gate_out_reg |
| DC 逻辑层数 | 70 | 67 |
| macro / black box | 0 | 0 |

面积是标准单元逻辑映射的 cell area，未包含布线面积。到达时间少 6.4436% 是这个固定约束、两次独立映射的结果，不能取其倒数宣传 Fmax 提升。两臂的最坏路径都从实际 LUT 地址寄存器出发，经异步 LUT mux、prefix 更新和判界到 gate 寄存器；不是给 LUT 结果免费后只量一个比较器。

真实 cut 与数值接口

[operator_slice.sv](operator_slice.sv) 有两份 `32×16` runtime LUT，共 128B；同步 cfg 写、异步地址 mux 读，没有 initial ROM。注册 prefix、两半 code、m、tau、P/N 和 gain/constant 上下文；GROUP 实际生成 tail 或 q，PLANE 实际取两半表、执行两级 prefix 加法并注册 gate/lock。两臂输入寄存器、输出四个判决寄存器及其负载相同。原式更新真实 tail 递推，规范化式保留真实 signed49 q 寄存器，没有 q 输入端口。

这里从已注册 prefix/code/P/N 开始，未包含 Y 产生、Y→plane 转置、符号 header 初始化或 A→P/N 构造。P/N 输入代表十项合法系数和；功能测试由实际配置的系数独立生成它们。这是用户指定的算术 slice 边界，不是免费完成全 PSN 前端的接口。

原式使用 `L=nv*2^m+N*(2^m−1)`、`H=nv*2^m+P*(2^m−1)`；只接受各实际中间量都不溢出 signed48 的域。规范化式用 `delta=positive?1:0`，GROUP 注册 `qN=tau+N−delta`、`qP=tau+P−delta`，signed49；PLANE 比较 `nv+N > qN>>>m` 和 `nv+P <= qP>>>m`。负数算术右移实现 floor，m=0 和正负 gain 的等号方向均保留。

**与同时完成的全链版本的差别必须保留：** [normalized_bound.sv](../normalized_certificate/normalized_bound.sv) 已将 P/N 状态改存 P−1/N−1，并用两输入加 carry-in 共用 GROUP/PLANE。本 probe 在综合前冻结为原 P/N 状态，GROUP 显式减 delta；两者 PLANE 数学与移位支路匹配，但 GROUP 算术、reset、共享结构不同。因此本表不是该全链源码逐行综合的结果。

功能验证

[tb.cpp](tb.cpp) 用独立 signed `__int128` 原 L/H 公式，而不是复述规范化谓词，先核原式中间域再对两臂逐判决比较。三组运行时系数表含混合符号、±16bit 边界和全零；遍历 m=0..23、两种 gain、constant、L/H 相邻阈值、signed48 tau 两端，以及多 plane 递推。每个表项经真实 cfg 接口写入。原 signed48 不合法输入不拿截断凑通过。

[FUNCTIONAL.json](FUNCTIONAL.json) / [functional.log](functional.log)：60,888 组上下文，每臂 82,968 次判决，9,360 次 L 或 H 恰等 tau，9,616 个 tau 端点上下文，全部通过。四个输出中的 lower/upper 用于观察即使已锁定后的递推，因此该 probe 不把早停周期当作测量对象。功能验证是 RTL 与独立参考对比，没有做映射后等价验证。

映射约束与可查收据

[map.tcl](map.tcl) 对两个静态参数分别且仅一次 `compile_ultra`：指定 TSMC28 `tcbn28hpcplusbwp35p140ssg0p9v125c.db`，slow 0.9V/125°C，3ns，setup uncertainty 0.2ns，hold 0.05ns，I/O delay 0.25ns，input transition 0.1ns，output load 0.01，max fanout32，ZeroWireload。与已读 m2248 约束一致，无 clock sweep。所有产物位于本目录。

[run_dc.py](run_dc.py) 实际持有 `/tmp/date_dual_synopsys_same_uid_eda_queue.lock`，用 `27030@ic.ismd-nemo` 顺序完成两点后释放锁；[DC_RUN.json](DC_RUN.json) 两臂 returncode 均为 0。runner 的 launch 收据阻止无意重复综合。本次没有 PT/FM/CTS、布局布线、PPA_ADMISSION 或生产 runner。

原式：[area](dc_0/reports/area.rpt)、[全局 timing](dc_0/reports/timing_all.rpt)、[判决 timing](dc_0/reports/timing_decision.rpt)。规范化式：[area](dc_1/reports/area.rpt)、[全局 timing](dc_1/reports/timing_all.rpt)、[判决 timing](dc_1/reports/timing_decision.rpt)。[SUMMARY.json](SUMMARY.json) 保留逐路径数据；`/opt/anaconda3/bin/python3.12 summarize.py` 只解析现有报告，不重跑工具。

报告边界与实际警告

- 全局 top20 与四个判决端点报告有效。按源寄存器分类的补充 `-from` 查询误用了 cell collection，被此 DC 拒绝；这些 `timing_*_to_decision.rpt` 无效，不能引用为单独 q/tail 支路延迟。本次保留错误原件，没有为补图追加综合点。
- 全局报告也覆盖非判决 endpoint：原 tail 寄存器路径为 2.37724ns；规范化 q 寄存器路径为 2.33289ns。后者起点仍是 lo_code：未作 GROUP/PLANE case analysis，静态图保留共享 mux 和寄存器使能相关路径；不能把它描述为实际 GROUP 必须读取 LUT。均未超过相应判决路径。
- `check_design` 唯一连接警告是两臂 `prefix_in[47]` 被移位逻辑消去；合法域由功能参考检查。两臂无 setup、max-transition 或 max-capacitance 违例。`constraints.rpt` 仍列默认零 `max_leakage_power` 目标未满足；不声称“所有约束通过”或能耗收益。高扇出网使用工具 fanout1000 估计，未做物理布线。
- 本 lane 私有 LUT、P/N 和 tail，而真实 H8 核跨 lane 共享相关状态；父核还有 FC1、唯一 Y、参数读取、输出 pack 和归约控制。**不能把面积乘 80，不能从此表推出整核 Fmax/等面积 PPA，更不能替代真实工作负载 RTL 周期表。** 真实 normalized 全链周期与原 joined 相同；局部映射结果只补一条物理可行性证据。
