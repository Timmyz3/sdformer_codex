普通RR下，模数四位置打包仍相对**当前RR＋静态证明三P10对照**减少组件周期：真实64块冷/无BP为 **814321→787603拍，省26718拍（3.2810%）**。176条命令及共享资源账目全部通过。**RR＋借用既有consumer64链的四P更新仍是未测强控制，本项不足以晋级标题或声称优于所有强控制。**

|实际范围|RR＋tripleP10 mode1|RR＋modular fourP8 mode2|净省拍|
|---|---:|---:|---:|
|159起3tile，冷/无BP|42284|41090|1194|
|19197起3tile，冷/无BP|32045|31884|161|
|128起64tile，冷/无BP|814321|787603|26718|
|128起64tile，暖/无BP|812473|785755|26718|
|128起64tile，冷/有BP|884397|858814|25583|
|128起64tile，暖/有BP|882629|856904|25725|

两模式都在两个source tile装入后同时启动，五类source/W/z/psum/ALU请求有交集时普通RR，互不冲突可同时推进。没有seq/阶段错位模式。三P10超静态范围时在同context精确回退dualP13；这是功能fallback，不另计候选。模式只改变Q1表示与由此产生的事务。

算术和权重真正共享：[interleave_stream.sv](interleave_stream.sv) 第81行是唯一Q1/Q2阵列，第132–172行在操作数选择后执行唯一8个signed19×13乘法和8条32bit carry链；[rr_context.sv](rr_context.sv) 只有operand/result总线，没有数据乘法或carry链。top同一套加法链也在864个已付冷Q1配置beat中累加唯一26B正负范围界，range_ok广播。proof加入shared_ALU_grants，不把两份host证明结果免费送进context。

每context同520B z、416bit向量口、完整source/psum/qblock/holding及8B correction状态，consumer仍是原独立完整FP32 identity→J20→I24后端。新两臂共担分段、修复、证明与选择硬件，预算见 [resource_contract.json](resource_contract.json)。这不是与旧208bit D3或单context模块的同面积比较。

ZADD、REPAIR_ADD、NORMALIZE_ADD和BASE_MAC都原子申请z＋ALU；REPAIR_READ/NORMALIZE_READ也要z grant。[rr_context.sv](rr_context.sv) 第67–94行先产生独立请求，再门控实际z读；第167–178行在grant前冻结执行，随后第240–257行才提交修复/规范化。因此高位费用实际进入RR时间线。

|64冷/无BP的实际内部量|三P10|模数四P8|
|---|---:|---:|
|Q1更新issue|76424|60113|
|Q2 MAC|198720|198720|
|高位修复issue|0|0|
|规范化issue|0|640|
|规范化的仲裁等待|0|693|
|shared ALU grant（含864 proof）|276008|260337|
|shared z grant|354768|323426|
|全部资源冲突/仲裁等待|180553|174163|
|两个context的活动周期之和|1319312|1265876|
|consumer join等待|497336|470618|

不能把单context的4.1929%乘到RR。这里本征服务义务省 `3×(76424−60113)−2×640=47653` 拍，仲裁与consumer等待改变实际重叠，最终总周期只省26718拍。两个context活动周期包含重叠和等待，也不能加到wrapper周期。实际闭合关系是 `shared_ALU=proof+Q1+Q2MAC+repair+normalize`、`shared_z=z向量读+标量读+写`、`conflict=core arbitration stalls`；完整窗口/加载/退休总账由 [verify.py](verify.py) 和C++ TB核对。

真实8块、3块、64块在本工作点没有触发high修复，不能据此说修复免费。额外双tile all-one输入在mode2实际执行300次修复并等待600个仲裁拍；20次规范化另外等38拍。正/负极值双tile各执行200次修复、等待400拍，三P10对这两种Q1均触发精确fallback。源是逐channel常值的真实输入数据，仍由RTL绝对source地址请求装载；不是gold供数。背压/暖重启也覆盖这些碰撞。

验证范围为：16fixture×两mode×有/无BP×冷/暖128命令；三种双tile压力24命令；159及19197起真实3tile16命令；128起真实64tile8命令。合计**176命令，raw p、实际J20、I24各核对2826240值，全通过**。测试检查输入请求保持、输出holding、tile/row/last、480beat退休及无reset重启。`verify.py`重新从原生source/Q1/Q2逐K重建centered高低位、修复义务、规范化和raw，并按context合计核所有执行/资源账目；没有用预测值驱动RTL。

现有consumer64链仍属于同一预算，已有fourP借链方法必须接入同样RR后再作判断。本次没有执行该控制，不能把“不借宽链”本身等同硬件优势。下一次比较需保留416bit z口/两个context/相同source与consumer权限，并真实计入borrow与消费者的冲突。

新颖性边界也未解决：已核对的 [ISCAS2025官方摘要](https://epapers2.org/iscas2025/ESR/paper_details.php?paper_id=1103) 已讨论窄本地psum在将要溢出时与output buffer中的部分和合并，以减少guard bits；本次只读摘要，不据此判断全文缺少哪些细节。通用overflow repair、子字打包或多context RR都不能单独充当本项标题贡献。

全部结果是Verilator组件周期。未跑19200块、EDA、物理SRAM映射、Fmax/能耗、整网FPS或新训练/质量评价。保留为组合兼容性与当前对照内的正结果。

复现：`python3.12 run.py`；`python3.12 run_stream.py`；`/opt/anaconda3/bin/python3.12 verify.py`；`python3.12 run_stream.py --stage 64`；再次运行verify。Verilator4.028使用`-Wall --cc --exe`。源码由 [build_from_sources.py](build_from_sources.py) 在本目录生成，旧树与已完成modular_packing只读。计划见 [PLAN.md](PLAN.md)，收据为 [results.json](results.json)、[results_small.json](results_small.json)、[results_64.json](results_64.json)、[checks.json](checks.json)。
