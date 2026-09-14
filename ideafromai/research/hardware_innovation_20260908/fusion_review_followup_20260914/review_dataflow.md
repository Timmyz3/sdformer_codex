独立复审结论：**支持旧 RTL 的功能与组件周期结果；D3 必须保留普通 RR 为强控制；发现一处 D2 资源表错误和一处配置措辞错误，已在新目录给出更正。没有发现第二套 producer 算术偷算或 gold 驱动调度。物理 SRAM 端口、面积、Fmax 和能耗仍待证。**

范围为旧 `dataflow/{d1_forward,d2_halo,d3_interleave}`、`phase_borrow`、`joint_selected`。旧树、生产代码和论文均未改动，也未重跑 19200 tile。以下旧文件链接相对于本报告，行号列明；所有新构建、测试和更正位于 [audit_dataflow](audit_dataflow)。

|项目|核对旧收据的周期|裁决|
|---|---:|---|
|D1，64 tile 冷/无 BP|STORE+forward 968747；无 psum 写 968747|支持“省事务、无周期收益”；D1 没有 full 收据|
|D2，19200 tile|full reload 324217349；halo 309563909|支持省 14653440 个源装入拍；普通 halo，不能作标题|
|D3，19200 tile|seq 291702149；阶段错位 260697004；RR 233289744|支持 RR 最强；阶段错位相对 RR 退化 27407260 拍|
|phase，19200 tile|dualP 324025349；四P借链 310814282|支持同 416bit 接口内省 13211067 拍|
|joint，19200 tile|共同 halo/direct + dualP 261166493；四P借链 247955426|支持同一 RTL 的组合结果；仍是单 context|

这些 full 值来自既有实际结果记录，本次独立重算了结果账目，并由新代表 RTL 验证源码延续性；没有将本次检查描述成新的全帧仿真。

**需修正，P2：D2 资源表误套 D1 模式。** [D2 resource_contract.json](../fusion_ten_trials_20260914/dataflow/d2_halo/resource_contract.json) 第 7 行写 mode1 W480/R0、mode2 W0/R0，实际 D2 只比较 full reload/halo，二者每 tile 都写、读 480 个 psum 向量；[halo_core.sv](../fusion_ten_trials_20260914/dataflow/d2_halo/halo_core.sv) 第 174–183 行没有对应转发条件。两条 full 收据均为 9216000 次读和写，D2 REPORT 的表正确。更正不改变周期结论。

**需修正，P3：静态 mask 不是不存在。** D1/D2 `resource_contract.json` 第 10 行的 “no expanded/mask inputs” 应改为“没有展开权重或动态 source/latent oracle mask”。实际 param6 是外部输入的静态 `k_live`，每冷命令付 864 beat；它精确等于 Q1 每列是否非零，已独立核对五组 real_0 参数。两处准确替换文本见 [contract_corrections.json](audit_dataflow/contract_corrections.json)，未写旧树。

**支持，算术共享与公平控制。**

- D3 的 Q1/Q2 唯一数组在 [interleave_stream.sv](../fusion_ten_trials_20260914/dataflow/d3_interleave/interleave_stream.sv) 第 78 行；第 125–153 行仅一套 8 个 signed19×13 乘法表达式和 8 条 32bit carry 链，乘法/加法操作数在计算前选择。两个 context 经 operands/result 总线使用它们（第 181–205 行）；[thread_context.sv](../fusion_ten_trials_20260914/dataflow/d3_interleave/thread_context.sv) 第 75–95 行没有数据乘法器或另一个加法链。第 59–64 行把 ZADD/BASE_MAC 的 z+ALU 请求绑定，第 116–123 行在 grant 前冻结提交。RR/seq/阶段错位只改变启动条件（top 第 169–171 行），没有额外给候选端口。五类资源冲突与全收据 grant 账目闭合。
- “单 producer 算术”不是整个模块只做一种组合计算。D3 另有消费者 8 个 32×32 乘法器、8 个 64bit 加法器（[i24_consumer.sv](../fusion_ten_trials_20260914/dataflow/d3_interleave/i24_consumer.sv) 第 87–98 行），它们可与 producer 同时工作；两 context 各有四个局部 source 读 mux、support 比较/OR 和 priority encoder（thread_context 第 70–90、136–144 行）。这些是共同额外硬件，旧 D3 资源说明已列出，不能从 grant 计数推导它们零面积/零开销。
- phase/joint 确实借用消费者宽链：[phase consumer_stream.sv](../fusion_ten_trials_20260914/phase_borrow/consumer_stream.sv) 第 70–81 行实例化唯一 `wide_phase_alu`，consumer 优先且 grants 互斥；[wide_phase_alu.sv](../fusion_ten_trials_20260914/phase_borrow/wide_phase_alu.sv) 第 5–17 行只有八条 64bit 链，13/26/39 位断 carry。消费者 ADD_BIAS/ADD_IDENTITY 从外部 `add_y_bus` 提交（[i24_consumer.sv](../fusion_ten_trials_20260914/phase_borrow/i24_consumer.sv) 第 97–103、170–176 行），没有保留对应的第二套加法链。FP 转换、RNE increment、独立 producer 32bit 链和两组乘法器仍存在；“借用”仅覆盖 bias/identity 那八条链。
- 四P不是减少 T10 计算。52bit z 字存四个 signed13 域，mode2 仍读写相同整字，mode3 把四个同 T 的空间位置更新合并；见 [phase_core.sv](../fusion_ten_trials_20260914/phase_borrow/phase_core.sv) 第 23、127–150 行。各臂 z 容量均 520B，但它们的 416bit 向量口不能用 D1/D2/D3 的 208bit 口作同面积分母。

**支持，数据与退休闭环。** D1 的强控制和候选都在 STORE 捕获相同 result holding 并进 DIRECT_SEND，差分只有 p_mem 写使能（[forward_core.sv](../fusion_ten_trials_20260914/dataflow/d1_forward/forward_core.sv) 第 174–192 行），所以相同周期有实现依据。D2 的 source 物理地址 XOR 轮转与按行重载在 [consumer_stream.sv](../fusion_ten_trials_20260914/dataflow/d2_halo/consumer_stream.sv) 第 124–136、284–290、336–344 行；没有隐藏 copy。joint 直接逐行消费的状态保持/输出背压在 [phase_core.sv](../fusion_ten_trials_20260914/joint_selected/phase_core.sv) 第 181–201 行，mode2/3 不访问 psum；数组声明及 mode4 分支仍在，不能据此宣称面积已删除。

TB 由 RTL 的绝对 source、identity tile/row 请求返回真实输入，raw/J/I24 gold 仅参与比较；见 [joint stream_tb.cpp](../fusion_ten_trials_20260914/joint_selected/stream_tb.cpp) 第 67–75、90–106 行，D3 同结构。返回是可背压的同周期本地响应模型，不是实际 DDR/SRAM 延迟。输出在 stall 时检查数据/身份保持，最终检查 480×tiles 输出、raw/J 数目、retired_tiles、1848 冷配置和完整周期账目（第 78–86、111–116 行）。D3 只在 consumer 完成、480 个 beat 已接收且相应 core done 后切换/退休（top 第 363–397 行），奇数尾只启动 context0。

**待证，P2：语义端口预算不等于物理单口宏。** phase/joint 的 scalar z 读是无条件组合读，并持续送 producer 乘法（phase_core 第 53–60 行）；另有 ZREAD 整向量读（第 142–144 行）。没有额外有效提交，也没有只惠及候选，但当前写法保留多个读表达式及组合选择路径，不能凭 FSM 互斥或计数器就断言综合一定收敛为一个单口 SRAM。消费者 READ_A/READ_B 也写成两个数组读表达式。应在后续有必要时统一显式地址/使能，或用综合后的 memory/port/netlist 证明；本次不把未发生的物理实现失败报成 RTL 功能错误。

**待证，状态、元数据和组合路径费用。** D3 主存储/消费者暂存合计至少 40976B（两 source 3840、两 z 1040、两 psum 30720、Q1/Q2 4128、两 Q2 block 256、consumer 992），尚未计 local_source、src_masks、support、holds、控制和计数器。相应单 context 主项为 23048B，故跨 family 不同面积。静态配置明确付 1848 beat，动态 source/origin/identity 也付；`position_live` 和 `rank_live` 在 40 次真实 ZSCAN 中生成，未由 gold 预载（thread_context 第 173–180 行）。pending 优先选择、40bit decrement/mask、局部多 mux、FP 转换与舍入各自在一拍中的组合延迟没有建模；周期节省不能直接称 ASIC 速度、能耗或整网 FPS。

本次新验证使用 Python3.12、Verilator4.028 `-Wall --cc --exe`，五组均从旧源码复制后在新目录重新构建：[run_representatives.py](audit_dataflow/run_representatives.py)。全部 11 个模式分别跑 `159..161` 和 `19197..19199`，无 BP/有 BP、冷/暖各一次；另各跑 `128..191` 无 BP 冷暖并逐项比对旧整数计数。**110 命令全部通过，raw/J/I24 各 6420480 值；全部新 64-tile 整数计数与旧收据一致。** [representative_results.json](audit_dataflow/representative_results.json) 保留原始计数。

|3-tile 冷/无 BP|159 起（跨行）|19197 起（帧末）|
|---|---:|---:|
|D1 modes1/2|43316 / 43316|33315 / 33315|
|D2 modes0/1|51077 / 50309|38364 / 36828|
|D3 seq/offset/RR|47690 / 44509 / 42378|34977 / 32625 / 32071|
|phase modes2/3|51047 / 49370|38334 / 38154|
|joint modes2/3|42518 / 40841|31749 / 31569|

此外 [check_receipts.py](audit_dataflow/check_receipts.py) 用 `/opt/anaconda3/bin/python3.12` 独立检查旧全部结果账目、D3 grant、静态 metadata，并从原始 source bits、Q1/Q2、FP32 identity 重建 67 个 tile 的 raw/J/I24，各 257280 值吻合旧 gold；结果见 [receipt_audit.json](audit_dataflow/receipt_audit.json)。不依赖旧 verifier 的预测函数。

继续执行只推荐 **一项尚未测的接口融合**：以 D3 普通 RR 为底座，加入有界逐行 raw→consumer 通道，再接消费者宽链的真实争用借用。先固定两 context、共同 FIFO 槽数、416bit z 口、source/W/z/psum/ALU/wide grant；比较 RR+materialize、RR+direct、RR+direct+borrow，seq/offset 保留诊断。两 context 允许的 private 状态、消费者顺序和 loader 权限在各臂相同。必须出现并计数 producer borrow 与消费加法的同时请求，验证全压、饥饿、奇数尾及最终 beat；旧 phase/joint 中自然相位不重叠的“零冲突”不足以支持该组合。只有胜过共同权限的 RR 后才讨论新的执行差分；普通 halo 不计新标题。

根代理后续处理：本轮已把上述两项准确更正应用到旧D1/D2资源JSON，原SV与结果数值均未改变。这里“旧树未改”描述的是独立审阅阶段。
