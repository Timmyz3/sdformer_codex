# Round33：预计算 bitmask 控制器兑现

本轮把六份预计算 mode bitmask 与 runtime base pointer 做成了**可执行的有限供数模型、实际控制 RTL 测试和同条件 DC 综合**。动态与普通静态使用相同 bitpack 和读表协议；动作 6 保留无升级。旧目录及其冻结结果均未修改。

结果只覆盖控制核与一个完整热点卷积的有限供数。**128 维 voxel 统计、ridge、六路 readout、特征读取、系数存储和预测器执行费用均未纳入。** 没有完整 selector 的硬件收益、整网加速或能效结论。

## 合同与实际控制路径

[实施合同](IMPLEMENTATION_CONTRACT_V1.md)；[RTL](rtl/bitmap_controller.sv)；[位图内容](BITMAP_CONTENT_V1.json)；[来源绑定](SOURCE_BINDINGS_V1.json)。位图来自冻结 E3 的六个 row-major 151-tile 宏，不读取运行时 GT。

- 六份各 910 bit 的 mode mask，每份按 128B 对齐，共 768B，放在既有 2KiB statistics-RF reservation。每份 32 个 32-bit 字；有效 tile 为 0–909，四个剩余完整 tile 和 padding bit 都为 RNE4。真正尾区域始终 Q8。
- action 0–5 选择一个宏，6 表示不升级，7 非法。选择只在卷积 start 时锁存，保持到 finish；busy 状态的新 start 被拒绝，不改变当前选择。
- 单 32-bit descriptor 读端口、最多一个未返回事务。动态地址是 `{held_action, tile_id[9:5]}`，静态加载所选 mask 后基址为零。两者同样提取 `tile_id[4:0]` 对应的 bit，输出 Q8/Q4 的 start plane 0/4。
- 完整有效 tile 经 request/address、同步读返回、response/位提取协议；尾和无升级走寄存旁路。响应受 backpressure 保护。没有跨 tile 的描述字缓存；bitpack 降低容量，**没有把每个 span 的读字流量自动缩小 32 倍**。
- 初始化按共享 DMA 64B/tick、每事务 32-tick startup 计费，然后用单 32-bit 1RW 口逐字写 RF，二者串行。动态有效动作每卷积重复加载六图，普通静态只加载所选一图，无升级不加载图。这是保守重复 setup，不假定四层常驻复用。
- start/finish 各付一个接口握手拍。selector 是输入接口，未虚构一个 off-chip selector DMA，也未把上游预测器算成零成本。运行时动作 6 只有在上游决定之后才知道；模型中的低成本旁路不免除产生该决定的费用。

RF 初始化写入由测试平台/供数模型提供，运行时读由 RTL 控制核提供。初始化与 active 卷积互斥；新 DMA 引擎、RF 宏及完整初始化仲裁器没有在本轮 RTL/DC 中综合。

## RTL 与模型的边界

RTL 测得的 **3 拍完整查询、1 拍旁路**，是从 request acceptance 到 response acceptance 的无 stall 延迟，**不是相应的连续请求吞吐率或 initiation interval**。response 被接受后，状态机才返回 READY。本供数模型串行执行每个 span 查询，随后执行至少一行 PE/scale/store，覆盖回到 READY 的间隔；不假设连续无间隔发射查询。

RTL 实现的是给定 `complete_tile`、`tile_id` 的 descriptor 地址/位提取与控制协议。**空间行/列计数、完整/尾判断、tile ID 生成和 span 复用调度器尚未综合。** 它们由有限供数模型产生；旧模型逐行 gather geometry 的服务费用仍保留，不能据此声称这些逻辑已有 RTL 面积和时序证明。

模型复用当前连续 span 的返回 mode。对动作 6，它知道所有完整区域恒为 RNE4，因此将相邻完整区域聚合，直到 complete/tail 状态改变或 batch/output16 边界。RTL 自身对每个收到的请求执行协议；它不会自行产生或聚合请求。无升级聚合策略是尚待整合的调用侧调度义务，不能表述为控制 RTL 已独立实现全空间调度。

同样，外部 RF 的同步返回假设由测试平台提供。DC 约束了控制核端口，但没有验证实际 RF 宏与 DMA 的端到端时序。

## 实际测试与综合

`run_rtl.sh` 使用 Verilator，三个独立 top 分别为动态、普通固定图和无升级控制核。测试覆盖所有 910 个合法 tile、全部合法动作、真尾、完整标志下的 910–1023 非法 ID、动作 7、busy 更新、非法 finish、意外读返回、随机 memory stall 和 response backpressure。

最终共有 **27,188 次有效响应**，三组测试合计 **349 个非法协议/输入案例**；所有断言通过。测试使用实际位图字，独立黄金结果使用宏区间算术判定，未复用 RTL 的位提取逻辑。无升级核没有执行完整位图读，其结果文件相应将该延迟标为 null，不借用其他核的测试覆盖。

| 实际 DC top | cell area（库单位） | 4ns 下 worst setup slack |
|---|---:|---:|
| bitmap_ctrl_dynamic | 86.534000 | +3.32230ns |
| bitmap_ctrl_static | 68.698000 | +3.44778ns |
| bitmap_ctrl_none | 64.484000 | +3.35696ns |

全部使用与 E4 相同的 TT28 `tcbn28hpcplusbwp7t35p140tt0p9v25c.db`、4ns clock、0.05ns clock uncertainty/transition、0.2ns I/O delay、0.05ns input transition、0.010 output load 与 high map effort，实际重新运行 DC。详见 `results/dc_*/` 的 summary、area、timing、constraints、mapped netlist 和 run log；均无 violated constraints。报告中的低两位 start-plane 常量/短接警告来自合法输出仅 0/4，未掩盖成全设计无警告。

**面积只含控制核；不含 RF/SRAM、DMA、PE、scale/BN、空间调度器或预测器。** 此处比较不意味着静态控制已是所有可实现普通控制的面积/周期下界。

## 完整热点卷积供数结果

[供数源码](bitmap_supply.cpp) 从冻结独立 Q8/RNE4 双表示模型派生，原文件不改。完整几何为 T10/H240/W320/K864/Cout96。双表示及各自 scale 仍占 129,792B weight SRAM、22,016B weight RF；所有权重装载/refill、每行 scale、有限 input/output bank/DMA、完整输出与三次 BN pass 均继续计费。

新控制查询的 response latch 取代旧模型额外的 precision-switch 拍，不重复收费。每有效动作有 519,480 次完整 bitmap 查询和 14,940 次尾旁路查询；无升级有 57,060 次聚合旁路查询。`SUPPLY_REPLAY_V1.json` 保存全部计数及 192-word 访问直方图。

| 动作 | 普通静态 affine+BN ticks | 动态 affine+BN ticks | 动态附加 ticks |
|---|---:|---:|---:|
| 宏 0 | 229,693,995 | 229,694,165 | 170 |
| 宏 1–5（各自执行） | 229,694,508 | 229,694,678 | 170 |
| 6：无升级 | 207,641,722 | 207,641,722 | 0 |

动态相对同图静态的运行时查询协议、动作行、PE beats 和 DMA stall 相同。170 拍完全来自额外初始化：六图比一图多 640B，DMA 多 10 拍，单 32-bit RF 写多 160 拍。动态没有额外的每 span groupID 比较。四个同形卷积若分别冷初始化，附加服务的算术和为 **680 拍**；这不是整网延迟或收益，也不含评分器。

[独立核账](check_results.py) 从完整行几何及宏区间重新推导全部 14 臂的 Q8/Q4 行、核心 beat、span、读字地址直方图、初始化、DMA、RF 容量和 BN 顺序；全部通过，见 `results/ACCOUNTING_CHECKS_V1.json`。

旧 macro6 的 **513,674 拍**是旧 byte/groupID 简化服务模型的冻结结果，保持有效且不替换。新结果使用明确的同步 RF/response 协议，公共查询费用也改变，所以**不能直接跨两个模型减绝对周期来宣称总体加速**。本轮证据是：采用普通预计算图和 base pointer，可使同一新实现下动态选择的附加控制费用很小；旧附加费用不是不可消除下界。

## 后续公平预算与未付费用

151 动态对 154 静态的旧邻预算判断不能机械沿用。新控制附加费用须与真实 voxel feature、ridge、readout 的计算/访存/存储费用相加后，再确定静态能购买多少额外 tile。当前没有“预测器免费”或“151 动态已经胜过同预算静态”的结论。

拟接算法接口为当前 raw T10 voxel → 128 维统计 → ridge auxiliary 输出 → 六路固定 readout（具体候选及最终权重由算法实验冻结）。该接口尚未接入本硬件模型；也没有引入运行时 GT。BN/PSN、其他 consumer、跨层供数及整网时延仍未闭合。新的 768B 控制数据不解除先前 stage3 大权重超过 slot/RF 的未覆盖问题。

复现入口为 `reproduce.sh`，须在新副本运行以保护本轮冻结；结果索引 `results/SUMMARY_V1.json`。失败记录保留在 `results/failures/`：首次工具参数错误已修正；未发生 RTL 功能、独立账本或 DC compile 失败。本轮未使用 GPU。
