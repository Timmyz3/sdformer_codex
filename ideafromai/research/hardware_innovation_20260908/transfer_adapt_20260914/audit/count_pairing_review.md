# mode21 固定时间配对独立复审

范围：仅静态核对 `pair_psum_overlay/temporal_pairing/decomp_core.sv` 相对父 `pair_psum_overlay/decomp_core.sv` 的全部差异，读取对应 TB、验证器及现有 JSON；未浏览、未运行实验、未改实现。2026-09-14。

结论：未发现 mode21 相对 mode20 的功能性错误或隐藏新增数组/存储端口；现有证据支持“固定时间配对减少 count8 块服务”的局部增量。新增可配置选位及逆映射逻辑需要计入面积和时序，不能按免费排列计算物理收益。

## 合法性与接口

- 配置 π=[8,2,6,3,9,4,1,5,7,0]，语义为新时间索引到原时间索引。`L_GATHER` 令新门字第 t 位等于原门字 π[t] 位。原始 source_mem 及外部输入格式保持；已有完整 T10 门字使这里无需新增时间缓存。
- Q1/Q2 不依赖时间索引，所有类归约及累加分别留在每个时间坐标。新内部结果满足 p'[t]=p[π[t]]；不存在 endpoint、差分或 prefix/merge，只有一个 z 域。本改动没有改变同一输出内 K/秩的加法顺序。
- `DRAIN_READ` 读地址为 floor(row/10)×10+π⁻¹[row%10]；`result_addr=row` 保持原始顺序。π 是 0..9 的双射，故每十行恰好一一恢复。480 行先全部 STORE，再 DRAIN；等待 result_ready 时 result_data 和 row 保持，不引入额外读拍。
- count8 的类别频数上界、字节间 carry cut、count_high>127 时标量退休回退、双 P 负借位修正全部沿用 mode20。置换只重新分组时间槽，各计数数值的集合不变，不会放大计数上界。大于 255 类频数必须由既有编译器走 direct 的契约仍适用。
- mode21 的所有 mode20 门控分支均被纳入，直接残余和计数归约使用同一 π，避免两条路径落在不同时间坐标；mode14/20 仍走恒等映射。
- `cfg_kind=2` 在 IDLE 写入 40 bit；复位值为恒等排列。TB 用一个 cfg tick 完成配置，之后撤销 cfg_valid 再 start，静态配置总拍从 2721 到 2722，冷 64 tile 从 101089 到 101090；每 tile 的 source/origin 及 start 费用另计。warm 重复不重配 π，收费符合实际接口。
- 配置合法性检查仅在 VERILATOR 宏下：范围及重复值在 start 检查。实际硬件依赖软件提供有效双射；若 cfg 与 start 同拍则不能假定新 π 已生效。现有 TB 满足“先配置后 start”，本审阅未证明任意动态重配序列。

## 存储与端口

| 相对 mode20 的项目 | 静态核对 |
|---|---|
| 持久状态 | 仅新增 time_permutation[39:0]，5 B；inverse_time 为 4 bit 组合信号 |
| z | 仍为 8×20×26 bit，即 520 B，向量读写宽度 208 bit；无第二域 |
| p_mem/count overlay | 仍为 8×480×32 bit，即 15360 B；count 仍只占每 bank 前 160 行的 Q1 阶段 |
| count 存储端口 | 每 bank 仍一个地址、单拍至多读或写之一；无新读写表达式，原生命周期断言保留 |
| 计算与元数据 | 8 个 32 bit ALU、8 个 19×13 乘法表达式、class/representative/valid/hold 均未增加 |
| 新组合代价 | 四个 10 bit 门字各位的可配置选位，π 逆查找和输出地址选择；连接、扇出和延迟不可视为零 |

源数据仍从 local_source 取同一门字后选位，未增加 source_mem 端口。综合是否复制小 local_source 或形成多路复用网络取决于映射，代码级端口核对不能代替物理实现。相对 direct/原生 4P，class 元数据及独立 bank 地址选择本来就属于候选的额外成本；相对 mode20 未增加它们，不等于相对原生免费。

## 已有结果与机制归因

以下均取 `temporal_pairing/checks_held.json` 和 `checks_disjoint.json` 的无 stall、首次 64 tile 汇总；service=core+configuration+64 个 start。

| 集合 | mode14 冷 service | mode20 core/冷 service | mode21 core/冷 service | mode21 相对 mode20 |
|---|---:|---:|---:|---:|
| held 128–191 | 983682 | 809293 / 910446 | 787581 / 888735 | −2.3847% |
| source-halo-disjoint 4000–4063 | 1077632 | 859283 / 960436 | 834785 / 935939 | −2.5506% |

相对 mode14 冷 service 改善分别为 9.6522%、13.1486%；这是整个 count overlay+pairing 的收益，不应全归给本次排列。

- held：count 更新块 52595→42026，退休读取块 6482→6195，count bank 读/写各 337038→269604；core 节省严格为 2×10569+2×287=21712，减新增配置 1 拍得到冷节省 21711。
- disjoint：更新块 64664→52711，退休读取块 6989→6693，bank 读/写各 415438→339202；core 节省 2×11953+2×296=24498，冷节省 24497。
- 两集合各自 mode20→21 的 first_issues、aux_events 和 Q2 mac_issues 均不变。收益来自把同时活跃时间更集中地放入一对，减少块服务；没有少算等价输出。
- 每个集合现有 768 条命令、2949120 个逐值 raw 输出检查，覆盖三 mode、两 stall 配置、两次连续遍历；各 245760 个不同 gold 坐标，不把重复检查说成不同样本。held 的 512 条 mode14/20 旧记录全部字段复现；disjoint 的旧记录匹配数为 0 是没有旧对照文件，当前三 mode 已各自核对 gold。
- small/short 的 checks JSON 亦通过。本审阅读取通过记录和验证逻辑，未重新执行。固定 π 来源于 earlier endpoint 校准，当前未再拟合；disjoint 是同一源帧上与校准 halo 分离的空间测试，尚非跨帧泛化。

父任务提供原生 4P/416 bit 口冷 service 为 933565 / 996457；对应 mode21 改善 4.8020% / 6.0733%。本次未读取该强对照代码及 JSON，因此仅列为父任务给出的外部比较，不能代替独立复核。208 bit z 口小于 416 bit，也不能抵消 class/mux 成本后自动宣称等面积更优。

## 创新性判断

本次 mode21 增量创新性暂评 **3/10**：时间重排、共现配对和减少稀疏块访问都是通用手段；没有新差分表示、新算术或新存储复用机制。这里值得保留的是：从另一目标已固定的 π 直接迁移到 count8 四 P×双 T 块更新，付出 5 B 配置及真实逆映射后，在两套 raw 集合上独立降低服务拍数。此分数为机制增量判断，未做本轮文献排重，不是首创认定。

**真 X（一句）：** 利用完整 T10 门字可重索引的接口，把既定时间共现配对转为既有 count/psum overlay 的块访问消减，保留原生输出且不增加 z 域。

**最强反对（一句）：** 所有节省都能由普通时间配对降低活跃块数解释，而新增选位/地址逻辑的 Fmax 和面积未测，因此目前更像有效调度优化，尚不足单独支撑强架构创新。

尤其只有配对集合 {(8,2),(6,3),(9,4),(1,5),(7,0)} 对当前无 stall 块计数关键；排列的全局先后次序不是这里新增的依赖机制。端点路径的“相邻差分路径优化”不能直接充当 mode21 的新颖性。

最重要后续对照：在同一物理存储/时钟约束下综合 mode20、mode21 和原生 4P，计入 class/mux 与配置，确认新增排列的周期节省没有被面积或关键路径代价吞掉。

证据入口：`pair_psum_overlay/temporal_pairing/decomp_core.sv`、`stream_tb.cpp`、`tb.cpp`、`verify_run.py`、`checks_{held,disjoint,small,short}.json`；父级资源契约为 `pair_psum_overlay/resource_contract.json`。所有路径相对 transfer_adapt_20260914。
