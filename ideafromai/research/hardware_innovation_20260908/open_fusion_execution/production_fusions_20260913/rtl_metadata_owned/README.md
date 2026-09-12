# 独立 H4 摘要与许可位原语：功能验证

2026-09-13 已用本机 **Verilator 4.028** 完成功能仿真，结果为 [PASS](results.json)。本目录只有隔离的 metadata 算术、一个结果寄存器及 C++ 时钟测试台；没有修改生产 RTL，没有综合、布局布线、PPA、全链 RTL 或硬件加速结论。

## 位合同

[metadata_primitive.sv](metadata_primitive.sv) 的三个操作与当前 CPU 原型一致：

| operation | 输入与计算 | 返回 |
|---|---|---|
| 0 / IMETA_GATE | 128bit 为 8 个小端 uint16 门字。每字仅低 10bit 有效；前四字、后四字分别 OR 成一个占用位。group 必须为 0、2、…、22 | `summary_i OR (occupied << group)`，24bit |
| 1 / IMETA_WORD | 仅消费输入低 64bit 的四个 uint16，忽略高 64bit；低 10bit OR 成一个占用位。group 为 0…23 | `summary_i OR (occupied << group)`，24bit |
| 2 / IMETA_PERMISSION | 两个 24bit 掩码分别来自既有 geometry RF lane4/5，两个有效位置位来自 lane6 | `permission[p] = valid[p] AND geometry[p][group]` |

摘要是 **OR 累加**，与 `source_metadata/metadata.py` 当前实现逐位一致。调用者先清 RF94，每像素每个 H8 组恰提交一次；该清零和源门产生过程在本原语之外。摘要并未在模块内另设一份状态，`summary_i` 从既有 RF 提供，结果交回原 RF 写回路径。geometry 的其他 lane 不属于本模块的输出；完整 RF lane 保留由调用者负责。

有效索引范围是接口前提，仿真覆盖合同内输入。每个 uint16 的高六位必须被忽略，测试明确包含这些位；SV 仅对这一有意未使用的输入位关闭 Verilator UNUSED 提示，其他 `-Wall` 检查保留。

## 时序与背压

一个 elastic 结果寄存器提供 `in_valid/in_ready`、`out_valid/out_ready`；输出被阻塞时，valid、操作类型、摘要、占用位及许可位全部保持。寄存器空闲或当前结果被接收时才能接收下一操作，支持每拍退休并替换。

测试台用外部 RF 写回封装解释 CPU 的两槽延迟：issue 槽 t 的操作在边界 t+1 到达已寄存结果，在下一个边界 t+2 写入既有 RF。无背压时逐次检查恰为两槽；结果被阻塞五槽时写回相应延后五槽。调用者仍须执行 RAW/WAW、单 issue 和共享 RF 写回仲裁；本模块不实现该全局调度器，也不额外保存多个 pixel 摘要。

## 实际测试结果

[tb_metadata.cpp](tb_metadata.cpp) 使用独立的逐 lane、逐时间位参考，固定 seed `0x20260913`：

- 1536 个 producer 单比特输入：12 个 H8 位置 × 128 个输入位，含每字高六位不产生占用的情况。
- 3072 个 consumer-word 单比特输入：24 个 H4 位置 × 128 个输入位，额外确认输入高 64bit 不被消费。
- 384 个许可位真值表：24 个索引 × 4 种位置有效位 × 4 种两路目标摘要位。
- 2048 个混合随机事务，包含任意旧摘要、门字与两路 geometry 掩码。
- 64 个连续 pixel，每个先清外部 RF，再执行 12 次实际 DUT 结果写回；共 768 次 RF 更新，对完整 24bit 摘要和 uint64 sidecar 零扩展核验。
- 4716 个结果背压保持周期、6452 对相邻周期接收，验证保持以及同时退休/替换。
- 7809 个事务被接收，7808 个正确退休；最后一个在显式复位时失效。接受数等于退休数加复位丢弃数，无无故丢失或重复。

复现命令：

```bash
bash run_functional.sh
```

[run_functional.sh](run_functional.sh) 用 Verilator 生成 C++ 功能模型，再用 make 编译测试台；本机版本没有 `--build`，因此分两步运行。临时编译目录位于 `/tmp`，脚本结束后清理。保存 [build.log](build.log)、[simulator_version.txt](simulator_version.txt) 和 [results.json](results.json)。

## CPU 接口静态复核

另核读了 `source_metadata/{PLAN.md,metadata.py,run.py,run_binding.py}`，以及新增 sn2 摘要段。以下是代码审阅，不能当作这些流程已经由本 RTL 验证：

- 源 producer 摘要来自实际 128bit 门 collector；consumer-built 强控制从真实源 SRAM 经单 SR 响应读取各 H4，清零、更新、sidecar 存储均调用实际收费操作。传入目录的 gold `words` 只用于 shape，实际掩码来自 SRAM 摘要，实际非零门字仍从 SRAM 读取。
- 几何 RF lane4/5 在原布局中为空，lane6 保留位置有效位、lane7 保存许可。当前源摘要用 RF94，源程序和门 RF95 不与其重叠；目录 geometry RF80–88、CACHE RF64–72、NRV RF93 的使用时段分开。
- 仅 ip1 被许可时，首个 `IWORD_PAIR(reset=True)` 会清掉 ip0 旧内容；两位置都不许可时，旧 CACHE 不会被 `INRV_CACHED` 消费。保留 c-major、rem-minor 的 K864 顺序，掩码零的 H4 可跳实际门字读取及 NRV decode，但并未免费生成非零 NRV。
- 新 `sn2_with_summary` 在原 sn2 的 RF0–39 运算之外用 RF93/94 保存两个位置摘要。每次真实门 commit 后更新；清零和最终 sidecar 存储均收费。gate_base 到 pixel 的 `((gate_base-GATE)//192+p)` 映射与原 P2 输出布局一致；单位置尾组只使用 RF93。摘要完成前经过 `wait_reg`，没有提前读取在途 RF。
- `consumer_both` 扫描真实 sn2 SRAM 生成相同摘要；`producer_both` 从真实 sn2 collector 生成。整数目录传入的 `shape_only` 不承载载荷；source_base=GATE、meta_base=SN2_META、dir_base=原整数 DIR，后续算术仍消费实际 NRV/SRAM。
- 源 sidecar 从 112640 起，每 pixel 一个 64bit 字，低 24bit 有效；新增 sn2 sidecar 从 114688 起，与源 sidecar及当前 source/preview/整数工作区不重叠。

初版普通 consumer-built 控制逐 H4 执行读取、IMETA_WORD 和 RF 等待；producer 每次更新两个 H4。该普通控制有真实收费，但其顺序扫描尚不能被当作所有普通实现的最优者，例如既有 Machine 允许的 read+op 同槽预取没有在初版扫描中使用。producer 与 consumer 的差额须按实际控制版本陈述，不能全部归因为普通无法复制的生产接口收益。

本次 RTL **未实现** SRAM sidecar 读写、IMETA_GEOMETRY 的完整 RF lane 写入、坐标边界生成、源或 sn2 算术、H4 cache 更新、NRV 发射、完整顺序比较、全局仲裁或端到端服务计数。这些 CPU 流程与数值检查应引用其自身运行产物；不能把上述原语 PASS 写成“RTL 全链通过”或“RTL 加速”。新增摘要与控制逻辑的面积/时序成本尚未测量，同端口容量不等于同面积。

## 最终独立复核：扫描强控制与共同 P1 后缀

2026-09-13 再次只读核查最终 `source_metadata/metadata.py`、`strong_suffix.py`、`run.py`、既有 `flows.retained_z` 及 36 份最终 JSON；没有重跑 CPU 或 RTL 测试。**本节替代上文关于“初版逐 H4 串行扫描尚未给予预取”的限制：该问题已修复。** 未发现本轮实现的实际响应时序、状态容量或共同后缀公平性存在阻断问题。

### `scan_summary` 的响应和提交顺序

最终普通 source/sn2 扫描均调用同一 `scan_summary`。首字读取与 RF94 清零同槽发起；经过显式响应槽后，`saddr` 必须等于本 H8 首字地址。首个八字节保留在既有 16B collector，再请求并接收第二个八字节。此时形成的 16B payload 完整后才发射 IMETA_GATE，且该操作和下一 H8 的首字预取同槽发生。下一轮之前仍有显式响应槽；最后一组不预取越界地址，最后一笔摘要必须等待 RF94 写回后才写 sidecar。

因此，下一次预取不会改变当前已经收集的 payload，也没有在单 SR 响应之外保留第二个响应队列。上一笔 16B payload 在 IMETA_GATE 发射后已经死亡，下轮首字可复用 collector。固定端口背压会在原 `advance` 内延后请求/操作接受；collector 的活载荷在接受之前保留。一个 SR 请求与一个整数操作共槽是既有 Machine 能力，没有增加第二 SR 或第二 issue。摘要旧值来自 RF94，门数据来自 collector，没有第三 RF 读口。

最终消费者与生产者均每个像素执行 12 次同语义 IMETA_GATE；普通扫描已得到两字聚合和下一首字预取权限。普通扫描 ready 每像素实收 24 次 SR、12 次 metadata 更新及清零/响应/最后提交，共 52 槽；源 producer 的对应增量为 38 槽。这里的 source report `summary_slots` 只表示每像素一次摘要存储，不能替代完整 metadata 成本，完整费用应看阶段与操作计数。

source/SN2 摘要仍在两个已有地址区，corner 共 81+64 个摘要字，interior 共 121+81 个摘要字；最大有效 sidecar 为 source `[112640,113608)`、SN2 `[114688,115336)`，与当前工作区不重叠。普通扫描使用同一个 16B collector，发生在 source 或 sn2 已结束之后；没有同时运行另一个门收集器的阶段。新增 metadata 逻辑的面积/物理时序依然未计量，同 RF/端口容量不能写成同 PPA。

### `strong_suffix` 的数值和观察器

共同后缀从既有 `flows.retained_z` 克隆，U24 的 30 个向量 RF 保留到两块 V/H48 完成，U 供数 RF88/89 在 V 开始前已经死亡，V 输出只使用 RF30–89。U、V 均使用原 q 中的 exponent；原 U RNE→sat、V RNE→sat→bias→sat 的顺序和既有 indexed MAC 保留。

新增的 U 观察发生在全部 U `complete` 之后，复制 RF0–29 到返回用 host 数组；V 仍读取实际 RF0–29，观察器不提供后续操作数。该位置原 `complete` 已等待所有 U 完成，新增 drain 不需要额外保留活运算。它没有新增片内 latent 缓冲，也没有把已量化 U 替换成未完成的累加值。

run clone 只替换原 P2 PED U/V 调用，保留真实原 I24 供数、Conv2/merge、投影门、PED 与门外送和最终检查；所有模式在调用前都同步 clone 自己 globals 中的 `directory`，没有冻结到错误的普通/摘要目录。该后缀只支持本次 `count=2`、rank24、非 late-V，代码有明确断言；不作更一般的接口声明。返回的 U 是输出观察，本轮 JSON 并未新增独立 U 金值检查，不能额外声称所有 U 中间值都有新参考核验。

### 36 份最终产物交叉核对

完整集合为三函数 × 两窗口 × 五种 ready 模式（baseline、consumer、producer、consumer_both、producer_both），加三函数 corner 的 baseline/producer_both 固定 stress，共 **30+6=36** 项，没有缺项或重复键。

不只检查 `common_consumer` 标签：36 项都实际包含 `P1_retained_Z_U/V` 阶段、480 次 U clear、1920 次 V clear 和 7200 个 PED 外送 DMA 槽；旧 `resident_U_ped`/`resident_V_ped` 阶段均不存在。每项阶段总和等于唯一全机 service_slots，consumer 时长等于全机结束减真实 producer_end。每函数/窗口的五种 ready 模式，IMAC_INDEX、IRNE、ISAT 和 IADD_COEF 指令数分别一致。

全部 source、preview Z/raw/BN1/sn2、updated、projection_gate、PED 检查的 differences 都为零。最终普通扫描的 IMETA_WORD 计数全为零；每像素 12 次 IMETA_GATE、12 次第二字预取、一次清零/首字预取及一次摘要存储，都与窗口像素数一致。两种 consumer 模式的阶段计数也直接确认每像素 52 槽。由这些阶段与计数可排除把旧顺序扫描或旧 P2 后缀结果仅改标签后混入最终集合。

共同资源字段均为 RF `[96,8,48]`、512 字源 ROM、131072 字节状态和系数池。该检查证实当前代码与存储结果的执行证据相符；未使用哈希，也未声称构建身份或形式化证明。

### 修正后的差额边界

ready 时，相同位置的 producer 与完整优化扫描 consumer 的生产接口差额在三函数中完全相同：

| 摘要范围 | corner 少用槽数 | interior 少用槽数 |
|---|---:|---:|
| 仅 source：consumer − producer | 1134 = 81×14 | 1694 = 121×14 |
| source+sn2：consumer_both − producer_both | 2030 = 145×14 | 2828 = 202×14 |

producer_both 相对 consumer_both 的全边界减少为 0.110%–0.136%；摘要索引/跳零相对 baseline 的其余变化不能全部归给生产接口。该 14 槽/摘要像素差额在普通 matched34、dense、lifting 中一致，尚不能支持 lifting 独占机制。

也不能统一把 both 当作每窗最优：dense corner 的 producer 为 1845872 槽，优于 producer_both 的 1846623；普通 consumer 为 1847006，优于 consumer_both 的 1848653。应区分同布局机制消融和各窗最强已测普通控制。最终 stress 只有 baseline/producer_both 对照，没有 consumer_both stress，不能从中推出相对优化扫描的压力差额。

本节只结束本次代码与结果审阅。原语 RTL 的既有 PASS 不扩展成扫描器、P1 后缀或完整 NRV 链的 RTL 通过；仍不作整层/整网、PPA、普遍调度最优性或整个生产接口家族的正负裁决。
