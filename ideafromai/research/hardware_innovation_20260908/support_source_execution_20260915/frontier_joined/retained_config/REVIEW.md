# P32 参数保留独立复审

审阅范围为本目录相对 `../active_prefetch/` 的生成脚本、实际 SV/TB 小差分，以及最终 `expanded_old.csv`、`expanded_new.csv` 各 340 行。未重跑 RTL；用 Python 3.12 独立重聚合并逐任务核计费与前版工作量。此前同 P32 重复读参数这一普通强对照缺项已补齐，应以本目录为最终比较口径。

## 生命周期与协议

- `joined_core.sv:119` 在每个 top 命令进入 IDLE 时把 point 清为 0，随后完成 DREQ/DRSP 才到 PSTART。故 `point!=0` 的 reuse 提示只对同一命令的 P1..31 生效；上个命令留下的 point=31 不会污染新命令 P0。
- `frontier_source.sv:187` 的唯一功能改动为复用时直接 GSTART，首次仍 BREQ；A、τ、D/info 或 roots/rank 在此 P32 内确实不随 P 变化，外层 mode 也保持不变。静态寄存器已由 P0 按本模式加载。独立 source 模块本身信任 reuse 提示，合法性由此 wrapper 保证；本验证不代表任意模型热切换协议。
- 每次 start 仍失效化 slot_valid、batch_valid、cache_valid，清 FIFO、source_seen、pending、pv、pf_done 与 group_id；GSTART 重建代码/节点状态。旧 X/图数据位不必物理清零，但不能由失效标签被消费。没有跨 P 偷留 X 槽或图缓存。
- 请求、预取、DRAIN、输出保持及源到桥握手逻辑未改；五臂共享参数驻留、活动前沿 PF、128B X holding 和 128B 图 cache。没有新增参数数组或读取端口，只增加 reuse 输入与状态选择。106 个独立乘法单元及既有局部存储边界沿用父审阅，256KiB 仍只是外部参数/X 池，不是总存储。
- TB 的 static 配置断言由 960 改为 30，graph 由 672 改为 21，均为整个 P32 命令总数。每个新 top 命令仍有实际冷配置请求；不是免除首次配置。最终测试同一 executable 初始 reset 后遍历多个模式/输入，可覆盖这一 wrapper 的连续命令路径。

## Root 映射与 TB 口径

`frontier_source.sv:43` 保持 local root 地址 222；wrapper 加 5952 后为物理 6174，bank 6。`tb.cpp:71` 为 code/class 在 6174 装各自实际 root，两个模式都只付这一 root word 请求，所有地址偏移保持 bank 对齐。父 PLAN、RESULTS 及本目录“统一 root bank”的表述与代码一致。它消除了根配置放在不同 bank 的偏差；图的实际访问路径与 BP 日历仍允许不同，不能称所有地址布局已中性。

TB 仍逐实际 producer_active 比较 (P,c,t) 的 U/gate，每个批内独立去重 channel 以核 source_channel_refs；未生产的 oracle 门不计作实际核对。每个目标 t 仍从全 10 个 s 做完整点积，参数保留没有把源 gate 或 code 答案变成 DUT 输入。最终 H384 的 Y/U/gate 比较及同 mode4 消费者对照保持。

## 独立数值与计费复核

680 个唯一任务全部有 PASS 收据。逐任务与 active_prefetch 前版匹配：源 produced_pairs、channel_refs、scalar_mac、X_words，以及消费者配置、系数、MAC、更新和桥读写全部相同。BP 下图预取请求及服务日历可以改变，未要求这些时序结果逐项相同。

独立断言全部通过：source_config_words 为 static 30 / graph 21；dictionary_words=12；backend_config_words=1588；bridge_writes/reads=1920/1280；source_scalar_mac=10×produced_pairs；bytes=16×words；总 words 等于字典、源配置/X/图、后端配置/系数之和；DONE 周期等于 10 个 wrapper 状态周期之和，各阶段分项也闭合。无 BP 每任务相对前版恰好省 static 1860 / graph 1302 拍，分别等于 31×30×2 和 31×21×2。

最终消费者 Y/U/gate 各核对 83,558,400 值；实际源 U/gate 各核对 11,263,822 值，最终 code/class 标签 1,305,600 个。每任务 DONE 比最后 gate 晚 2–4 拍，不能统一减 2。下面只报 `cycles_to_last_gate`。

| 31 个非 pair 选择训练帧，共 992P | W′ ready / BP | W″ ready / BP |
|---|---:|---:|
| static64 + next-X PF | 1,931,780 / 2,059,064 | 1,931,780 / 2,058,873 |
| one code | 1,839,283 / 2,155,990 | 1,839,283 / 2,155,853 |
| resident frontier code | 1,723,094 / 2,029,718 | 1,723,094 / 2,029,589 |
| one class | 1,835,560 / 2,153,684 | 1,820,543 / 2,135,857 |
| resident frontier class | 1,719,032 / 2,023,947 | 1,717,909 / 2,021,488 |

普通 frontier code 对 static 的 ready 节省均为 10.8028%，BP 仅 1.4252% / 1.4223%。两函数 BP 都有 6 帧负例：train6、7、13、26、27、30。W″ 的 class 对同函数 frontier code 只再省 0.3009% / 0.3991%，ready 31 正，BP 30 正 1 平。W′ 对应 0.2357% / 0.2843%，BP 28 正、2 负、1 平。

更少源 MAC 没有带来更少总流量：W″ ready 下 static / frontier code / frontier class 分别为 4,043,296 / 6,131,088 / 6,097,216 bytes。code 的实际 target-pair 数 422,403、channel refs 69,949、X 112,474 words；class 为 418,883、68,536、110,504。refs 是批级去重的使用次数，不能直接当成物理 X 读次数。

## 结论边界

没有发现本小差分的数值、生命周期或端口合法性错误；父 RESULTS 的最终数字与独立聚合一致。参数驻留、活动预取和多路径图执行都是普通权限，不增加新颖性评分。应保留正但薄的 class 增量，同时把普通 code 的 BP 基线收益修正为约 1.42%，不继续使用未保留参数时约 3.3% 的主张。

这里证明的是两个各自固定整数函数、固定 P32/H384 和当前 bank/BP 合同下的完整子链。31 帧仍属于训练缓存，W′/W″ 的新网络 AEE 未闭合；不能跨函数借质量、外推动态 BN/FC2/整网或等面积/Fmax。此前“批内参数应只读一次”已完成，不再列为待办。

