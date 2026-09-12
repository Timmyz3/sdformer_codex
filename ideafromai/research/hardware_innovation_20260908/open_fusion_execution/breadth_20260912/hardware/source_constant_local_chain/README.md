# PoT2 源接完整局部双消费者

**四例已实际跑完。约 35% 的源叶周期下降，在完整局部链中变成 7.61–9.19% 的净服务下降。** dense 和 lifting40 都获得同样的两项常量投影权限；两者的新 source 函数不能继承未量化父函数精度。

| 新源函数 | 窗口 | 未投影父函数服务槽 | 新完整局部服务槽 | 相对父函数减少 |
|---|---|---:|---:|---:|
| dense PoT2 | corner | 2,055,566 | 1,887,086 | 8.1963% |
| dense PoT2 | interior | 2,743,658 | 2,491,384 | 9.1948% |
| lifting40 PoT2 | corner | 1,930,822 | 1,783,904 | 7.6091% |
| lifting40 PoT2 | interior | 2,585,116 | 2,379,946 | 7.9366% |

同样有 PoT2 权限时，lifting40 比 dense 的局部服务少约 5.47% / 4.47%。这个差值同时受各自训练参数、门活动与真实消费者请求影响，不能全部算作提升因子结构的新颖性。普通 PoT2、CSE 和供数底座本身不是标题 X。

执行从原始 I24 halo 开始，在同一个 96 RF、128 KiB state/coefficient、SR64/SW64/CR256 的 Machine 上付费运行新源程序、完整 K864 preview、静态 BN1、T10 sn2，再由实际驻留门进入 Conv2 U16/F、BN2+rawI24、投影门及连续 PED U24/V96。没有将源与消费者两张旧表相加，没有删 raw I24，也没有把量化后的门仍当作父函数门。

四例共 1,934,080 项新 source、preview 和整数消费者检查均零差。独立 CPU preview 保留实际升序 FMA/RNE 边界；GPU 两个新函数的小捕获及既定 diverse10 正由算法代理在队尾完成，因此当前只是新函数的 CPU 完整局部计费，尚未声称其全链 RTL 或 GPU 位级等价。

必须保留两个普通 dense 源对照：本表使用较快的公共 CSE（同函数源 RTL ready 319 周期/tile、40 工作 RF）；同一 PoT2 dense 函数的逐行两链 CSD 也已经通过原 ROM/RTL（472 周期/tile、13 工作 RF、229 字）。后者本轮没有重新交织消费者，不能从源时延加减推算其完整链，但它证明 40 RF 不是 dense 的状态下界，下一轮共享状态/交织必须面对这个合法控制。

结果见 [summary.json](summary.json) 和四个同名原始 JSON。输入参数直接来自 `breadth_20260912/source_constant_probe/{dense,lifting40}/deployed_constants.npz`，消费者参数没有再次量化。复跑使用已有 Python 3.12 执行 `run.py dense corner` 等四个固定组合，再执行 `summarize.py`。旧 matched_local_chain 的源和结果均保留。

## GPU收口更新

实际新函数首帧的两个halo已由root接管队列捕获并逐项核对；源门、sn2门、updated-I24、投影门及PED整数端点和参数全部相同，浮点preview差异仍显式列出。见[gpu_alignment.json](gpu_alignment.json)。以上早期“待捕获”状态至此关闭；有限窗口核对不是整网等价，也未改变原服务表。
