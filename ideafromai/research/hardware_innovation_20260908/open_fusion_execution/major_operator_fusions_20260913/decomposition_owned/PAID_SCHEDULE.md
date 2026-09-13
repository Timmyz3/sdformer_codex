H8 共享减少了私有 pair 选择费用，但此执行策略仍没有净收益。最终 [paid_schedule.json](paid_schedule.json) / [程序](paid_schedule.py) 是有限 RF、公共事件遍历与端口位服务模型；不是逐物理地址仿真、RTL 周期或整个卷积层延迟。

范围为当前首帧 64 个采样空间位置，各自完整 T10、N96、K864。P1、两次 H48 wave，八个全宽加法 lane，系数/metadata 共用 256-bit 读端口，48-bit 源端口、256-bit 输出端口。所有臂获得相同硬件：每 lane 96×48-bit RF、2R1W，六个小计数 ALU、八个 3-bit 6:1 选择器。外加明列的公共控制器、10-bit 优先编码器、64-bit mask/地址状态、24-bit operand latch、48-bit 源响应 latch、512-bit 系数响应/拼接寄存器；没有 lane 私有 event walker。

| 最终策略，全部 64 个 P1/T10 样本 | 服务步数 | 相对 dense |
|---|---:|---:|
| 同重构 W 的 dense 执行 | 713544 | 1.0000 |
| 普通 3:4，私有 omitted-ID、压缩存储 | 1149608 | 1.6111 |
| 同一个普通 3:4 W，dense-zero 存储 | 713544 | 1.0000 |
| unsigned pair，私有 ID、独立计数 | 1938998 | 2.7174 |
| unsigned pair，私有 ID、六 pair 共享计数 | 1797852 | 2.5196 |
| signed pair，私有 ID、独立计数 | 1941124 | 2.7204 |
| signed pair，私有 ID、六 pair 共享计数 | 1799978 | 2.5226 |
| H8 共同 omitted-ID，压缩普通 3:4 | 729862 | 1.0229 |
| 同一个 H8 普通 3:4 W，dense-zero 存储 | 617255 | 0.8651 |
| H8 共同 pair-ID，unsigned 计数 | 890128 | 1.2475 |

H8 的共享选择使 pair 服务比原私有 ID 共享计数减少约 50.49%，同时十帧 AEE 从 1.1512492510 变为 1.1966744037，仍通过 NB0。它依然比同值 dense 执行多 24.75% 服务。H8 普通 3:4 在同值 dense-zero 布局下少 13.49%，但不能把这个控制的收益转记给 pair。原私有结构零即使减少标量 AAC，也会在八 lane 的公共 slot/time 发射中失去大部分收益。

这里已付的调度与存储：

1. 初始 216 个四通道 T10 源字各用一次 48-bit 读；每个 H48 wave 另读/检查全部 216 个缓存字，包括全零组。没有免费非零目录。源字 40 bits、剩余 8 bits 可附标志，检查仍收费。
2. 每波 480 个累加器用 60 次 SIMD8 零写初始化，另用 60 个 256-bit beat 写出 FP32 宽度结果。全部样本每臂共同支付 7680 次清零服务。
3. 全宽 add 的系数+psum 使用每 lane 两读一写；weight fill、mask 选择与 add 串行，不赠送额外 RF 端口。共享计数也按保守策略串行，可能的重叠尚未映射。
4. 每个八 lane tile 执行同一 coefficient slot、同一 time。每个非空槽先付一次 setup，再对 union-live time mask 的每项付一次公共 walker/operand-mask 读取/地址步骤，以及一次 add；没有使用 `max(per-lane events)` 假装八个私有压紧器。
5. 私有 unsigned/signed pair 的 count 与两个例外位超过每 lane 3-bit 选择宽度，按两次选择收费。H8 共享的 20-bit count masks 与 20-bit 例外 masks 也分两次选择；共同 omitted-ID 的三列 T10 mask 共 30 bits，同样支付两次。两词/lane 控制暂存可容纳至多 50 bits 的 signed masks。

RF 每 lane 分配为源缓存 27 词、H48/T10 累加器 60 词、系数缓冲 6 词、两个共享计数 bundle 合计 1 词、流状态和 operand masks 2 词，总计 96。六系数词的额外 16 bits/词可容纳 metadata。dense 一次最多四项系数能放入同一六词预算。小计数整数范围最多 [-2,2]；3-bit 输出宽度、sign/shift 控制不等同于已经综合的新定点流水线。

同精度存储预算如下。均为 N96、G216、FP32，含同样 384 B bias 桥接；原 Conv 实际 bias=None，桥接值为零。按最小编码计 B，不含对齐、地址表或可选 576 B 全局 lookup：

| 格式 | 系数+bias B | 最小 metadata B | 合计 B |
|---|---:|---:|---:|
| dense W | 332160 | 0 | 332160 |
| plain 3:4，独立 omitted 2-bit | 249216 | 5184 | 254400 |
| unsigned pair，独立 pair 3-bit | 249216 | 7776 | 256992 |
| signed pair，独立 pair 3-bit +共享 signs | 249216 | 7884 | 257100 |
| H8 plain 3:4，共同 omitted 2-bit | 249216 | 648 | 249864 |
| H8 unsigned pair，共同 pair 3-bit | 249216 | 972 | 250188 |

这些数量不是 metadata 免费的理由。当前端口账本允许按请求位数紧密拼接，尚未按权重文件的静态地址逐笔模拟跨界读、响应保留和跳过槽后的地址距离，因此不能称为完全支付物理系数请求。所有臂共同获得这种服务假设；后续物理实现可能改变排序或幅度。源的同 tap 四通道/T10 打包已作为接口输入，卷积滑窗 gather/packet 形成、跨 P 重用及 PSN/PED 均在范围外。

数值导出是 FP32，NumPy 分解验证是 Float64；48-bit 累加、32-bit coefficient/store 仅是这里的服务宽度。没有 θW 定点量化、RNE/饱和与输出截断验证，不能把此表或 GPU AEE 写成完整定点执行。保守串行也不是所有合法映射的最优下界，不能据此宣布 count 家族或普通分解路线无可行硬件切口。

最终表取代两个探索中间值：最早 236880/341997 的流水估计漏了 RF 冲突；随后 363014/473805 的串行估计仍含免费 lane 独立事件压紧。二者均不用于结论。程序保留的旧 `schedule()` 不被 `main()` 调用。
