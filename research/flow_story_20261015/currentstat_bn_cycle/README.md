# 外部current-stat BN的参数化事件模型

**已实现并运行：54处实际BN几何、统计依赖、rsqrt、归一化、有限端口/带宽与回压的CPU离散事件参考。尚未得到整网周期或硬件收益。** [bn_event_model.py](bn_event_model.py)不执行模型，不读权重/GT/质量；输入来源是[R5完整BN形状账本](../../flow_story_20261014/damage_allocation_probe/hardware_contract_complete_bn_audit.json)，SHA `9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567`。该账本包含54个外部BN、19,056个通道统计域、976,584,960输入元素及真实shape/stride，R8没有新增全BN hook。模型保留当前数学统计图，**分块树归约/rsqrt的FP32数值等价尚未验证**，因此不能把输出直接贴到现有AEE旁。

执行日程是population variance：`sum→mean→center/square/reduce→variance→eps/rsqrt→a=γ·invstd、b=β−aμ→y=ax+b`，无running更新。块内树归约、跨块累加和系数依赖都有显式DAG；normalize输入必须等待该组全部统计及affine完成。add/multiply/rsqrt单元分别配置端口、lane、II、latency和outstanding上限；外存与local SRAM分别配置read/write端口、word/cycle、latency、outstanding与是否共享读写端口。每个vector波组/传输burst占用issue槽并于完成时释放credit，buffer-slot复用由前一持有者退休控制。调度器跳到下一发射/退休事件，没有以总MAC÷PE替代依赖等待。

**真实stride决定两种普通日程。** 15处输入在每通道空间维连续，逐通道统计；另外39处（36 attention＋3 patch merge）通道维连续，按全C分组并行维护独立统计域。后者不能当作按通道连续存储，也没有免费transpose。burst在实际HW连续段内切分；分组buffer与8＋buffer_slots套标量/partial寄存器容量都检查。不支持的实际stride直接报错。局部归约会破坏working buffer，因此resident输入副本与工作区分别收费；若一组原输入加工作区放不下，`auto_channel_resident`使用显式spill重读。没有假设最大589,824,000B的BN输入整层驻SRAM。

三个已运行的参数例子均是**人为声明的敏感性条件，未经过SRAM宏、FP单元、RTL/DC或频率标定**。`example_queued.json`使用4个4096-word工作buffer、65,536-word local容量、16,384-word标量寄存器容量；add/mul各16 lane、II1、latency4，rsqrt单lane、II8、latency16；外存独立读写各16word/cycle、latency32、8 outstanding；local各16word/cycle、latency1、8 outstanding。所有word为4B。完整参数必须显式提供；CLI不替用户猜硬件规格。

| 已运行配置 | 相对上述示例的变化 | 54处独立BN按指定日程串行的周期之和 |
| --- | --- | ---: |
| [queued](sensitivity_queued.json) | 上述参数 | 626,142,077 |
| [one_buffer](sensitivity_one_buffer.json) | 相同单元/端口，仅1个工作buffer | 1,778,408,780 |
| [shared_ports](sensitivity_shared_ports.json) | 保留4 buffer；外存读写共享4word/cycle、latency80、2 outstanding；local共享8word/cycle、latency2、4 outstanding | 2,206,096,139 |

这里的差异验证了队列和端口会改变该日程，**不是某压缩策略的加速、真实芯片周期、最优dense下界或时钟测量**。不同BN之间目前串行，channel-major输入各通道也串行；全C分组允许通道统计并行，组内chunk在credit限制内重叠。更好的多通道/跨模块调度、单遍或分组统计、producer融合等普通优化仍应给所有基线机会。

三配置均严格重现4E add/sub、2E＋4C multiply、C rsqrt的账本：3,906,339,840、1,953,246,144、19,056。queued配置的6处BN允许当前分组输入resident；逻辑外存读2,883,712,992 word（含packed γ/β）、写976,584,960 word，local读8,786,339,520、写8,763,299,520 word。它们只属于这里声明的buffer、归约和输出落存日程，不能充当不可避免的DRAM流量；producer直送、输出consumer融合和重排机会尚未连接。

**检查与复算。** [test_bn_event_model.py](test_bn_event_model.py)的8项检查通过，见[cpu_checks.json](cpu_checks.json)。其中9个小DAG与独立逐时钟golden的每个任务issue/retire时间精确相同，另检查手工pipeline/credit、共享端口反压、尾块、完整统计/normalize屏障、resident容量与fallback、grouped布局、拒绝无证据transpose和输入配置不变。没有验证BN数值数据通路、rsqrt精度或RTL。

```bash
python3 -B test_bn_event_model.py
python3 -B bn_event_model.py --config example_queued.json --output NEW_RESULT.json --trace-limit 24
```

输出文件必须不存在。每个BN都有源名字、shape/stride、分组、周期及首组屏障时刻；13类几何日程保存resource占用/峰值outstanding、算术/传输量及有界issue/retire trace。API `channel_graph`＋`run_graph`可取得全部任务父依赖和时间，用于以后RTL小例子的golden。所有记录标明`full_network_cycles=null`、`RTL_validated=false`、`native_FP32_equivalence_validated=false`。

**与R8/MLP的连接合同。** [R8实际服务审计](../layer_repair_round8/actual_service_audit.json)已核六臂每FC164,200行及完整T10 producer范围；其SHA为 `46170d94f17d787368705097bde2771070cc129974e6af5d568dff5ae6ba2259`。下一整图调度器应以原生module path连接`producer_output_ready→BN input/domain→stats_done→normalize_output_ready→consumer`，让MLP、conv、attention及BN共用同一资源/存储实例。此BN参考当前假定输入已在外存、输出写回完成；不能无条件把其总和加到MLP静态issue上，更不能继续使用全BNfolded旧分母。R8六臂外部BN形状相同，但输入/统计值可能改变；若引入spike零跳过或数值近似，还需各臂真实活动/精度证据。旧Pk16×Pn16服务模板只是待实现假设；16lane局部核RTL/DC不是本模型或整网验证。

仍未关闭：实际FP32加/乘/rsqrt的II/延迟与数值合同；寄存器广播/partial读写端口（当前假定其可随算术供给）；SRAM宏/地址bank冲突、DRAM行/对齐/突发细节；γ/β离线packed表示与真实装载；全网producer/consumer顺序和buffer生命周期；通用融合/多通道调度强基线；RTL/DC及同精度完整质量—周期验证。端口模型已能表达带宽/回压敏感性，这些缺项不因此成为零成本。当前不据该表判断通过或未通过10%净PE门。

**地址合同补充（原结果保持封存）。** 当前tree把奇数n的未配对元素作为数学carry传入下一层，并用归约add数和聚合scratch读/写描述任务；尚未给出该carry的实际地址、压紧搬移或保留原址后的gather调度。特别是`n=5→3→2→1`时，不能未经证明就假设未配对值已经在下一层连续位置。全C分组也尚未证明交错bank映射、各lane地址、gather及冲突回压可按所设聚合端口吞吐完成。故这些结果是**数学任务DAG＋抽象端口的条件周期**，不是bank级可实现调度或实际SRAM周期；相关费用未关闭，不能当作零或直接用于10%净PE验收。

继续前最小地址验证应对奇数/偶数chunk、真实尾部和多C分组给出每级`src_address→dst_address`、carry策略、bank/port及读写冲突，验证无读前覆写、无丢失/重复、每通道统计域不混合；若需要复制/transpose/gather，增加显式事务与依赖后重新运行。可以选择原址保留的可实现索引器或显式压紧搬移，不预先规定额外开销为零。本次只补声明，不扩RTL、不修改原事件源码或敏感性结果。

独立逐拍golden核对的是**同一数学DAG在所给抽象资源规则下的issue/retire**，不是BN数值正确性、FP32等价、carry地址正确性或真实bank验证；这些必须各自有独立证据。初次交付README备份在[README.initial_delivery.md](README.initial_delivery.md)，与原[delivery_receipt.json](delivery_receipt.json)中的SHA一致；本补充另立receipt，避免改写初次交付结论。
