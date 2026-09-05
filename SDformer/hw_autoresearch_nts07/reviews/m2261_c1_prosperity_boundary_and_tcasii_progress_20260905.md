# C1 与 Prosperity 的边界，以及 TCAS-II 本轮推进

2026-09-05。源码/真实 trace/新思结果核对；不新建封存合同。

## 先纠正 C1 的对象

当前映射岛的 subset parent、residual、stable popcount/index order 均继承
Prosperity-compatible 映射。不能把 1.6945× vs strongest-zero 全部归因于
本项目电路增量，也不能说 Prosperity 没考虑片上容量。

实查 `rtl_m935_c1_match_pipeline`：

- 64-row task，非空、popcount≥2 的目标行选择最大 popcount 的合法子集，
  同分时原始 index 最小。不是“grant 时找仍存活的最早 parent”。
- `parent_live_q` 在目录建立时标记被选中的 parent，任务内不随最后消费者清除。
- `written_bitmap_q` 标记已经写好的结果；任务内没有容量驱逐。
- 九个 128×128 1RW 宏，逻辑只寻址前64行；不能虚构“首选被驱逐→备用救回”。
- parent read 与 retained-result write 竞争单口；残差算术可和读重叠，
  architectural psum commit 是另一接口。
- 原稿中的 eviction fallback、动态 last-consumer reclaim 和 grant-time
  recheck 叙事已删除。数值证据未改。

主来源：[Prosperity 论文](https://arxiv.org/html/2503.03379v1)、
[官方实现](https://github.com/dubcyfor3/Prosperity)。论文 stable sort 的目标是
满足依赖，不等于最小化物理生命周期；它的 k 也为16，因此“16bit AND
替代TCAM”不是凭位宽就成立的新颖性。该 AND/popc 匹配已在本项目 RTL 中。
论文 largest-index tie 与官方 artifact first-argmax 的区别还需作为直接对照，
不能把一个已有 tie 规则的收益当成新调度。

## C1 新探针：同一 forest 的顺序和结果驻留

`m2260_c1_hot_parent_probe.py` 保留 parent/residual；新序为确定性
first-child/next-sibling DFS。小槽采用 FIFO 替换，不使用未来最远访问 oracle；
引用数在子行完成后减少，溢出的仍需结果写回原 1RW scratch。

第一版10样本固定K子流的 1.08448×，经独立评审发现不遵循旧账本的
chunk→K 邻接，**不再作为原调度复现引用**。

修正实验：sample0–2 × 四Conv × spatial chunk0/23/46 × 每chunk全部432K，
15,552 tiles；各连续K段独立起停，不让隔开的空间段免费重叠。
计入独立DFS遍历和8次并行引用数重载。相对原稳定序、dead-write-only：

| 新点 | 局部 phase-pipeline ratio | 不重叠 ratio | parent SRAM 访问减少 |
|---|---:|---:|---:|
| 只改线程顺序 | 1.01866× | 0.95605× | 30.73% |
| 稳定序+2槽 | 1.01761× | 1.01110× | 32.72% |
| 线程序+2槽 | 1.06724× | 0.98585× | 97.71% |
| 线程序+4槽 | 1.07062× | 0.98802× | 99.97% |

这不是全层/全网或 RTL 周期，更不是总存储访问/能耗减少。两槽仅 payload
就有288B；双bank链接、order表、refcount合计最低480B，尚不含tag/仲裁布线。
同拍 hot hit 与 victim spill 读取不同槽，要求真实 **2R1W** 路径。
原九宏并未删除，逻辑寿命缩短不会自动减少最小编译宏面积。

独立打铁模型7/10、硬件就绪3/10、算法独创2/10（不是收稿概率）。
两随机seed的周期payload回归：984 replay、160,557 cycles、5,772,480
scalar compares，0 mismatch；pending/queue/consumer三类负控均检出。
尚缺RTL反压、epoch/输出bank切换及实际2R1W时序。先做强prior/timing对照，
不扩多候选matcher。

## 原有 C2 收口：新结果，不是新建议

同约束门控与hold修补后的 ordinary/TSBG：

| 轴 | logic area (µm²) | setup (ns) | hold (ns) | mapped→mapped FM |
|---|---:|---:|---:|---:|
| ordinary | 234,537.406509 | +0.122717 | +0.000004 | 77,180 PASS |
| TSBG | 235,700.638392 | +0.072405 | +0.000002 | 77,155 PASS |

0.496%面积增量替代新门控版本的0.0118%说法；旧0.0118%仍仅属于原ungated对照。
3ns、SSG-max/FFG-min、原I/O及uncertainty不变。极小正hold裕量不是post-CTS稳健签核。

checkpoint-derived **candidate INT8 FC weights** 的 gate-SAIF/PTPX：

| 预选窗口 | ordinary/B4 cycles | ordinary/B4 nJ | logic energy reduction |
|---|---:|---:|---:|
| low | 4717/4717 | 310.67/346.48 | **−11.53%** |
| median | 6733/4465 | 443.10/327.25 | +26.15% |
| high | 22294/7554 | 1467.03/555.64 | +62.12% |

TT0.9V25C、理想clock、zero-delay gate activity、无SRAM/CTS/SPEF。
不是FC量化精度准入，也不是人口平均/frame energy。
原稿已换为这组更强门控baseline，保留低复用负点。

## 已有 co-fill 与 Pro 新建议如何取舍

1. **先做完 co-fill 的公平物理对照。** 持久partial-valid LRU4三轴均已有
   数值VCS。新的一拍SRAM、bank-ready始终有效的9个pilot全部对上CPU；
   4320冷G48块上 union 对同样group-major但demand-fill为1.23276×，
   refill事务−36.16%，bank read数量完全相同。不能包装成额外稀疏率。
   24块略慢，最差609→616cycles，保留。
2. union DC：191,514.45µm²，setup +6.334ps，hold −12.154ps。
   matched group-demand 正在综合；未有配对面积/功耗，不借旧M2018数字。
3. **选择性保留值得下一步小测，不立即替换。** 原同streaming缓存对照已
   说明完全零拷贝无额外周期收益，warm反例841→1094cycle必须保留。
   下一B4需求要有实际就绪身份并计入descriptor预取/存储，不用全未来oracle。
4. **per-bank提前释放先统计可消费性。** 当前 bridge_ready/accept 是单一
   bundle级信号，不是四个独立consumer-ready；“一个慢consumer堵某bank”
   的例子不自动映射到本机。只有释放后真能发下一请求才有吞吐价值。
5. **8选4重组先测。** 必须有同层、同weight身份、已就绪的8行；不能拼接
   本来独立选取的first/middle/last B4并声称真实邻接。不跨神经元时间依赖。
6. **FFN结构裁剪本轮不启动。** 它会换checkpoint/状态规模/数值身份，当前
   TCAS-II收口不需要它。多候选parent也不优先于上述matched结果。

TCAS-II暂保持已有稿可读，不把未经验证的新候选塞入贡献。更聚焦C2的成稿
方向合理，但应由co-fill强baseline与配对能量决定，而非新名称数量。
[Bishop](https://arxiv.org/abs/2505.12281) 与
[SpikeX](https://arxiv.org/abs/2505.12292) 的权重共享/任务组织必须正面引用；
typed-signed协议能力不能冒充真实神经元是多值发放。
