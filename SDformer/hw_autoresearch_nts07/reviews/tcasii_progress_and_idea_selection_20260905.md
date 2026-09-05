# TCAS-II 收口与新增机制取舍，2026-09-05

## 判断

当前是可继续打磨的 C1 + C2/TSBG 电路组件稿。最优先补齐的证据是 C2 的 matched logic/SRAM energy 和 setup/hold；无需为普通 TCAS-II 投稿临时改成 FPGA 项目。保留两条主贡献，新增优化若成立就嵌入 C2，不再同时讲完整光流系统、算法创新和第三个硬件引擎。

这是证据审查与执行建议；录用由编辑和外审决定。当前工作仍有实质缺口，不能称为 Strong Accept 或已完成投稿封装。

## 现场证据

| 项目 | 可以引用的当前范围 | 尚缺什么 |
|---|---|---|
| C1 | strongest-zero 对比 1.6945× same-ledger 周期模型；九宏岛 166,514 μm²，3 ns setup/hold 通过 | 51.84M-row 模型与九宏岛是不同证据范围；跨序列与完整容量物理实现仍有限 |
| C2 K8 | 等带宽 K1×8 对比 1913/1945 周期；4.541× throughput/logic-area；logic area −77.61% | 不把面积效率写成同资源周期倍率 |
| TSBG | 2,880 固定区域 VCS workload，1.8345×、read −58.13%；matched logic +0.0118% | matched hold 仍有 −16.4 ps；matched energy 没有正结果 |
| full-token robustness | 11.16M quartets，2.0874× VCS-calibrated CPU model | 仅为建模鲁棒性，不能提升为 RTL、full-network 或 energy/frame |
| pre/post-read 消融 | M2231仅解析恢复，M2232独立结果审阅98/100；24 commits/axis、4608 products/axis、0 mismatch；2304/2304/576 reads | 已准入定向功能/省读因果，非population、mapped area或power；旧M2215失败保持不变 |
| ICC2 库准备 | M2239只解析恢复、M2240独立结果审阅98/100；Milkyway option可query/set/readback，local_output_dir本版本未注册 | 后续转换源删除不支持的option并设已验证路径；没有转换/NDM/P&R结果，不与当前功耗DC抢队列 |
| 功耗源修复 | M2233补齐M2160传递依赖，M2234独立审阅98/100、P0/P1/P2=0；已启动唯一一次M2235 | ordinary/TSBG × low/median/high 六点；仍须DC/PTPX与M2236结果审阅，当前不能引用新功耗 |
| 稿件 | 新增通过审阅的pre/post-read因果消融；五页Letter，195词abstract、6 keywords；严格PDF检查PASS，末栏仅references | authors/funding等仍待填；功耗/hold未闭合，版式PASS不等于投稿证据全部完成 |

上表的 C1、K8、TSBG 倍率采用各自冻结分母，不能相乘。三轴消融经M2232独立审阅后已回填Evaluation，仅说明定向请求因果，不改摘要倍率。M2232 review SHA为 `05309255148c9d55b2ee84dc10abb925ccf2f72a4d2906a27cc2e2a8926bd732`。

功耗仍有输入边界：M2217/M2235使用真实ep34 activity/sign，但权重来自TB的确定性INT8验证函数，不是checkpoint FC权重。新功耗只能标成这些固定输入下的matched比较，不能自动称为真实网络功耗。若要增强TCAS-II实用性证据，完成当前六点后优先补真实FC权重的toggle敏感性；本地已有 `system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth`，这项权重绑定不要求重训。新权重对应新活动身份，不能覆写本轮SAIF或沿用其功耗数字。

## 是否上板

[TCAS-II 官方指南](https://ieee-cas.org/publication/TCAS-II/tcas-ii-manuscript-submission-guide)要求有意义的电路/系统创新及相对 prior 的优势，并要求完整的首次投稿；没有列出 FPGA 或流片为普通稿件的必选条件。指南规定五页及最后一栏仅放参考文献。

作为直接先例，[Yassin 等 TCAS-II 2024 电路稿](https://research.tudelft.nl/en/publications/a-power-efficient-oscillatory-synchronization-feature-extractor-f/)采用综合与后布局仿真，报告功耗/通道和面积。该例说明“无 FPGA 上板”不等于不匹配；它也说明物理与能量证据应扎实。

本项目已经投入 28 nm ASIC 约束、宏、DC/PT/Formality。FPGA 映射会改变 SRAM 端口、时钟、寄存器和加法器代价，不能补证 ASIC 的 3 ns hold 或 ASIC energy。只有已有可用开发板、接口和稳定厂商流程，并且明确需要端到端实物演示时，才值得把它列为辅助验证。本轮未确认现成 FPGA 平台，`vivado` 不在当前 PATH。

## 更有价值的新候选：按消费者完成来释放权重分片

现有 C2 先把一个 group 的 12 个权重 beat 填入四行 LRU，随后各 context 从完整行中依次取数。group-major 已经把该 group 的所有消费者排到一起，因此可以考虑：

1. 接收一个 8-bank × 16-lane × INT8 权重 beat；
2. 从已加载的 B4 activity/sign descriptor 得到四个消费者的 pending bitmap；
3. 每次仅在某个 context 的 Acc24 update 被接受后清它的 pending 位；
4. pending 全清后允许释放/覆盖该 beat；所有 context 的 sign、destination、Acc24 和 completion 仍私有。

这个想法的电路意义是将 **已知的最后一次消费转换成短的物理存储寿命**，从而减少 payload 寄存器、LRU/大mux的代价。多播、缓冲旁路和 weight reuse 本身有强 prior；[Eyeriss v2](https://arxiv.org/abs/1807.07928)已系统研究数据复用和带宽的关系，不能将本候选写为首次发明 broadcast。

更直接的电路 prior 是 Carmona 等 [Elastic Circuits，TCAD 2009，Fig. 21](https://www.cs.upc.edu/~jordicf/gavina/BIB/files/ElasticCircuits_tcad2009.pdf)：eager fork 用每个接收者的状态记住已经完成的传送。因此 pending bitmap 本身也不是新协议。候选可成立的对象差应限定为：在既有带 epoch/generation 的 SRAM response slot 上，实现 B4 signed-Acc24 消费者的最后消费释放，省去 adapter→row-cache 的重复数据搬运，并测出相对于 ordinary 同样零拷贝实现的电路收益。

[Josipović 等 FPGA 2018 的动态调度HLS工作](https://www.epfl.ch/labs/lap/wp-content/uploads/2018/11/JosipovicFeb18_DynamicallyScheduledHighLevelSynthesis_FPGA18.pdf)同时提示代价：多播的慢消费者可能拖住整个返回槽，增加FIFO才能解耦。不能一边删掉缓存一边假设原有预取并行度不变。M803已经有response hold-slot锁存，候选应复用这条所有权路径，而不是再加一个宽payload副本；下一步必须对槽占用、bank响应乱序及慢Acc24收费。

CPU 初筛已运行在全部 2,880 个现有固定区域 workload，共 4,320 个 48-group chunk。结果与源码分别位于 `results/tcasii_consumer_lifetime_screen_20260905/result.json` 和 `system_simulator/scripts/explore_tcasii_consumer_lifetime_20260905.py`：

- TSBG LRU4 与 LRU1 缺失均为 78,333；
- 1,662,312 个 accumulator update 在分片重排前后保持同数量，并保持每个 `(context,slice)` 的更新次序；
- 128 个有正负号的定向 Python 算例，49,152 个 Acc24 比较，0 mismatch；这是软件结构检查；
- 普通 true-LRU4/1 分别有 194,250/195,345 次缺失，差仅 0.564%。因此 **ordinary 也必须享有一行实现**，不能用过大的四行普通缓存夸大候选优势；
- 冻结 ordinary age4 的 194,240 与 true-LRU4 不完全一致；脚本明确复现 RTL 的 age 更新与 tie 语义，不能混用两个分母。

四行完整 payload 为 6,144 B，一行为 1,536 B，一个 beat 为 128 B；这些是 payload 容量算术，不是面积或能量节省。M803 slots、Acc24、标签、mask、控制和时序寄存器都必须另外计入。

独立评审给出了更简洁的实现：**直接借用 M803 的现有 response slot 作为权重持有者**，pending consumer 位全清后再做 `core_rsp_accept`。这样不必先从 adapter 复制到行缓存，再读给各 context。M803 现有 payload slots 本身为 1,024 B；候选不能说总存储只有 128 B。需要证明 held response 稳定、没有占槽死锁、epoch/abort不泄漏，以及 ordinary 也享有零拷贝/单beat控制。这是本轮更值得测试的具体电路点，结论见 `reviews/tcasii_consumer_lifetime_independent_review_20260905.md`。

一个决定性限制：M2018 在 `ST_DONE` 后保留 cache。跨 B4 的 group 序列 `[0,1,2,3,0,1,2,3]` 上，LRU4 只需四次填充，LRU1 需八次。所以上面的冷启动性质不能外推到热缓存。候选应保留适用边界或为多出的重读取数收费。

建议只升级到一天的端口/周期模型：同一 M803 延迟、同一 Acc24 端口，比较 ordinary-LRU1、ordinary-LRU4、TSBG-LRU1、TSBG-one/two-beat。若额外周期不超过 5%，且总组件面积/能量有实质下降空间，再写可综合候选。当前没有候选 RTL、PPA 或周期结果。

### 若过门，怎样让两贡献更凝练

可把主线组织成“有界物理生命周期的数据复用”，而非继续列更多稀疏名称。C1在生产端判断中间乘积是否还有未来消费者，省掉无用parent写入；C2在消费端用最后一次实际Acc24接受释放返回槽，省掉二次缓存复制和不必要留存。两者共享的是电路设计问题——复用收益是否值得付出存储、端口和保持时间——而不是声称发现了新的多播或缓存理论。

候选的必要不变量是：`pending_next = pending & ~accepted_contexts`，且 `release` 仅在本beat所有必要消费者接受后发生。`valid`、预测ready或收到SRAM响应都不能充当消费完成。signed权重只能共享原始weight，不能把一个context已乘过符号的product直接给另一个context。输出提交和epoch/generation错误隔离继续沿用C2。若总面积/能量无明显改善，这个组织只保留为解释，不把候选加入摘要，也不替换现有通过验证的实现。

## 其他候选排序

| 候选 | 本轮定位 | 准入规则 |
|---|---|---|
| pre-read/post-read 因果消融 | 必做评价，解释 C2 的机制 | 相同输入、资源和观测口；请求因果 + matched energy |
| 消费完成后的 beat 释放 | 第一新增候选；可替换 C2 内部缓存实现 | 热/冷缓存收费；同 Acc24/M803 端口；不超过5%周期回退，映射后显著省面积/能量 |
| B4-union selective bank fill | 第二候选；叠加在 read 前的 bank 选择 | ordinary同样支持mask；既有directed RTL功能不足以代表PPA；总能量额外≥15%、area增量≤2%、setup/hold达标才进主文 |
| 低复用自适应 ordinary/TSBG | 暂不投入新RTL | 现有固定区域oracle周期增量极小；若功耗曲线显示明确交叉点再考虑 |
| 空tile、近似attention、额外训练正则 | 不纳入此次两贡献brief | 没有足以覆盖新身份、AEE和硬件验证成本的证据 |

上述经验门是项目的投入筛选标准，不是TCAS-II官方录用条件。微小收益若对应几乎零代价仍可作为支撑消融；不得把门未过等同于机制一定无效。

## 收口顺序

三轴 VCS 与 LM命令发现的解析修复均已通过独立结果审阅，前者已进正文。M2235六套measurement SAIF均已生成且解析通过，当前ordinary DC正在mapping；两轴DC及六次PTPX尚未完成。下一步基于LM诊断闭合物理库和hold；新beat候选只占CPU模型工作，不阻塞这些实证。随后精简防御性文字，用时序图、bank activation与logic/SRAM energy表组织五页正文，最终核对作者信息和引用。2026-09-05版严格PDF检查已PASS且末页目视通过，不能等同于硬件/提交信息全部完成。

内部自评维持约 3.8/5 的 Weak Accept 路径；新假设和解析器修复不自动提升分数。只有 matched physical/energy 与公平对标补齐，才有理由重新评估到更强的接收档位。
