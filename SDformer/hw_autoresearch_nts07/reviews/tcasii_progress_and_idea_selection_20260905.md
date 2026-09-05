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
| pre/post-read 消融 | M2215 原始 VCS 24 commits/axis、4608 products/axis、0 mismatch；2304/2304/576 reads | 原生产parser失败；M2216独立确认可新身份parse-only恢复，旧失败包保持不变 |
| ICC2 库准备 | M2223 runtime证明 Milkyway option可query/set/readback；local_output_dir在本版本未注册 | checker混淆echo源码与runtime且存在隔离目录rename；M2224允许仅解析修复；没有转换/P&R结果 |
| 功耗源修复 | M2226发现最后一级M2160未锁SHA，尚未启动生产 | M2233修复完整传递依赖闭包，另人独立评审后再启动新身份 |
| 稿件 | 五页 Letter；195词abstract，6 keywords；最后右栏为references | 第五页正文未达到项目检查器目标，authors/funding等仍待填；应以功耗和机制图填充有效内容 |

上表的 C1、K8、TSBG 倍率采用各自冻结分母，不能相乘。三轴消融的原始日志当前只用于诊断；修复解析器和独立结果审查完成之前不回填论文。

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

先恢复并独立审核现成的三轴 VCS 结果；完成传递依赖修复后跑冻结三档 matched power；基于 LM 诊断闭合物理库和 hold。新 beat 候选只占 CPU 模型工作，不阻塞这些实证。随后精简防御性文字，用时序图、bank activation 与 logic/SRAM energy 表组织五页正文，最终核对作者信息、格式和引用。

内部自评维持约 3.8/5 的 Weak Accept 路径；新假设和解析器修复不自动提升分数。只有 matched physical/energy 与公平对标补齐，才有理由重新评估到更强的接收档位。
