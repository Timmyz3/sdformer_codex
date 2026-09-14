2026-09-14，实现前计划，已检查旧实验避免重复。

旧`pro_fusion_trials_20260913/interval/REPORT.md`已实际实现R24+onepass内部U16的P2×T10×K864端点执行：真实2404→3502拍，孤立脉冲使557事件变1088端点；付费全tile二扫选择也负。它明确指出未增加独立状态不能逐k混合direct/delta。本轮不是重新命名该恒等式或重复其U16切口，待试接口是其明确未实施的逐K混合独立两域。

当前函数依`r8_consumer_fusion_20260914/REPORT.md`：AT-LIF二值门g，静态θ已吸收，z=Q1g、raw=Q2z，Q1→Q2无中间RNE。完整K864、T10、R8、P4、N96保持，原生96×4×4源gather保留；没有PSN时间截断、跨非线性、旧prev-anchor连续变换或gold供数。

共同迁移控制mode0为native2P13直接，mode1为全部端点d_t=g_t−g_(t−1)，RTL前缀恢复z后原Q2。真正适配mode2在每个有效K从真实T10门字计算两位置合并后的直接/端点更新次数，付一拍选择，严格较少才用端点。直接与端点分别累加两份z域，最后有费prefix与merge；若无K选端点可省去未发生的prefix/merge，但两域clear仍计费。

所有mode都编译在同一union硬件：原8条32bit ALU、8个19×13乘法器、全局一份208bit z服务。z从原520B增至两域1040B，不能声称与旧单域等面积；直接控制同样预留第二域，但不强迫其清零永不用的数据。混合额外20次clear付费。prefix和merge各为显式208bit读、共享2P13 add、写，不能组合读两域或免费相加。前缀保持复用原Q2 acc寄存器；不增加数据加法器。Q1=-4的下降端必须用13bit XOR+carry-in在原ALU形成+4，不能三位取负回绕。

先沿用pair_sparse的20个真实/边界fixture，并增加固定长run、交替、两者跨K混合、-4端点符号控制。测试所有3840 raw、source/weight/output BP、连续无reset跨mode。独立模型核对所有义务与状态周期；先按真实八块是否有完整净收益决定是否进入64流，不以CPU工作量代替RTL。若无真实净收益，报告选中了多少长run、选择/双域clear/prefix/merge的实际成本及尚未测试的接口，不扫参或把负结果扩大为家族结论。

最近邻边界沿旧报告已核实的Sigma-Delta固定线性层差分/积分、DeltaCNN、LoAS全T权重组织；Comperity仅取得摘要，不能声称已复现全文架构或避开方法范围。已知恒等式、逐列择优与carry切断均不独立构成新意。

仅本隔离目录、Verilator4.028 --cc --exe + make；不EDA、训练、生产修改或commit。

第一轮真实结果后补充的必要强控制：保留mode2，增加mode3。CHECK遇空源直接跳过；若四个T10字都没有相邻11，则每个1必是上升端，逐bit有endpoint_union包含direct_union，故可精确选direct。只有含相邻11才付SELECT。第二域在首次真正选择endpoint后才逐行付20拍清零，期间保持当前k/src/pending。它消除明显无效的选择及初始化税，不是阈值扫描或未来性能预测。新增一个先direct、K中途转endpoint的小fixture验证lazy clear不会丢当前工作。

原序mode3实测没有任何获利K之后，根代理提供了明确新接口机会：本层源是完整非因果T10门字，Q1/Q2对T线性且θW静态，因此执行T可变而对外T必须保持。只用同帧tile0–31做一次pair-state Hamming最短Hamilton path DP，固定得到[8,2,6,3,9,4,1,5,7,0]；held128–191只验证，不参与选序。新增mode4沿用mode3，RTL加载40bit排列并付1拍配置、L_GATHER位mux、原psum DRAIN_READ逆地址恢复原T，不能TB重排source/gold冒充RTL。先全部小fixture及每T输出1..10的inverse检查；正结果后才跑held64的mode0/3/4。此为同帧tile留出，不是跨序列泛化；时间重排本身也不宣称首次。

独立审阅后只补必要对照mode5：采用与mode4完全相同的固定排列、40bit配置、L_GATHER位mux及DRAIN_READ逆映射，全部K执行端点，只用一个z域，付20清零拍及20次prefix读/add写，不付SELECT、第二域clear或merge。先原26fixture、原T身份检查及跨mode，再同held64完整冷暖/BP；实测对比用于分离排列与混合两域贡献，不将70535机会数冒充未实现周期。

审阅还指出原cal0–31与held160–191源halo共享y1..2；它只有输出tile分离。保持原集合结果与审阅不变，引用audit准备的同帧4000–4063原生source/gold新增64tile，与cal输入交集为0，冻结同一排列不再校准。只跑mode0/4/5相同冷暖、BP、完整raw流程。本轮不增加其他机制，所有结果仍不构成跨帧/序列泛化或与J/I24实际融合的验证。报告同时保留根代理未改旧RTL的416bit四P强控制更快这一限制。
