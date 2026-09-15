# 自由 U3M 接共同双 context：执行完成，当前布局为负

已把固定自由 U3M 三项函数接入真实共享 producer 与唯一完整 I24 消费者。两个 leaf 只保留状态、source/Z/psum/cache/holding，数据乘法器、32bit ALU 与权重存储均已移到 top。没有复制两个完整单核并行。共同资源下，当前空间布局比 R8 bitmap7 慢 **90.92% / 93.62% / 80.95%**（两64 / 18序列36tile）。它的已完成825帧 AEE 为 **1.2583431022**，优于固定部署 R8 的1.3276350226，但此质量取舍没有带来硬件服务改善。

| 完整冷服务，含 go | 原帧128–191 /64 | 原帧4000–4063 /64 | 18序列×2tile /36 |
|---|---:|---:|---:|
| R8 原生四P | 787,604 | 831,997 | 376,567 |
| R8 borrowRR | 786,422 | 830,887 | 376,447 |
| R8 count21 | 772,612 | 810,789 | 369,392 |
| R8 bitmap7 | 750,650 | 792,116 | 357,261 |
| 空间 U3M 不借宽链 | 1,433,737 | 1,534,402 | 647,152 |
| 空间 U3M borrowRR | 1,433,162 | 1,533,686 | 646,458 |

| 空间 borrowRR 完整服务 | 128–191 | 4000–4063 | 跨序列36 |
|---|---:|---:|---:|
| 暖 ready | 1,431,986 | 1,532,510 | 645,282 |
| 冷 BP | 1,515,680 | 1,619,018 | 688,668 |
| 暖 BP | 1,514,415 | 1,618,248 | 687,363 |

全部比较使用相同顺序的原 C96×4×4 low10 源门字和原 FP32 identity。两方独立重算各自输出，源251,904字、FP32 identity629,760字、origin328标量直接逐值相同。两套64仍来自同一帧；36tile覆盖18序列固定首帧edge128/interior9664。small集合两方不同（空间15、R8 12），不把small总服务作跨函数排名。新表每tile都加载1536源字并另握手origin一拍，含padding的外部BP已改变；没有拿旧隐式padding的BP表充当本轮分母。

周期从go、静态配置、源和origin加载一直计至最后I24输出及done。每batch最多两context，加载完同时启动；唯一消费者按tile顺序消费480行。只有I24退休才释放owned，原始raw完成不会释放；当前沿旧RR保守整batch回收，再加载下一batch。冷第一次实付Q1 576+physical U3 576+consumer24=1176配置；暖保留参数。两context原子申请 source/W/Z/psum/ALU/wide，同资源冲突轮转，独立资源可同拍授权。所有source/W/identity/output/原子仲裁/宽链等待都实际收费。

| 主资源 | 实现及公平边界 |
|---|---|
| Producer 数据算术 | top 唯一8×32 carry chain、8×signed19×13乘法器；leaf没有数据ALU/乘法器 |
| 消费者 | 唯一原FP32→J20→wide→RNE/sat24 I24；8×32×32乘法器与8×64宽链 |
| 借用 | Q1 ZADD可申请同64链，低30bit切两个15bit场；消费者/producer沿旧borrowRR原子授权，consumer等待计费。R8使用其13bit切场；切点和owner mux是实际逻辑 |
| 每context source / p | 1920 B / 15360 B，两context都完整物化P |
| 共同Z预算/服务 | 8bank×40row×52×2=4160 B、单416bit授权；candidate每bank只用低32bit（有效256bit），每写高20bit补零；R8同40rows容量但未优化利用新增深度 |
| Cache / 三M / holding | candidate cache312 B payload，共同许可384 B/context；三M总96 B、tail32 B、Z hold32 B、Q1 hold8 B、source窗口20 B、mask10 B、output32 B和support等另记 |
| 静态容量许可 | 主Q1/Q2共12096 B、R8 plane2592/class2592/rep96/排列5、consumer768，总18149 B；各臂只装自身参数并付自身配置，非全参数同时驻留 |
| 静态live和辅助副本 | candidate两576bit在top一份共144 B；R8 k/v-live每context120 B，两份240 B，plane_live162bit共享，count_live80 B/context及bitmap80 B/context等保持原账 |

R8的52B Z holding、count_hold32B、bitmap_acc32B及cache384B容量没有被本表删除。完整数量见 [resource_contract.json](resource_contract.json)、[R8资源审阅](../r8_reference/RESOURCE_REVIEW.md)。这里是共同容量许可、算术数量和服务权限，未声称各自裁剪实现等面积、等Fmax或R8已充分优化新容量；没有EDA/能耗证据。

| 冷ready实际工作：空间borrowRR / R8 bitmap7 | 128–191 | 4000–4063 | 跨序列36 |
|---|---:|---:|---:|
| Q2 MAC | 634,068 / 198,720 | 660,936 / 210,384 | 217,296 / 69,276 |
| Z授权 | 859,944 / 230,884 | 951,532 / 247,754 | 307,916 / 81,482 |
| psum授权 | 122,880 / 61,440 | 122,880 / 61,440 | 69,120 / 34,560 |
| producer32授权 | 733,908 / 287,951 | 760,776 / 321,231 | 273,456 / 102,835 |
| context仲裁等待合计 | 659,086 / 169,620 | 717,910 / 189,575 | 149,798 / 37,955 |

空间的两条R8 stripe保留真实halo生成、D3原地变换、三M归约、恢复和第二stripe psum加回。Q2 MAC约为R8的3.1–3.2倍，psum授权正好两倍，Z有效字段也未利用完整416bit容量。借链只让两64/36的冷ready各减少575 /716 /694拍（约0.04% /0.05% /0.11%），不能抹去这些工作。跨序列借用39,550次还使consumer宽链等待175拍；各等待列可重叠，不能直接相加代替窗口。真实窗口和逐FSM/grant闭合均保留在结果中。

数值合同仍为固定physical_coeff3=[母U0,U1,U3]、母U2=0，D=[z0−z2,z1+z2,z1−z3]，raw未除二p2，输出scale减半后已冻结a_q40；本轮无重新量化或中间RNE。独立CPU由真实源计算Z/D/三M，同时按相位展开W计算P，再从原FP32 identity重算J/wide/I24；与原自由U3M fixture逐值一致。D abs≤17124，M任意前缀≤239174076，恢复界≤381857954；RTL仍有Z15、D16和p32前缀断言。36seq使用不变上游source/id重新计算本函数输出，未冒充已有新36帧网络capture。825质量来自[原完整评估](../../representation_transfer_20260914/quality/unconstrained/deployed_valid/spatial_integer_summary.json)，没有代跑GPU或用RTL周期代替网络质量。

主测试32个job、1432次tile执行；另held→disjoint→sequences→small不复位换源32个job、1432次tile执行。合计64个job、2864次tile，raw/J/wide/I24各10,997,760值、Z/D各3,665,920物理字段全部通过。完整输出顺序、末包、输出被阻期间稳定，追加回放还逐拍检查parameter/source/origin/identity请求保持；每个context只在grant下访问SRAM或提交数据，owned断言拒绝I24退休前重载。独立检查8180项，含source重建的工作量、每个FSM状态、观察到的六类grant、拒绝授权等待、消费者与顶层周期闭合。新BP请求保持检查没有改RTL/周期，主记录保留，换源记录单列。

这次完成的是经典phase3/Winograd剪枝与旧RR的合法共同执行接口；尚没有相对成熟R8强控制的硬件净收益，不把接通底座写成新机制。较好AEE与更长服务/较大系数状态可保留为明确取舍，负结果只约束当前放置。也没有给重新INT8量化的直接模型排序，不能借expanded W32位宽声称跨模型最优。

复现：`implement.py`、`generate_top.py`、`prepare.py`，`run.py --stage small`后依次held/disjoint/sequences（`--skip-build`），`verify.py`；`run.py --stage swap`与`verify.py --stages swap`；最后`compare.py`、`finalize.py`。Python `/opt/anaconda3/bin/python3.12`，Verilator4.028 `-Wall`及独立make。结果每记录一行：[SUMMARY.json](SUMMARY.json)、[换源检查](verification_swap.json)、[跨函数同输入比较](comparison.json)，原始`results_*_b*_s*.jsonl`完整保留。旧树、生产、模型训练、主稿和Git均未改。
