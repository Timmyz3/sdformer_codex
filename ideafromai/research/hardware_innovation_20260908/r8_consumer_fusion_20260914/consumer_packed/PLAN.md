# 双P packed 核接完整 FP32 消费者

本目录只适配共同消费者，不改 `../packed_rtl/packed_r8.sv` 或已完成 `../consumer_rtl/`。作者已确认核心固定、112runs全绿。先读其PLAN和接口：mode14 scalar-packed与15 dual-P同一模块，8bank×20×26bit的208bit latent向量端口、单bank26bit scalar读取、八个19×13乘法器、八条32bit分段加法链、同C1原生窗口、完整Q2 R8块缓存与寄存器psum。不能把该资源点与旧mode9/11说成同面积。

固定金函数继续为Q1∈[-3,3]、Q2 signed16、`z=Q1g,p=Q2z`，无中间RNE；输出row=og*40+P*10+T，N8signed32，共480向量。直接复用本轮A800原始source和FP32 identity，以及独立rawp/J/I24全帧金值。先确认固定Q1/Q2完全相等，再运行。

后端 `i24_consumer.sv`逐字复用最终FP32→J20→wide64→I24，不改冻结函数。真实FP32转换、常量A/B分别读取且跨40行驻留、宽乘、两宽加、最终RNE/sat和输出背压全计。rawp由packed同实例产生，identity独立真实接口；TB只响应SV地址并检查三个出口。

新wrapper配置顺序4Q1(864)、5Q2(96)、6k_live(864)、7a/b(24)，静态共1848拍一次驻留；每新tile仍1536源+1原点，480FP32 identity请求和480实际转换。两模式同状态/端口、同加载、同输出完成条件。移除不存在的expanded和blockmask配置不能只给候选。

执行顺序固定：本轮8真实tile及已定义数值/边界fixture，双背压、两次无reset；然后连续64tile id128…191跨行、cold/warm与背压；全绿且仿真时间合理后，两模式各一次go完整19200tile。核心及消费者cycles重叠，总账继续`Σconsumer_cycles + static + paramstall + source_load + origin + sourcestall + 2*tiles+1`，不相加生产者与消费者周期。不新量化、不扫布局、不训练、不EDA、不改生产。
