# 连续原生 tile 流：共同 mode5/6 调度与真实加载边界

旧 `r0_source_retirement_20260913` 只读。本目录原样复制最终 `pair_parent_merge.sv`，新增 `stream_wrapper.sv`；执行差分继续只为模式5/6，不增加新代数或切换剪枝规则。

一次 `go(first_tile,count,mode)` 启动完整连续作业。SV 首次经共同128bit配置握手读取10,368个W行和288个mask项，静态驻留。随后由SV维护tile_id和1536词索引，从全帧地址`c*76800+y*320+x`请求单个T10低10bit源词；源原点`(2*ty−1,2*tx−1)`和图界均在SV生成。图外padding不发外部读，仍支付一拍源配置写零；origin另付一拍。没有TB预gather、传origin、支持码/目的集合或中间和。

固定一个源buffer，不能与该tile计算重叠加载。每tile加载后由SV发core.start，输出携带tile_id、原480行地址、tile_last/job_last。所有输出握手完成且core.done后才退休当前tile、累加核心计数并推进下个tile。最终done要求完成所有tile，输出被阻塞时data/地址/身份/last均保持。连续作业使用同一RTL实例和同一份W/mask；重复warm作业不把配置再次收费。

模式5/6共用所有存储、加法链、配置口和背压。外部源是每拍至多一个10bit响应的弹性接口；W配置为每拍至多一个128bit响应；既有内部source/W许可与结果ready也接受阻塞。TB只按实际请求地址应答，并逐值检查整帧gold。输入halo在相邻tile重复请求，全部收费；没有未计价的行buffer/预取/源退休上游优化。

首验证固定id128..191跨第0/1输出tile行，共64连续tile，dense/block_magnitude25/cin_fullcost25、两模式、无阻塞/command-relative整体阻塞、两次同实例作业。测得仿真速度后结合完整帧工作量估计19200tile时长；若合理小于30min，执行三臂两模式完整帧。C++优化仅为仿真速度，不改变SV；Verilator4.028 `--cc --exe -CFLAGS -O3` 后独立make，无EDA。

累计64bit统计包括静态W/mask加载、外部源请求/内部源配置/padding/origin、各load stall、核心source/W/psum/sum/merge/各stall、输出握手及tile退休。总周期独立恒等式为`static_W+static_mask+parameter_stall + source_load+origin+source_load_stall + Σcore + 2*tiles +1`；两拍/每tile为core launch与完成捕获，末1拍为job finish。数据分支整层输入与gold不代表全网bittrue/AEE，此目录只证明r0整数线性层的真实连续执行。
