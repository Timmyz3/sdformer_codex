本项只检验：单context四P低位打包的收益，接入强普通RR后是否仍成立。B是两context争用source/W/z/psum/ALU时，少发Q1更新可能被repair、规范化、排队及消费者等待抵消。X候选是精确片段更新在有界共享执行中的兑现，不认定overflow repair本身新颖。

强控制与候选都用两个已装载context同时启动的普通RR，唯一共同Q1/Q2权重阵列、八个signed19×13乘法表达式、八条32bit加法链；consumer保持原独立完整I24后端。二者同416bit z向量口、每context520B z、1536×10bit source、完整psum/局部Q2/metadata/holding，冷配置/源/identity/末输出都计费，不做loader-compute重叠，不接direct/halo。

仅比较mode1（静态证明tripleP10＋第4P，超界精确dualP13 fallback）和mode2（centered fourP8＋有义务high5 repair＋Q1后20拍规范化）。不加入seq/阶段错位或近似模式。两臂具有同union状态及分段硬件；单context上已看到的差异不作为本项结果。

从modular_core抽离所有producer数据加法/乘法和Q1/Q2数组；context只保留状态、存储、operand/result总线和请求。共同仲裁五类source/W/z/psum/ALU：ZCLEAR/ZREAD/REPAIR_READ/NORMALIZE_READ/ZSCAN申请z；ZADD/REPAIR_ADD/NORMALIZE_ADD/BASE_MAC原子申请z+ALU；STORE/DRAIN_READ申请psum。读取及提交都必须获grant，resource交集空时可同时推进，其余RR。额外repair和规范化必须真实竞争，不能每context藏carry链。

正负界证明放到top：每个已付Q1配置beat调用同八条32bit链作双13bit界累计，保存唯一26B界状态并广播range_ok；冷864次proof ALU计入shared_alu_grants，warm不重算。各context8B correction状态共同保留。修复调度由实际overflow OR决定，diagnostic popcount只计数字段。

先16fixture×2模式×有/无BP×冷/暖，以及159起3tile和帧末3tile覆盖跨行/奇数尾/两个context自然碰撞；逐值核raw/J/I24、hold、identity、退休、proof及共享grant。若候选在真实小集相对RR triple10有净收益且无功能/资源问题，继续128起64tile有/无BP冷暖。核对独立原生源义务和每一类grant/等待/总周期，禁止core周期离线相加。只做到64，不跑full/EDA或重新训练。

最近邻碰撞记录：ISCAS2025《Optimizing Area and Power of MAC Arrays in DNN Accelerators via Overflow-Aware Partial Sum Management》官方摘要涉及窄本地psum的溢出管理。全文未取得，不能声称其没有本项细节；此组合不以通用overflow repair命名贡献。文献差分另审，本项先完成公平RTL测量。
