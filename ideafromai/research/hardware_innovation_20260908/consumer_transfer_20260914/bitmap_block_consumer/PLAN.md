2026-09-14，接入前计划。原样复用bitmap_block_resident当前decomp_core，沿用bitmap_consumer已经验证的单context consumer_stream与实际FP32 identity→J20→I24消费者，只改wrapper合法mode为14/9/8并透传cache_reads/cache_writes。

mode14为native dualP13。mode9按K16条带驻留80B bitmap和5B live，每个活动P/T条带实际读取208bit z、完成原生单比特或三plane工作、写回全字。mode8再将非空K16的三plane通过原W口预取到Q1阶段闲置的Q2 qblock前三行48B，复用已有128bit单读路径；单比特仍走Q1原读，Q2 VLOAD覆盖全部qblock后再使用。新增读写计数均进入wrapper总账，不新增数据数组或并行生命周期。

保持一个context、8×32 ALU、8×19x13乘法、208bit z口、520B z、15360B完整psum和1920B source；位平面权重副本2592B及八棵pop16仍是真实增量。mode8的cache mux→pop→ALU组合路径没有物理时序证据，周期收益不能宣称同频率PPA。consumer沿用原8×64主链、8×32×32乘法、FP转换和RNE/饱和边界，不借宽链，不扩RR。

先引用现有23个小fixture进行各mode冷暖/BP及跨mode，再159/19197跨row、两套64（128–191和4000–4063）同实例两遍，逐raw/J20/I24全值对原gold并独立复算cache/plane/z义务。两套仍同帧；不重排T或校准。父目录和其他目录只读，只写bitmap_block_consumer。只Verilator4.028 --cc --exe＋make，无其他机制、EDA、训练、生产或Git提交。
