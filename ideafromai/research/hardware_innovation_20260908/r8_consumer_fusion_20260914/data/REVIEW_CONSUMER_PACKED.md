# 独立附审：packed核到真实FP32 identity/I24整帧

在已审REVIEW_PACKED和REVIEW_CONSUMER基础上，仅核对新`consumer_packed`映射、现有收据及完整真实源，不新跑RTL/GPU。未发现阻断。消费者i24_consumer.sv直接按文件字节比较与已审FP32版完全相同；未用hash。wrapper差分限packed实例/对应计数、mode14/15及配置集合，source地址、identity请求、最后480beat退休与producer/consumer完成汇合未改变。

新cold配置1848词=Q1 864+Q2 96+k_live864+a/b24；没有expanded权重与其块mask。两模式共同享有该配置，不能拿与旧12504词集合的差当双P独享收益。latent宽口、520B状态、cachedOS与单周期异步MAC风险仍以REVIEW_PACKED为准。

[review_consumer_packed_counts.py](review_consumer_packed_counts.py)独立核对2,772项完整消费者/加载/退休守恒，覆盖128小fixture、8个64tile、2个19200tile命令，rawp/J/I24每检查点分别491,520、1,966,080、147,456,000值。公式仍为3385×tile+join_wait+output_stalls；总周期加入一次实际配置、source1536×tile、origin、真实背压和wrapper控制，不叠加相互重叠的producer cycles。

另 [review_full_source_pairs.py](review_full_source_pairs.py)从data/first_source_words.npy直接按真实C96/3×3tap与同t双P读取，独立计算每tile脉冲、空间对并集、双活及Q1词需求；不以执行counter替代源。全部64tile和整帧记录的72项对应核对一致，见 [review_full_source_pairs.json](review_full_source_pairs.json)：

- 全帧原标量更新23,767,485，双P同t重叠3,993,893，合并后19,773,592。
- Q1供词7,062,080两侧一致；完整mode14/15源、W、第二W、psum、MAC、FP转换、宽乘/加/最终RNE计数相同。
- 完整出口总周期336,199,028→324,217,349，差11,981,679，严格等于独立真实源双活数×3。

这些证据覆盖此固定一帧、完整19200连续tile及真实FP32 identity出口，不扩张到外部DDR、映射时序、全网RTL或其它帧硬件收益。数值函数与已完成825的新消费者相同；没有为packing重选训练/量化参数。
