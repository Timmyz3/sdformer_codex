# 连续层接口：根代理只读审阅

2026-09-14。阅读stream_wrapper.sv及tb.cpp完整文件、REPORT和汇总；没有修改实现、没有另跑六个整层作业。数据代理的独立计数见[REVIEW_STREAM](../data/REVIEW_STREAM.md)。本范围未发现阻断问题。

实际go之后由SV持有tile_id并循环。origin由tile_id行主序生成：每行160个2×2输出tile，共120行；source地址从通道、4×4邻域及该origin得出，23位覆盖7,372,800词。图外不请求外部源，仍付内部源零写一拍。静态W/mask首次装载一次，resident仅由reset清除；合法接口支持同参数后续作业，不支持不reset换mask/权重，应保留这个范围。

每tile第480个beat握手后，accepted_beats变为480；leaf的最终done在之后到达，wrapper才累计核心计数并推进tile，因此无提前复用源buffer。末tile另经FINISH再done。输出的tile_id、row、last与数据在等待ready时保持；TB按实际握手检查并对完整gold逐值比较。外部源和参数是单词弹性响应，不是多笔在途DDR请求。

全作业计数用64位，单tile核心计数32位；完整工作量的总账实测在TB对齐，不能从每tile模型外推冒充整层运行。完整帧实际只跑无阻塞；跨行64tile另跑固定作业内背压和同实例重启，两种验证边界分开表述。源halo目前重复装载约四倍、没有缓存或重叠，但双方收费相同，不隐藏这一实现代价。

输入源为3090所捕获matched-dense快照；A800新quality的浮点消费者及部分gate可能不同，不能宣称这两台环境整网bittrue。整数层结果、mask函数和相应源合同清楚。约5%为该整层串行执行器的Verilator周期收益，尚不是同Fmax、PPA或整个r0残差块/整網收益。
