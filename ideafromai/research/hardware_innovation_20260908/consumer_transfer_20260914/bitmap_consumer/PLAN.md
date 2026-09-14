2026-09-14。已确认最小单context接入可行：复用d1_forward的consumer_stream原生参数/source/identity loader与实际FP32 identity→J20→I24消费者，leaf替换为bitmap_pipeline/decomp_core.sv原样副本。只扩展wrapper mode与诊断计数，bitmap算法和资源不改。

四臂为14原生dualP13、15旧完整位平面、13构造/供数流水/live块适配、10再加已读取单比特bitmap的原生signed3路径。逐tile全raw原序交给原消费者，所有J20转换、64bit乘加、RNE26和I24饱和实际执行。只有一个context，不借consumer宽链，不增加RR范围。

共同producer仍8×32bit ALU、8×19x13乘法、208bit z口、520B z、15360B完整psum、1920B source。位平面额外4320B源bitmap、2592B权重副本、八棵pop16以及live/流水控制沿原叶保留；不是与416bit四P/borrow等面积。参数装载1848拍包含consumer系数，bitmap位平面由实际Q1配置写入形成；source格式生成、bitmap访问、plane读和pop/ALU均继续计入RTL。原先raw收益不乘入完整consumer结果。

先使用count_rr现有23个小fixture的原生binary参数/source/raw及FP identity/gold引用，包含−4、计数边界、padding/输出保持、原T签名，并做无reset跨mode；再真实159/19197跨row和128–191、4000–4063两套64，冷/暖及source/weight/identity/output BP。每个raw/J20/I24值对原gold。两套64来自同一帧，后者与旧校准输入不重叠，但本叶不重新校准或重排T。

只写bitmap_consumer，父bitmap_pipeline和其他目录只读。Verilator4.028 --cc --exe＋make，无EDA、训练、生产变更或提交。完成本有界消费者验证后收口。
