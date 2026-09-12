# native投影→全域BN→join：已读实现后的实际缺口

当前有三个真实实现，但尚不是共同时间线：旧 `native_projection.py` 是实际resident projection gate经完整K864/H96、三个H32权重tile到4×4 raw窗口；`default_bn/onepass/onepass.cpp` 是完整192000×96捕获raw到单遍moments/256树和规范化；`default_bn/consumer_fusion/consumer_fusion.cpp` 从已算统计和捕获raw/PED开始，完成不物化中间BN的真实join。后二者各自新建Engine，因此不能加到本轮局部总数上。

最直接缺口是全域native生产：现全域 `full_proj_words` 可作为真实独立输入边界，但本轮完整生产者只执行两处窗口，没有产生全部240×320×96门。全域raw为73,728,000B，PED packed24为55,296,000B，均超128KiB。必须选择并实际执行外部缓冲或重算；BN先完成192000个位置的均值/平方和，才可规范化每个位置，不允许拿捕获mean/var接上。

数值也有边界：native现实现将实际weight作TF32舍入后按K递增做FP32 fmaf；GPU卷积有不同归约顺序。onepass825验证的输入仍是GPU生成raw。新native完整回放即使参数相同，也要独立核raw/BN/join差分，不能直接继承onepass825数值身份。

可立即继续的一个接口是：以捕获全域projection-g与真实对应PED作为外部输入，统一C++ Engine逐H32执行native，付费写raw，再在同Engine上沿原T-major、两stripe、256树执行moments，最后原位normalize＋PED ADD。原输出物化已被普通BNFF融合删去，应继续不写中间BN。若把统计与native合流，必须保存每H32/stripe原顺序和树状态；native当前P2/T10次序与BN T-major不同，不能免费换序改变统计再称原函数。

该接口将闭合“捕获门/PED→native→BN→join”后段，而不是“真实I24→整个r1”。连接到整个生产者还需全域前级调度/halo重用，之后才能讨论整段收益。本轮按根代理优先级先补Dg解码器，不在报告中制造假闭环。
