# 硬件执行入口

最新完成[真实共享资源交错](actual_interleave/README.md)与[完整A窗口扩展](actual_interleave/full_window/README.md)，共40例CPU数值时间线。完整A消费者＋B源相对立即消费/P1留Z强对照：dense/lifting/34少0.68%/1.90%/2.61%；当前布局作为公共底座，不作为独占创新。另见[新参数局部链](matched_local_chain/README.md)、[两项源局部链](source_constant_local_chain/README.md)、[完整native后段](native_bn_join/README.md)。具体函数与范围分别记录。

以下两项是本目录早期表示接口记录；后续完成状态以[阶段总表](../README.md)为准。

已完成两项实际执行；没有新增生产RTL/EDA/训练，也未改旧stage。结果均是CPU payload slot prototype，不是RTL加速比或整机指标。

1. **[W4/W8压缩权重](PACKED_RESULTS.md)**：完整同Machine局部链25臂，真实code+row-scale进入存储，所有整数输出零差。确实少系数字节，但当前解包/宽尺度分解增加局部服务约0.56%–0.86%；只停止这一放置的加速主张，保留低位质量和强对照。
2. **[新train-only Dg+c预测解码](prediction_decoder/README.md)**：四窗加固定压力30臂，固定5+5表覆盖全部1024模式且实际460,800预测值零差。已给表方案c缓存/零行传播，仍比完整RF缓存条件加慢21.93%–27.96%（ready）。只停止CR表放置，不杀查表家族；尚未测跨行紧凑RF表布局。

主线下一接口的真实缺口在 [NATIVE_INTERFACE_GAP.md](NATIVE_INTERFACE_GAP.md)：全域native生产、BN域完成与数值顺序尚需统一，旧局部native/独立BN/独立join表不能相加。本轮未以捕获统计制造完成假象。

上阶段唯一未经恢复的source34已实际十帧1.686686，低于质量要求；这个结果只属于那次mask，不否定训练后普通结构源。当前新普通结构恢复由算法代理负责，本硬件目录不继承其未完成成绩。
