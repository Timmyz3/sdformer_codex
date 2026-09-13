# 独立稀疏假说

在展开本轮primary方法阅读前固定如下四项；本轮不跑训练、RTL、EDA或CPU机制筛选。

1. **H1：源字消费者闭包剪枝。** 以r0源SRAM字及3×3 halo真实消费者为对象，联合粗细两层权重支持，使逻辑删除落成物理请求取消。预期最大漏洞是已被VENOM/HighLight/HiNM和普通通道剪枝覆盖。
2. **H2：有界lane私有压紧。** 旧公共walker的union(time×slot)有空洞；用显式每lane mask walker、独立psum地址、系数字驻留和RAW控制，检查能否付出硬件代价后接近每lane最长事件链。不能继续直接用max(events)计周期。
3. **H3：原生空间包直达卷积。** producer输出含T10的自描述位图/索引包，由line buffer完成halo，不预展开im2col；目标是旧源包生成和H8十二次重放。它很可能属于强A，先不申领新颖性。
4. **H4：物理burst级近似续算门。** 由真实prefix点积/已读脉冲统计产生续算决定，门直接控制取数；不能让goldmask进入TB。r0连续输出、r1门/连续PED及非因果T10PSN分别处理。最大漏洞是CGNet/旧preview接口。

阅读后收敛见[机制报告](mechanism_report.md)：保留H2；将H1与H3收紧到原生字的可实现闭包；H4只保留强对照而不升为独立创新。
