# Post-Y T10 子集和：先验迁移与证书适配

A：成熟 distributed arithmetic 两组5bit子集和，替换后级 A×Y 的逐时间 MAC。B：原 native96MAC 的完整 PSN（同真实32病例 CSV，仅异资源参照）。X：在同一子集核上加精确 signed prefix 上下界，80 个门全部有证书才提前退休。未声称完整 BitL、未借 T5 的17%收益。

共同 full/cert 合同：每组8h×10t=80输出，唯一80路48bit加/减器，表构建/正负系数和/符号前缀/子集相加/前缀/上下界分时用它；比较器80路也随上下界分阶段用。full 跳过不必要的证书上下界阶段，是同资源强控制，不给 full 强制证书税。证书到末位亦直接比较，负gain为 U<=tau，常量门优先，无中间RNE。

A200B、两半×32地址×10×16bit表1280B，一份寄存器表，每个半表/t行8个不同地址读mux；20个16bit bank各8个32:1 mux，非单口SRAM，不声称8份表免费复制。表构建共64拍（2行零赋值＋62行使用10路共同ALU）。P/N计算20拍使用同10路ALU，后缀正负界按当前m用2拍共同ALU生成，额外120B界暂存。

Y在本RTL持久存为320×96×24bit=92160B。外部生产交接为320次2304bit写；计算时唯一共同row读，32P各读T10次（总320读），进入2880B T10×96缓冲，较原单row288B多2592B。不能一边写Y一边消费：完整320写后才读；I/O拒绝保持请求与输出。每H8组在读该P的Y时更新12个BFP指数，按实际signed24绝对值生成，不从TB传指数。

参数单个128bit请求/响应服务，一次一笔：A13word、tau480word（2×48bit/word）、gain1、constant1、constant-gate8，总503word；warm可保留A和表，仍加载阈值/flags490word。参数结束再接受Y320row，Y完整就绪后计算，最后384次80bit门输出（每次映射[p,t,hgroup]）；真实背压，最后门握手后完成。计冷/暖配置、表构建、Y生产交接、Y读、算术、证书和输出各阶段。

额外holding除T10Y外：80×48bit prefix480B、80×48bit dot480B、80门和锁定位20B、12×5bit指数、P/N和tail界240B，组索引/比较flag等控制显列。tau5760B/flags共144B沿真实整数函数。无乘法器；与native96MAC的乘法、查表mux、holding差异禁止当同面积或同频。

先原37病例（32real+5签名/逃逸/负gain/constant/tie诊断）逐Y/U/gate独立gold，再full/cert ready/BP和cold/warm无reset连续换源。full逐U验证；cert仅最终门必须精确，另以每次证书界包络参考U的monitor检查。若较native慢，停本布局并分摊固定位平面、证书和供数税，不认定DA/BitL家族失效。
