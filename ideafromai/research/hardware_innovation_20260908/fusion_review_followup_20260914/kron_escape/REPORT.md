实际RTL验证通过，但当前Kronecker拟合端点停止：真实8tile同函数控制冷134140→131147，仅省2993拍（2.231%）；新A800十帧AEE **1.898806137135578**，未通过同组原NB0 **1.469968114433749**。不扩825、不为这个单项/两组escape扫参或开生产EDA；不据此杀掉所有Kronecker/结构分解。

|真实8tile|展开Wh控制|Kronecker+两组escape|
|---|---:|---:|
|冷总周期/无背压|134140|131147|
|暖总周期/无背压|119356|116243|
|冷总周期/有背压|144652|141487|
|暖总周期/有背压|128624|125331|
|producer MAC发射|17604|12491|
|S构建拍|0|4027|
|S向量写/读|0/0|1280/8090|
|Q2/A实际权重读取|756|206|

实际MAC减少为17604−12491=5113；完整core只省3113，冷配置新增120拍使总净省2993。MAC降29.04%没有变成同等周期改善。详见[SUMMARY](SUMMARY.json)。

16fixture×2模式×2背压×2冷暖命令=128条，raw/J/I24各491520值通过；零/一、padding、转换饱和、全factor/全exact组、极端signed乘数和pair cancellation均有实际测试。[运行](run.py)、[配置与gold](prepare.py)、[全部结果](results.json)。消费者覆盖与Q1完整供数没有省略，gold只用于比较。

[拟合](fit.py)只看thun训练336窗，固定8步整数ALS、两组escape为6和7，未用验证重选。当前布局保留中间S为19bit精确整数，没有凭实数式跨原I24舍入。所有常量及界见[fit.json](fit.json)；当前形成的是10个Kronecker组+2个原始组的混合函数。质量测试执行整个现有学生及真实消费者，10帧/516735有效像素，所有本段整数检查及零饱和通过，但任务质量明显失败。[质量结果](quality_diverse.json)、[评价脚本](evaluate_quality.py)。elapsed_seconds只是评价耗时。

新增审阅确实找到一个本轮开发bug：cold mode0后切mode1，原wrapper resident会使A/B/escape漏载。已加入factor_resident，额外15beat只在需要时加载；最终完整128条重新构建复跑，模式切换由独立reviewer补测：额外72命令通过，其中32个跨模式命令，另覆盖19bit S极值。保留该发现，不能把同mode冷暖PASS扩写为任意模式切换已验证。

新颖性也不足以把性能小正包装成标题：Hybrid KPRNN已给未约束行+Kronecker行以及两因子快乘；本项剩余差分是八lane输出组、中间S驻留、精确组与共享算术的具体执行，不是新分解或新hybrid算法。训练后同质量更小普通R8/更多Kronecker项尚未比较，当前结果不能称“比被借鉴工作做得更好”。[本轮近邻记录](../literature_followup.md)，[独立审阅](../review_kron.md)。
