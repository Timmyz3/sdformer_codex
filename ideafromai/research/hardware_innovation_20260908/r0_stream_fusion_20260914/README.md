# 2026-09-14：整层执行、整数因子与单方向变换筛选

继续在当前昂贵 r0.conv2 上推进实际 RTL；AT-LIF `{0,θ}`、静态θ吸入W，T10及真实后继保留。生产与稿件只读，目标仍是TCAS-II。

|工作|入口|当前实装范围|
|---|---|---|
|原生源与父和归并的整层执行|[stream_rtl](stream_rtl/README.md)|一次go连续19200tile，三mask、两模式；6个完整帧作业及24个跨行/背压作业|
|R8小整数因子＋完整收缩|[integer_factor](integer_factor/README.md)|全K864/C96/N96/T10的三模式；168runs；新链diverse10|
|横向Winograd＋纵向直接|[one_axis_transform](one_axis_transform/README.md)|与同模块强direct实际比较；不同于之前二维布局|
|完整数据与质量|[data](data/README.md)|原生全帧捕获、全层整数gold、完整源统计、三臂valid825与独立核查|
|先验与独立审阅|[novelty](novelty/REPORT.md)|借入A和未实现X分开；源码及协议审阅单列|

[阶段结果与下一步](REPORT.md) · [分开函数/资源的结果表](comparison.csv)。三臂Q16完整825已完成，AEE分别1.210689/1.265597/1.299231，均优于历史NB0 1.445353。生成的大数组、权重fixture、编译对象留本地，不入Git；源脚本、RTL、逐运行结果和复现说明入库。各条函数、资源预算、输入范围不同，不能横乘倍率或拼成整网FPS。
