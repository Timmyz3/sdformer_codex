# 普通 dense source PSN 的完整 CMVM 控制

直接读取 fixed_valid825 的 identity_permuted_base 导出；与 stage128x256 的 source_A 逐元素再量化、τ再编译均0差。该普通学生既有825 AEE=1.219801338299，不是 lifting 的函数或精度。

官方 da4ml0.6.0，固定 wmc/auto、不限延迟，一次完整10×10矩阵分解+CSE：**260加减节点**，66个直接共享节点，最长加法依赖7，逐输出深度[7, 6, 7, 6, 6, 7, 6, 7, 6, 7]。逐系数CSD仅作为弱算术参照：479加减。节点精确box宽度最高41位；还分别记录移位输入所需的保守算术端口宽度，不把输出位宽当免费窄加法器。

输入为真实signed24非对称合法域，As exponent=15，dot48→RNE15→saturate24。10维节点系数和最终A全域符号核验通过；全部1024域角点、4096固定随机向量和21基向量/零向量，共51410个输出，与独立整数dot及官方DAIS均0差。RNE状态和最终门0差；另有815个正负tie／截断／阈值边界测试0差。这不是实际图像再捕获。

普通强控制也获得静态门编译：source Q24只有门消费者，RNE/clamp与阈值可精确合成dot48 cutoff，包含奇偶tie；无需为这个门出口物化Q24，仍须保存raw I。JSON保留每行完整合法dot/RNE/饱和域及cutoff，不省略τ与θ的区别。

官方顺序与简单压力顺序的存取、临时状态另列，输出sink假定可接收；它们尚不是端口／流水／扇出闭合。官方cost、节点数、逻辑深度均不能称周期、面积或功耗；不与lifting40半步回写混成同函数比较。此项补齐A，尚未证明新的X。

来源：[官方算法](https://calad0i.github.io/da4ml/cmvm.html)、[官方实现](https://github.com/calad0i/da4ml)。产物是同前缀 .py/.json/.integer_dag.json/.official_pipeline.json/.dais，旧helper、训练与官方源码未修改。
