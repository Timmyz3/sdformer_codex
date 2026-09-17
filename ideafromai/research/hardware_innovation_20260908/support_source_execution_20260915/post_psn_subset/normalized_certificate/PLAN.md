# 只移动证书组合路径的固定适配

从已冻结joined_fc1 fork，native/full/cert、真实FC1写者/唯一Y、397词共同loader、120B H96门转排和所有时序状态原样。B为已经实测整链cold仅4.3165%的候选，对独立native只能承受约4.32%降频；旧LUT→双加→可变左移→界加→比较路径尚未映射。

等价变形：lo=(nv+N)2^m−N，hi=(nv+P)2^m−P。delta=positive?1:0，GROUP寄存signed49 qN=tau+N−delta、qP=tau+P−delta。PLANE比较nv+N > floor(qN/2^m)、nv+P <= floor(qP/2^m)，lower赋positive、upper赋!positive，constant优先。signed49覆盖tau48端点；算术右移实现floor，m0包含所有等号语义。

为使GROUP也只用两输入加法含carry-in，P/N寄存器保存P−1/N−1：初始化−1，原20个累加拍不变。GROUP公共bound位置计算tau+stored+(1−delta)，PLANE同位置计算nv+stored+1。不是藏第三个加法，native原算术不变。80个相同normalized_bound实例同时供joined和独立Y24诊断叶，避免两个不同实现自证。

160×49bit q寄存器＝980B，删除20×48bit tail＝120B，数据净增860B。旧160个48bit界加法改49bit，去20个tail减法位置；共同算术位置从356降336，但位置宽度不同，不能推出面积。删除80个nv变长左移和20个tail初始化变移，新增160个49bit阈值右移；参数右移与nv双prefix计算并行后再比较，实际mux/线/比较路径仍需映射。无旧bound/tail观察电路，TB改检查normalized的真实49bit和阈值量。

先数学边界probe，再完整RTL沿444主＋36warm生命周期；signed24全范围/负gain/tie及tau48端点用独立真实Y存储叶诊断，不将CPU极值Y注入真实FC1路径。预期逐命令周期完全等于joined，本次只改变关键路径和资源，不称新的周期收益。保持旧目录只读，无训练、生产修改或EDA。
