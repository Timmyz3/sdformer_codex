# 两项常量源的真实消费者费用

B：固定 PoT2 的时间源已经有共同源 RTL 结果，但源叶省拍不等于完整局部链省拍。新源还会改变 sn1/sn2 活动、完整 K864 请求，以及 raw I24 双消费者的执行费用，必须从同一 Machine 的真实时间线取结果。

A 与强对照：给新 stage320 dense 和 lifting40 相同两项常量投影权利，保留各自所有其他 literal 参数、阈值、RNE/sat、冷输入和完整消费者。直接复用当前 matched_local_chain 的 source→preview→sn2→Conv2/BN2+rawI24→gate+PED U24/V96，给双方相同 96 RF、128 KiB、SR64/SW64/CR256、H4 目录及驻留 MAC。各自与未投影 stage320 父函数对照，再在相同量化许可下比较 dense PoT2 与 lifting40 PoT2。dense 的同函数低状态程序作为已验证源控制单列，不拿不同源程序重复/相加成局部性能。

X 与限制：普通 PoT2 本身不是 X。只检验结构源在实际双消费者中是否还留下净服务收益。固定两种 source 函数、corner/interior 四例 ready，不训练、不扫常量/端口，也不加新的 EDA。独立 CPU 新 gold 与新参数逐值执行；真实 GPU 小捕获另行对齐，不能继承旧参数的 AEE 或仅凭源绿宣称整链闭合。
