# 部分时间源执行接入完整 FC1/PSN

B：源叶的前沿并行与四槽驻留已减少服务，但普通 code 已获得主要收益；源叶周期不能替代完整消费者周期。原 source 的 code/class 根配置分处 bank，也污染了小幅 BP 增量。

A：继承已验证 joined_chain 的真实 X→源 PSN→D 展开→FC1→完整 T10 PSN 链。backend 为普通内容去重 mode4，所有臂都用相同 106 个独立运算单元、8×128bit bank 接口、256KiB 参数池、3840B gate 桥。新 source 提供 128B X holding、128B graph cache，各对照同享；source 并未缩为总共 96 个 MAC。

X 候选接口：十个时间判决只生产当前请求的 (t,c)，每个实际判决仍完成全部十个非因果源乘积。最多四个不同 c 在原十个乘法 lane 上合批，持久槽减少跨批重读。这是执行接口试验；多路径决策图和驻留本身是借入机制，不预称新颖。

固定五臂：static64+next-X PF；one code；resident frontier code；one class；resident frontier class。class 只对相应修改权重函数逐位正确。原 W′ 与 source-cost W″ 分开。新增映射让 code/class 根都读取同一物理配置地址 6174，模式镜像在测试前装载，均计一次请求，消除仅根所在 bank 不同的日历偏差。

先用 2 个训练帧 P32+零/有符号边界诊断核对，再扩展已有 32 个训练帧固定 P0..31。后端使用同一固定阈值整数函数，不冒称动态 BN 或 AEE。必须核实际 partial producer、code、桥读写、全 H384 的 Y/U/gate，端口和反压；不得用叶周期相加代替真实运行。负结果只停当前布局。
