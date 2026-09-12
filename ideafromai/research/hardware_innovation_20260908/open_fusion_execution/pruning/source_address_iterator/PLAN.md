# 完整K864的生产者掩码地址接口：先写B与强对照

B：已有phase/H8生产者已经物化被删T10门字为零，完整消费者directory仍按c×9扫描后丢零。原Machine只有最后一个SR64响应字缓存；c-major访问穿插9邻域/P2，不能假装已享有相邻四通道的word复用。本轮只验证把已知掩码提前变为物理源请求能省多少，连续raw I24/PED义务不删。

A：Gustav式稀疏目录、普通SR64 word coalescing、静态块跳过和地址迭代器。它们均为借入/公共编译底座。新X尚未成立；候选接口差分是使生产者global-source相位/H8许可在消费者完整K864目录之前生效。是否值得保留须看对最强共同供数实现的净服务；不能把普通word复用或mask跳过冒称新机制。

强对照：①原c-major扫描（诊断）；②共同收费的SR64四通道收集、扫描已masked载荷后丢零；③同收集器、每k静态许可检查并抑制读（普通predicate）；④同许可/收集器，以保留offset位图发射k，避免遍历已知空项。各源负载分别为既有global_group2、phase_joint、水平P2共同row_phase_joint_pair；global_joint_pair与global_group2完全相同，不重复冒充新mask。固定ordinary/lifting × corner/interior四窗，不扫组数，不训练。

执行预算：同96×8 RF、128KiB状态/系数、SR64/SW64/CR256、原写回仲裁。每次目录只缓存9邻域×P2的18个SR64字（144B载荷，收费解包进9个RF向量），共同给各coalesced臂。9条几何描述、48bit静态mask、offset位图/循环控制均显式计费；H8跨两个SR64，源地址phase依据实际sy/sx，边界与P2部分许可逐源处理。最末SR64响应缓存仍沿用原Machine；不增加端口。门图输入DMA相同，metadata冷填额外收费，不能宣称外部输入字节被删除。

核对：保留原k=c×9+ky×3+kx顺序；各次目录NRV字节、live及后继Acc48/RNE/更新I24/门/PED均与相同mask完整扫描和独立整数gold逐值比较。完整局部链给服务/字节/目录分项；不是整帧、RTL或PPA。负结果只停本控制布局，三种现有mask与其新AEE准入独立保留。

独立审阅后的必要对照补全：predicate只付实际许可检查，位图构造仅iterator付；任意四phase行完全相同的mask统一获得常量编译权限，每H8选一次真实metadata bit并跨过整空组。只重跑受影响两臂，原scan/coalesced结果复用且独立重建NRV再核；首例临时JSON已由最终修正结果覆盖。SR64门字拆uint16、mask拆4×12bit，metadata高水位计至122912。原4窗/3mask之外仅增加一个固定读写背压验证，不扫配置。
