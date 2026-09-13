# 两源父和转发与按时间归并：最后一个固定执行接口

上一目录 [rtl](../rtl/REPORT.md) 的单 scratch 按码构造已经封口，不再修改或重选参数。本目录只增加 mode6，与同模块 mode5 比较；前者仍是 product-reuse 的执行 A，不构造新代数或完整 Prosperity 复现声明。

mode5 的原生 C4 四源读取、按需四 W 暂存、静态 C4 退休、合法消费者枚举、无空对选择气泡和两源共享全部继承。双方增加两份 signed17×8 父和寄存器，各自只在完整 T10 的 source-pair AND 非零时构造一次，共用原八条 32bit 数据加法链。mode5 依然分两次遍历 pair 时间成员，并由相应父和寄存器给 psum；其周期、请求、父和数和输出必须与上一目录完全一致。

mode6 先完成实际需要的父和，最后一次 W 读或最后父和 build 直接开始完整四源 OR 的时间枚举。某 t 只有一 pair 活跃，原 W 或该 pair 的父和直接给 psum RHS，无 scratch copy。只有两 pair 同 t 活跃时，支付一拍 PAIR_MERGE，将两边选出的原 W/父和用原八条加法链合成 signed18 scratch，随后仍单独 PS_READ、ADD_WRITE。前一个 stage 的 NB W/parent 写在下一状态消费；时间分支用 selected_time，不能用尚未更新的 time_idx。

源仍单 10bit 读口、W 每拍同行 8×16bit、psum 原 8×480×32bit 且读写分拍；没有数据额外端口、十五项表或 TB 供和。新增 272bit 父和状态与 32bit merge 诊断计数器为两模式共同预算，原寄存器全部保留。两个父和同时可读仅指这 34B 寄存器，不是增加 W SRAM 端口。

对每个实际 C4 源和目的，设 H 为两 pair 同 t 都活动次数，K 为有活动 pair 数。若无额外控制空拍，预期 mode5−mode6=`2H+(K−1)`：减少 H 次三拍 psum pass，新增 H 次 MERGE，并少 K−1 次空 pending 终止。该式仅作测试预测，正式结果来自 Verilator counters，再由独立源/几何闭式核对。

固定测试：dense、block magnitude25、Cin magnitude25、Cin fullcost25、mixed retirement25 五臂各八真实 tile，加全码/极值/重复时间、边界毒值、全一、全零、角点、全 mask 等原功能输入；mode5/6 各无背压/command-relative 背压、每次两条同配置不复位重启。独立完整 Q16 卷积 gold；输入装载/clear/drain 的计费规则沿用上一报告。完成该接口后停止新增实验。
