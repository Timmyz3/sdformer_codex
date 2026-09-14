# algorithm_sparse 独立实现审阅

2026-09-14。审阅者未参与这三项实现；只读 [lossy_r8.sv](../algorithm_sparse/lossy_r8.sv)、[wrapper](../algorithm_sparse/lossy_tile.sv)、[reference.py](../algorithm_sparse/reference.py)、prepare/TB/verify、参数与396条运行收据，另写独立整数公式检查，未重跑RTL或GPU。**未发现推翻功能或当前同资源周期差分的阻断问题。** 本结论不替代有损函数的825质量评价。

| 必要检查 | 独立结论 |
|---|---|
| 共同前端与强控制 | 九模式均完整执行C96/K864/R8/N96/T10与P4，使用原生16词源窗、双P打包、共同整零K过滤、完成z支持和整R8 Q2缓存。mode0并未换回旧scalar弱控制。|
| Q2位宽 | 所有模式同为signed16×signed14乘法，结果进入32位共同加法链；实际固定常量的原型残差仍能容纳于signed13；实现保守地单独存14位并直接送乘数，不能把14位说成此固定点的最低需求。不能与旧19×13核作等面积比较。|
| 位宽与累加 | 固定Q1∈[-3,3]，完整864项使每z∈[-2592,2592]。重建近似值仍在此域，原型残差保守界±5184，signed14容纳；时间hold的变化另有更紧界，见下文；每个完整/混合新旧坐标的Q2点积绝对值≤8×32768×2592=679477248，signed32可容纳。|
| signed13 Δ权限 | mode7复现mode6的逐rank deadband函数，mode8复现mode5的整组hold函数。先检查所有变化rank差在[-4096,4095]，且变化rank数小于新近似向量nnz，才装入13位Δ；否则装完整近似z并清acc重算。没有把越界差值先截断再判断。|
| 时序参考与psum | reference_z始终写绝对近似值，不写Δ；每个空间P的t0强制初始化。hold保留的是前一T的线性acc，delta将Q2Δ加到该acc，full refresh清零；每个T仍STORE并完整输出。|
| 原型函数 | mode3最近K4原型后仅保留一个最大weighted残差；argmin/argmax同分均取低index。384个Q2原型结果独立核对为精确Q2×codebook。即使某rank因无残差而未装运行时Q2 cache，其原型贡献已在静态表中；mode4固定零原型同样成立。|
| 实际消费者 | wrapper的raw和identity独立握手，等待core与consumer都done才退休。所有模式每tile均480次FP32→J转换、480次消费者乘法、960次加法、480次RNE与480个输出字。hold没有跳过真实identity、非线性RNE或I24义务。|

编码费用确实进入状态机：无全零旁路时，每tile mode1/2/3/4/5/6/7/8上界分别为320/160/1160/200/296/152/152/296拍；最终版mode1–4的零向量走2拍EREAD/EWRITE。K4原型并非TB给索引，四次距离、优先选择、再算残差、表读均实际执行；八个真实tile的mode3还付660个256位原型结果读。codebook有独立8lane组合读口，结果表1536B，所有九模式具备这些资源。EWRITE中的nnz计数、signed13比较和选择是组合硬件，虽无额外状态拍，仍有面积与关键路径代价。

冷配置每tile实际3450拍：原共同3361拍加阈值13、codebook4、原型结果48、消费者24。第二命令不reset，配置0拍；输入装入被本小fixture的配置合同包含，不能把这里的“暖计算”直接当流式全层成本。该union资源合同公平支持当前模式比较，但增加的encoder/表/乘数宽度并未通过EDA证明等面积。BASE_MAC仍采用异步z标量读→cached Q2选择→乘法→加法的一拍权限，需要另做物理时序验证。

| 实际8tile无背压强比较 | 控制总拍 | 候选总拍 | 意义 |
|---|---:|---:|---|
| rankdrop 2 → groupdrop 1 | 98870 | 97682 | 减1188拍（1.202%）；两者是不同有损函数，质量不能共用。|
| zero-prototype 4 → K4+残差 3 | 93185 | 98813 | 加5628拍；原型改善此控制的质量但有明显编码税。|
| rank full refresh 6 → rank Δ 7 | 107612 | 102596 | 同函数减5016拍，证明不给rank差分会削弱控制。|
| group full refresh 5 → group Δ 8 | 100916 | 100808 | 同函数仅减108拍。|
| rank Δ 7 → group Δ 8 | 102596 | 100808 | 减1788拍（1.743%）；这是AS3应报告的强对照，两函数质量另比。|

[独立检查脚本](audit_algorithm_sparse.py)固定读取最终零旁路前的`pre_zero_bypass/results.json`快照，没有调用作者reference实现；从原生源重建z，独立执行七个有损/无损函数与两种Δ重放，再核对Q2、FP32→J20及整数I24。[检查收据](ALGORITHM_SPARSE_AUDIT_CHECKS.json)记录396行状态/配置/消费者账，380160个raw与380160个I24重算、42240个J、384个原型值、44项同函数gold别名检查均通过。最终396次作者RTL重放覆盖同一11fixture×9模式×双背压×两次命令，raw/J/I24各1520640值；独立重算没有把重复运行值再计算成新增覆盖。

静态域证明（据root指出的不变量独立核实）：mode5/6及其7/8实现中，每个rank的reference只能是同一固定Q1下某个更早T的真实z；t0是真值，之后只能保持或替换成当前真值，归纳成立。不同rank可来自不同T，但逐rank仍有 Δ_r=Σ_k q1[r,k] × (g_tk−g_ref(r)k)，因此 |Δ_r|≤Σ_k|q1[r,k]|≤2592。即使允许完整signed3含−4也≤3456，仍严格落入signed13。故本合同的越界fallback不可达；当前fixture没有触发它不是合法输入覆盖缺口。审计脚本中±4096/4097与±5184仅验证通用guard数学语义，其中越界组合不属于该固定Q1二值源合同，不能写作必要RTL边界用例。实际mode7/8分别选择296/246个Δ向量，最大差88。

其余覆盖范围：11fixture为8个真实块及zero/ones/poison_corner，固定Q1/Q2，没有更换因子的极值配置。背压/无reset重启已在396次内；本项没有新全帧RTL。

质量以[QUALITY_PROTOCOL](../algorithm_sparse/QUALITY_PROTOCOL.md)和各自GPU收据为准。审阅时groupdrop的[825收据](../algorithm_sparse/quality_valid_1.json)已落盘，其他函数仍有未完成项；本审阅不预判其PASS，也不把模式7/8再算两次新质量实验。十帧、旧函数825或局部整数绿均不能代替新函数825；后继网络仍有浮点执行，不能称整网bittrue。上述结果说明三个工作点被实际执行且强控制有效，不自动构成方法新颖性。

最终补强静审：EREAD在同一次原生z向量读中使用scan_mask判零，模式1–4跳至EWRITE并将code/residual元数据清零；codebook[0]固定全零，故即使work/max_rank保留旧值，重建仍只能得到零。rank/group直接剪枝对零的函数也相同。模式5–8仍比较真正prev，不会错误丢弃held非零参考。新增检查复用原读口，不引入免费另读；零向量仍付EREAD/EWRITE及最后scan/输出。最终SUMMARY真实八块有113/320个全零向量，mode1/2/3/4分别少678/226/3051/339拍，精确为113×(6/2/27/3)；mode3 encoder6229拍。主比较表已更新为最终收据，本段是静态差分及计数公式核查，未重跑前述独立数学审计，也不把替换的396次再计为额外覆盖。

更紧位宽边界见作者[integer_bounds.json](../algorithm_sparse/integer_bounds.json)：固定Q1逐rank正负和的区间宽度最大667，codebook每坐标位于同一区间，故原型残差与时间Δ均≤667；前文2592/5184是一般固定格式的保守论证。保留公共16×14硬件比较有效，但“当前参数实际必须14位”不成立。
