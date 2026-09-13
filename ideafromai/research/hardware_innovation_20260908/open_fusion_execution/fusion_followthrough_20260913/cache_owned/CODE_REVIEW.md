# 跨帧门缓存：独立代码审阅（2026-09-13）

范围：只读本目录`execute.py`、`PLAN.md`，沿调用链检查`matched_local_chain/run.py`、原`run_windows/run_chain/machine.py`、`integrated.py`、`encoded_consumer/binding.py/kernel.py`及捕获/执行收据。没有修改执行代码，没有运行训练、EDA或新实验。本文件是本次审阅者唯一新增文件。

**最终结论：发现的两项必须修问题及一项控制状态显式化问题，root已修正；改后静态核查未发现新的数值或资源冲突。** 六臂×四帧自然局部链均0差；自然72个warm query均无命中，当前布局只有额外成本。另读到12项定向source-only检查，repeat与非零扰动hit路径均实际执行且门0差。两类证据分开，不将定向命中纳入自然性能。本文认可的是有限CPU载荷模型中的当前代码，不是RTL/PPA或同面积/同频率证明。

## 已发现并复核的修正

| 问题 | 原风险/证据 | 当前修正与状态 |
|---|---|---|
| 证书常量地址96000覆盖preview系数 | `run_windows.window`先将完整preview系数写入[0,124544)，source初始化随后在96000写320B界值，覆盖U权重。首跑Z/raw/BN1出现约1e35数值，sn2亦失败。 | 已改`CONSTANTS=126976`，范围[126976,127296)，低于131072并避开preview映像和整数后缀系数；失败保留为`setup_coefficient_overlap.log`，不能作性能收据。 |
| 三个半径RF一次priority读取 | 原`finish_certificate`只收一槽，却读RF90/91/92三个向量lane0；原双RF读口无法在同槽完成。 | 已拆两槽：第一槽读RF90/91，结果存原`scalar_collector`；第二槽读RF92及collector。半径≤1024适合既有3B scalar collector，此阶段无其他活用途。无新增第三RF读口。 |
| query元数据跨历史读取的保存未显式定位 | 原valid/radius解析后存Python scalar，radius跨30个历史SR响应存活；虽然只需很小状态，但没有明确硬件归属。 | 已以有费ILOAD存RF94，初次valid分支读RF94；最后以RF72+RF94两读完成最大差与半径比较。新增ILOAD及等待计入exact/certified；raw不受影响。 |

root已对受影响的exact/certified ready/stress完成重跑；本文件最终表只用修正版六份JSON。以上修正由root实施，本审阅只反馈并复核。

## 整数闭边界与证书来源

原`ISOURCE`门谓词是`S>=K`或`S<=K`，令`a=direction*(S-K)`，统一为`a>=0`。`ICACHE_MARGIN`在真门取`a`，在假门取`-a-1`，给出可保持原门的**最大包含端点的整数dot扰动**。例如`S=K`的真门margin=0；`S=K-1`的假门margin=0；这两种情形都不允许±1的最坏扰动。负direction的端点通过同一式镜像，未发现off-by-one。

界值为`L_t*d`，其中`L_t=sum_j abs(As_q16[t,j])`；query取全部T10×H8输入差的无穷范数。每个门要求`L_t*d<=margin`，随后对十个时间行和八个lane求AND，选择最大固定等级。因`|A_t Δx|<=L_t||Δx||∞`，这个闭边界是保守的。常门不必约束半径，当前constant分支保持三个许可位；这是全域常门语义，不是跳过未计算的动态门。

门的K、direction和operand来自既有真实编译程序，继承已折叠RNE/sat的整数门原像。`observe_gate`在源operand尚未被覆写时，通过有费`ICACHE_DOT`读取实际RF并恢复原S；捕获的`source_S48`只与这个值比较，既不选择radius，也不决定hit。每个miss最多检查24×10×8=1920个实际dot；这与是否自然命中分开。

## RF、SRAM和生命周期

| 对象 | 位置/生命期 | 核查 |
|---|---|---|
| 当前I24 | 原源程序前十条加载RF0–9 | assertions确认source_t及dst均0..9。query只读取这些RF，写70–72及94；fallback从PROGRAM[10:]继续，当前输入不重复加载。 |
| query临时 | RF70历史、71差/绝对值、72跨T最大值、94元数据 | 历史逐T读入、DIFF/ABS/MAX均有费；lane归约采用固定1/2/4三步。最后分支最多读RF72/94。 |
| 证书临时 | RF73原dot/margin、74比较结果、90–92三级累计许可 | 原source程序dst<70；临时不覆盖源图。gate输出用RF95；所有操作沿已有两拍integer写回并等待。 |
| 历史输入/门/元数据 | state[114688,121024)，24×264=6336B | 每项240B真实I24、16B真实T10门词、8B valid/radius；8B对齐。source输入[90112,92992)、source门、preview/sn2和整数后缀低区均不覆盖它。 |
| 证书常量 | coefficient[126976,127296)，320B | 与state是原本独立的两个128KiB池；不能因为地址数值相近就判冲突。完整preview系数[0,124544)，整数后缀约41KiB低区；每帧重新填这些低区不破坏证书。 |
| 小控制暂存 | 既有scalar_collector及至多16B门collector | 两级priority暂存不跨source阶段；源门collector沿既有普通源模型，缓存返回至多两个SR64响应形成16B载荷。可容于原64B staging；没有引入随帧增长的执行数组。 |

`kernel.make_callback(None, False)`是raw连续消费者，**不会**走CODE=114688或RECON=110592的量化存储分支；因此这些旧常量名不构成本次history冲突。未来若换成量化callback，必须重做此地址寿命判断。

miss时先保存实际RF0–9输入，再执行原图，最后保存真实门及valid/radius。当前每项query/refresh完全串行，下一帧不会在半写入状态读取此项，因此最后写元数据足够；这不证明将来并发访问同entry也安全。hit时不写history，下一次仍与原已完成参考比较，避免逐帧小变化累计越过安全球。

同一个Machine跨四个输入持续保留SRAM、时钟、ready/pending及背压相位。`prepare`只构造外部reference；没有在帧间创建fresh Machine来保留历史却重置成本。固定映射为interior前两个pixel的12个H8/像素，不存动态地址tag；当前只适用于同参数、同几何、同坐标序列，不能推广成任意frame/window共享缓存。

## 成本和共同资源

warm query真实读取1个元数据SR64和30个历史输入SR64，执行十次I24加载、十次DIFF/ABS/MAX及三步lane max归约；当前I24原首读仍全付。miss刷新历史至少30个I24 SW64＋2个门SW64＋1个元数据SW64；首帧另付24项valid初始化。hit仍实际读两门词并写出完整source门，没有删除原raw I24或后继消费者。

certified另外付320B常量冷填、实际CR256界值响应、各门dot/threshold/margin/三级比较与AND、三级lane归约和两槽priority。范围计算在离线静态参数上，执行用实际CR响应；未把capture S48或Python计算的本帧dot当免费输入。

ICACHE ABS/MAX/归约/比较等固定两拍延迟是**新ALU及控制的模型假设**；lane旋转需要布线/选择，priority使用控制逻辑，未综合或验证时序。同RF/端口/数组容量不能写成同面积。源码中的ROM fetch统计沿用普通源程序合同，新增cache控制也没有独立RTL/微码容量闭合，不能称现有RTL已经支持全部新操作。

四帧连续背压保持原period32形状，各arm由于先前完成时刻不同，后续帧开始相位会不同；应报告总时间和start/end phase，不能把每帧差直接归因某一条ALU。ready与stress也不能转成任意系统带宽的外推。

## 完整消费者、gold隔离与收据边界

source由实际I24 DMA/RF生产全部interior 11×11×T10×C96门。`run_windows`虽然构造reference门词，但当前`stage.preview_directory`只使用其形状，NRV实际读取Machine SRAM并执行完整K864；reference门值不决定跳过。随后完整preview-U32/V/固定BN1/非因果sn2执行，实际sn2留在同Machine。`binding.consumer_run`继续实际K864 Conv2/U16/F、BN2/raw残差、projection门、原U24/V96连续PED及真实输出读取/DMA槽。

callback只接收Machine、常量、坐标和geometry，不接收reference U/PED。`prepare`用独立字面source、C++保序preview和整数oracle构造检查值；重建后的gold只放检查字段，不回填SRAM。source_S48观察数组、captured_endpoint_comparison和输出收集均是观察者，未反馈控制。

这里的“完整”限于固定interior局部链，不含上游I24生产、native projection、全域动态BN/join、完整帧或整网。GPU捕获的四次forward是输入和端点来源，不是该缓存的速度测量。精确改写可沿用同一matched dense父的已有NB0资格，不新增或借用其他父AEE。

自然已读六份最终JSON，每份四帧，阶段和与总槽一致；共24条局部链的source/preview/updated/projection/U/PED检查为0差，与GPU捕获sn2/updated/projection/PED也全部0差。certified每压力共检查7,680个实际S48。`summarize.py`先断言每臂四帧、各检查为零，再汇总；没有把逐帧部分JSON当作完成。

| 四帧总服务槽 | raw | exact | certified | exact / certified相对raw增加 |
|---|---:|---:|---:|---:|
| ready | 10,838,630 | 10,856,294 | 10,887,842 | 17,664 / 49,212（0.1630% / 0.4540%） |
| 固定stress | 11,961,370 | 11,980,858 | 12,013,018 | 19,488 / 51,648（0.1629% / 0.4318%） |

数据来源为[自然简表](summary.csv)及六份原始`*_ready/stress.json`。每个缓存臂有72个自然warm query、0个hit；不能把24条配置当24个独立视频样本。

另核`check_hit_path.py`和[定向收据](directed_hits.json)：每个mode/pressure使用独立Machine，先从真实首帧截取固定两个pixel冷填，再重复输入，最后全值+1；三步都调用同一个实际source函数，当前gate/dot由独立字面整数计算仅作检查。exact和certified在ready/stress的repeat均24/24命中；+1时exact为0，certified为21/24非零变化命中，剩余3项实际fallback并检查240个S48；各项1,920门位均0差。这补上真实history读取、cache返回以及有非零扰动的接受分支，**没有**执行synthetic完整消费者，也没有增加自然命中率/性能样本。

最后只把`capture_owned/opportunity.json`作为离线解释阅读：相邻帧72组的完整80门词相等次数本身就是0，同一L1-margin判据未量化半径也0通过，因此当前自然失败不能归因于{64,256,1024}三级量化过粗。239/576个lane的T10门词相同，但逐lane同类L∞证书仍0通过；这些是oracle/数学机会统计，不是有费硬件命中。当前全miss会每帧刷新参考，恰与此相邻帧统计协议一致；其他参考选择/粒度不在本次裁决范围。

本代码可以作为完整有费cache迁移的负结果保留；当前24项/三级whole-H8布局不提供加速主张。没有剩余必须修的已定位bug；真正RTL延迟、面积、一般地址/参数切换和未测输入覆盖仍在上述声明边界内。
