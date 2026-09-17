# 完整 patch 因子链：TC 有效，A/V 换序没有形成普遍收益

2026-09-15。已把两份冻结整数函数接成真实 **P4×T10×K864 source→U8→Zi→V/A→sn2 gate** RTL，未向 DUT 喂 Z、Y、Q 或最终门。固定同一个总128bit服务口、8路算术、相同所有缓存/容量后，区域函数保留 **TC+VA**：比该函数的 dense-masked+VA 节省 **8.0465% ready / 7.6625% BP** 服务。AV在该函数上更慢；普通 R32 的 AV 仅省 **0.3567% / 0.4505%**。这是完整有限执行点的筛选，TC、布局和矩阵交换都是成熟 A，不单列新公式或新颖性分。

| 冻结整数函数 / 执行臂 | 四capture共256个P4 ready总拍 | BP总拍 | 物理读字节 |
|---|---:|---:|---:|
| ordinary R32，VA（dense与TC完全相同） | 21,737,772 | 22,865,466 | 7,517,280 |
| ordinary R32，AV（dense与TC完全相同） | **21,660,236** | **22,762,448** | 7,517,280 |
| regional R96/active48，dense masked VA | 34,457,400 | 36,312,825 | 12,335,808 |
| regional R96/active48，dense masked AV | 35,568,990 | 37,438,113 | 12,335,808 |
| regional R96/active48，TC VA | **31,684,780** | **33,530,350** | 12,335,808 |
| regional R96/active48，TC AV | 32,477,674 | 34,337,270 | 12,335,808 |

每行输入为四份旧 `integer_valid10/capture_*.npz` 的全部固定64个P4，共256例；每份含八区域各8例。各函数内部 VA/AV、dense/TC 输出逐值相等；两函数之间不相等，不能跨学生把差异解释为同函数架构加速。A发生在latent域时为 `Q=Aq14·Z`，末端 `Uout=Q·V`；VA为 `Y=Z·V` 再 `Uout=Aq14·Y`。mask跨整个T10共用，两者间无RNE/截断/饱和/非线性，所以交换是精确整数恒等式；最终比较同一冻结tau48。旧断开尾16rank已从ordinary裁掉，不以R48空尾制造分母。

**源读减少、算术减少和周期分开看。** TC改善区域模型的8lane利用率及紧凑Z：U向量更新 **3,046,861→2,446,584**，U ready阶段 **15,283,028→12,510,408** 拍；总收益正好来自这2,772,620拍。当前固定U行cache/gather下，TC与dense的外部U读取都为453,804 word，故不宣称带宽减少。恢复V时两者实际读取147,456个H8字，并用同一TC编号恢复global rank；不是把紧凑序号当V地址。

| ready阶段（全256例） | R32 VA | R32 AV | regional TC VA | regional TC AV |
|---|---:|---:|---:|---:|
| U生成/源供数/Z初始化 | 8,093,496 | 8,093,496 | 12,510,408 | 12,510,408 |
| V供数/cache/投影（AV包含末端tau） | 11,431,504 | 12,933,000 | 16,957,504 | 19,026,768 |
| A阶段（VA包含末端tau） | 2,184,612 | 605,580 | 2,184,612 | 908,242 |

区域TC的非零 A 标量乘积 **2,731,292→1,362,311**，但非零Q几乎填满latent时间面，V更新 **4,196,160→5,897,040**。省去的A阶段1,276,370拍被V阶段新增2,069,264拍抵消，净慢792,894拍；不是漏算参数供数或只用MAC估周期。ordinary也有相同现象，最后仅剩77,536拍优势。首8真实小集普通AV的2.17%未外推到全256例。其余配置/TC扫描/最终96bit gate pack及done包含在总拍内；原始phase0/phase4的IDLE归属受上一命令stage寄存器影响1拍，全部命令总拍及上列三个算术阶段不受影响。

## 有限资源与真实流量

- **唯一8个signed16×32 multiplier、8个48bit ALU**。U门控AAC、V每lane有符号移位加（明确另需8路最高14位左移/符号选择逻辑），A用同一乘法器/ALU；没有80路T10或第二完整核心。共享acc8×48=48B、operand8×32=32B。组合乘加/移位的时序未EDA，不承诺等Fmax/PPA。
- 外部共同**256KiB图像，一个总128bit请求/返回、最多一在途**，与旧8×128模型不是同吞吐点。source每字3×40bit，完整864K需288字；U每字16×INT8；V一个字同rank的H8，`h%8`定位到16bit字段、只用6bit。该普通字段重排给所有臂，没有复现作者其他布局或把旧`h*R+r`同bank读取当免费H8。实际占用参数+一个P4源图像为ordinary44,416B / regional112,000B，共同分配容量不变。
- 每P4真实读 descriptor1字、当前区域mask1字、A13字；tau需360字，在每(hb,t)实读三个字后供四P复用。U/V/source走同口，所有请求、返回等待、输出反压和done计入。外存最初写入该图像未测，周期起点是start后对这块已存在存储的服务，不能称完整外存冷装。没有跨命令参数cache免税；每命令动态cache/配置都按RTL重新处理。
- Z共同 **8bank×480×16=7,680B**；phase scratch共同 **8bank×480×48=23,040B**，只存VA的Y或AV的Q。每bank在一状态只服务一个地址；U分READ/ADD两拍，A读8lane，V只读一个rank的bank，写入无同bank双地址。Z静态界15bit、Y32、Q30、最终U最多47bit；用16/48bit存储，无隐含舍入。TC减少实际使用Z行、AV紧凑Q行，但所有臂获同完整容量。
- 所有臂同 **H8 V cache96×64=768B**（8个byte/行，每byte6bit有效），经外部口实装live rank；VA/AV都用它，AV未逐P/T重复外读V。另source行cache16B、U行cache16B、通用返回holding16B、U组装holding8B、A200B、当前H8 tau48B、TC map24×5bit=15B、mask24bit、source/pending支持码各40bit。V cache的global→compact加载和非连续U gather均在RTL。CSV `U_cache_reads` 是实际行抽取次数，包含新fill后的首读（SV内部字段名仍为`count_U_cache_hits`），不冒称全部为省掉的外读。`zero_source_words`计864个逻辑40bit/K单元，`source_words`计真实128bit读取；`A_updates`为8lane MAC拍数，`A_scalar_mac`为其中非零标量项数，不以此声称乘法器门控功耗。
- 最终40×96bit gate pack **480B**逐H8写全后按P/T顺序退休；没有另存一份完整U阵列。状态/标签还包括两14bit cache tag、有效位、u_got8bit、两个8bit FSM、请求地址14bit、18个32bit索引/计数控制寄存器等，均在源码。TC由实读mask逐TC解码，不接受TB latent清单；dense也保留死块/live-rank跳过、同cache和AV权限。

## 核验与去留

[prepare.py](prepare.py) 直接读冻结的 [ordinary_r32.npz](../ordinary_r32.npz) / [regional_r96_a48.npz](../regional_r96_a48.npz) 和真实source words，未重新量化；Python int64重新生成所有Z/Y/Q/U/gate。C++再从原source和参数独立计算一遍，核对序列化gold及VA=AV，DUT只有真实参数/source口。[tb.cpp](tb.cpp) shadow实际Z/scratch写入，在阶段末核全部compact/dense数据；逐H8核最终U48和gate，最后核全部96bit pack；还检查实际TC解码、单在途、未初始化读取、请求/输出/done保持。

小集为首capture八区域各一P4，加zero/dense/交错支持三诊断；两函数×dense/TC×VA/AV×ready/BP各两遍。通过后四capture全256例各臂一遍，只有进程初始reset，连续换源/区域/mode。总 **4,448个完整P4命令、17,080,320个最终U/gate值全部通过**。[profile.py](profile.py) 从冻结U/V/A与原source独立重算 **71,168项工作/物理字等式**，并用完整ready状态费用加实际请求/返回/output/done等待核 **4,448条总周期恒等式**；全PASS。没有CPU中间数据馈入硬件。

保留regional的成熟 **TC+VA** 执行作为该整数函数的完成版；AV在regional上退出，ordinary AV只有不足0.5%的有限点收益，不据此新建创新标题。当前regional量化相对保存FP32权重的局部门翻 **9,390/983,040（0.9552%）**、Y相对RMSE约20%，ordinary也有2门差；这是编译阶段的局部数学参照，不是本RTL错误或网络AEE。详见 [INTEGER_RESULT.md](../INTEGER_RESULT.md) 与 [REVIEW_INTEGER.md](../REVIEW_INTEGER.md)。未借旧学生AEE，没有训练/生产/EDA；下游Conv2所需18个sn2位置与halo、完整网络质量及作者其余系统均不在本P4挂点闭环中。

复现入口：[run_all.sh](run_all.sh)，Python3.12、Verilator4.028/C++17、RTL assertion开启。原始 `ordinary/regional_small/full.csv` 与同名日志完整保留；[SUMMARY.json](SUMMARY.json)、逐记录 [summary.jsonl](summary.jsonl)、[profile.log](profile.log) 给出全量数值和独立核验。新增文件仅本 `rtl/`，未改旧编译、冻结因子或生产树。
