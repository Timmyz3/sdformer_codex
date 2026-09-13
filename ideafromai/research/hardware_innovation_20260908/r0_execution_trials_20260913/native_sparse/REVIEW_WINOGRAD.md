# Winograd 最终版独立审阅

2026-09-13。审阅对象为 [winograd_tile.sv](../winograd/winograd_tile.sv)、[prepare.py](../winograd/prepare.py)、[prepare_real.py](../winograd/prepare_real.py)、[TB](../winograd/tb.cpp) 及 **共同 64 项 static successor 后的最终 88-case 结果**；作者已确认当前文件版本。没有修改作者文件或重跑 RTL 战役。仅在本目录做 [有限数学/格式/计数交叉核查](review_winograd_checks.py)，结果见 [review_winograd_checks.json](review_winograd_checks.json)。

**结论：在所声明的整数函数和 flop/mux 多读资源合同内，未发现阻断性算术、地址、握手或静态跳零公平性错误。** 功能通过与同函数执行收益成立；新颖性仍未成立。该核的内部读口比邻目录 native_sparse 更强，两核周期不能直接排名。以下给出通过证据与实际边界。

## 1. 整数函数及舍入

`G2=2G` 在两侧各出现一次，因此 `U4=G2·Wq·G2ᵀ` 对应最终分子 **4 倍**普通卷积。SV 的 `Bt/At` 符号和 prepare 中一致：TR1/2 第二行做加法，其余做指定差；逆变换第二输出行/列均为 `x1−x2−x3`。对 9 个权重基与 16 个输入基的 **144 个双线性基组合**独立复核，`At[(G2 W G2ᵀ)⊙(Bt S Btᵀ)]Atᵀ=4·conv(W,S)`，通过。[SV 行 41–46、110–124；prepare 行 5–8、23–28]

任意坐标 mask 后，不能直接声称存在同函数普通 3×3 核。当前 `E4=L diag(Um) R`，`L=At⊗At`、`R=Bt⊗Bt`，形状每 `(n,c)` 为 **4×16**，分别保留输出相位和原生源位置。`masked_wino` 与 `masked_expanded` 都在完整 C 累加后执行同一个 `/4` RNE；prepare_real 又将 U4、E4 和 masked gold 与数据分支逐数组核对。这是正确的同函数控制，不是假“反变换 3×3”。

完整穷举 65,536 个二值 4×4 输入确认：15 个 V 坐标在 −2…2，只有 ξ=5 在 0…4。**−3 不可出现**，所以只检测 `vnow==3` 的付费 3U 缓存足够；±2/4 用移位。signed4 V、signed24 U/E、signed32 3U 和 signed48 累加覆盖当前合同；最终 signed32 范围有 prepare 的真实检查及作者保守界，不是未经说明的截断近似。

SV 行 138–143 的 `q=mem>>>2` 是向负无穷取商；低两位是非负模 4 余数，原 bit2 是 q 的奇偶位。因此 `r>2 或 (r==2 且 q奇)` 时加一，正确实现负数 ties-to-even。独立核对 −8192…8192 共 **16,385 个四分之一格点**通过；真实 masked 分子中还存在 **1,091 个负 tie/floor偶、1,095 个负 tie/floor奇和 3,224 个正 tie**，不是只用正数证明 RNE。

原位 inverse 顺序也安全：第一阶段先写 row0 四个坐标，再写 row1，后者依赖的原 row1/2/3 未被提前破坏；第二阶段 `(0,0)→(0,1)→(1,0)→(1,1)` 中，每次覆盖的位置都已结束最后使用。对 16 个 M 基向量按源码覆盖次序复核，与 `At M Atᵀ` 一致。各时间/半组/lane 地址相互独立。[SV 行 98–108、262–272]

## 2. 密排、半组零与 static successor

W16 的 H16 向量是 32B；U24/E24 是 **48B，起点在 CR32B 行的 0/16B 位置交替**。SCAN 用第一个存活 O8 半组确定 first_line，用最后一个存活半组确定 last_line；DECODE 相对实际 first_line 重定位且只解存活半组。所需范围最多两行，512bit assembled 足够；两行缓存命中/装入也实际经过状态和握手。[SV 行 86–93、213–240]

独立解码了四格式 **672,768 个存活系数**，逐值等于编译矩阵。真实 masked U 覆盖“仅上半组、base%32=0、仍需两行”的 **288 个向量**，以及上半组对齐后只需一行的情况。**半组清零不保证删除一个物理 CR 行**；应看实际 request/cache 数据，不能把 `half_zero_scan_cycles` 当节省的 CR 词数。该 debug counter 的条件还包含 `support==0`，它本质是 SCAN 观察计数。[SV 行 149]

64 项 successor 对所有模式共同开放：只查当前有限块内更后的非零 support；没有候选则移至下一块边界，整空块仍付 SCAN。按 metadata 独立重建访问集合，没有漏掉存活 key 或重复；真实每 tile 的 SCAN 为 direct 5,184、wino 9,216、masked_wino 7,296、E 20,736，与全部 64 个真实结果一致。

这是必要的强控制修复：E 的八 tile 总周期已由旧逐项扫描 **527,200 降至 398,176**，masked Wino 由 306,999 降至 291,639。最终同函数收益应为 **26.76%**，不能使用旧扫描分母夸大收益。旧版明确在 `old_scan_control/`，当前正式结果未混用。

这里的 support 是 **64 项并读加 current entry 的寄存器/mux 网络**。128bit 是配置宽度，不能据此声称已映射单口 128bit metadata SRAM；部分 group 起点与配置行并不对齐。

## 3. 有限资源与背压公平性

| 内部对象 | 源码实际权限；不能简化成什么 |
|---|---|
| source | SCAN 同时取得同通道 T10 的 16bit 空间字；transform 又有 8 通道读网。不能写成单源词 1R。 |
| V | SCAN 10 个 signed4 并读；TR2 8 写。 |
| temp | TR2 同时读两组 8 元素。 |
| M | inverse-A 每拍 **16×48bit 并读**；AAC/inverse-B 8R+8W，组合读加数据链后写回。不能按 native 的分拍 PS_READ/ADD_WRITE 或单口 SRAM 计费。 |
| 算术 | 8 条显式 signed48 carry-chain 由变换、3U、AAC、inverse、RNE 共享；地址/选择逻辑另计。 |

作者当前 [资源合同](../winograd/resource_contract.json) 已明确这些权限、外部 CR256 与两行缓存及未做 SRAM/PPA。三模式实例化同一个模块、数组预算和编译/跳零权限，**核内比较公平到该固定合同**；不是同面积/Fmax，也不是与 native_sparse 同硬件。内部数组仍是组合 mux，当前外部 CR 事务和状态周期可核查，不能额外把内部组合取值解释成已测 SRAM 能耗。

TB 每 case 完整配置、120 拍源输入、冷 CR cache、全部输出与 done 都计费。CR 最多一笔 outstanding；正常/压力响应分别 1/4 拍，REQUEST ready 与 RESPONSE valid 分开处理。输出背压逐拍保持数据和地址。独立复查 **88 行**：状态总和=cycles、request=response、CR bytes=32×request，REQUEST/RESPONSE/LOAD/SEND 分别与真实等待相符。另从实际 S/V 与半组 support 推导全部 **64 份真实 AAC 和 SCAN 次数**，全部一致。

入口是已经包含 padding 的连续 4×4 块；本核没有 native_sparse 的全图 origin 判断器，也没有全图 line buffer、tile 重叠管理、PSN 或后继 norm/residual。其较宽输入装入和多读网不可在跨核比较时省略。

## 4. 性能、新颖性与剩余覆盖

最终无背压八 tile：原 binary direct **178,948**、exact Winograd **351,567**、masked Winograd **291,639**、同函数 masked E **398,176** cycles。无 mask 与原 direct 逐位相同，当前却用 **1.965×**周期；mask 后仍为原 direct 的 **1.630×**。masked 对 E 的 26.76% 只证明这一函数的分解表示比其显式相位展开更省，不能据此宣称超过原卷积强 A。

**masked_wino 当前就是 WINS 类坐标结构加普通有限值/零门控执行底座。** 四种模式中没有进一步独立出的“普通 WINS 与新 X”同质量消融；坐标删除、`V≠0 ∧ support≠0`、±2 移位、两行缓存和 3U 预制均不能单独归为新机制。E 控制已补 static successor，但不等于遍历了所有合法公共子表达式/子模式复用；新标题仍不能由相对该展开表示获胜推出。

质量分支现已完成 diverse10：coordinate25=1.172116、dense Q16=1.163297，均低于历史较严格 NB0=1.454603；fresh NB0=1.462941。**本次不恢复 +0.005 门，也不因相对 dense 的 AEE 变化判整个家族失败。** 模型评价仍是 CUDA 浮点消费者，phase 路径没有模拟 RTL 末级 RNE，valid825/全网 bittrue 尚未覆盖；采用 [数据分支说明](../data_and_quality/README.md) 的限定。当前负结论来自所测硬件点比原 binary direct 慢，而不是擅自提高质量门。

非阻断覆盖缺口：当前真实半组数据没有覆盖“仅下半组、base%32=16、跨两行”这一组合；格式公式可核，但尚无该分支的专用 RTL 向量。TB 每 case reset，未检验不 reset 连续换模式；源码对 skipped-V 的 source guard 和 per-start cache invalidation 静态合理，但不应声称已做多命令覆盖。还有常规 SRAM/时序映射及整图消费者闭环缺口，作者已如实保留。

可据此收口 **F2/U24/H16/两行缓存/当前坐标 mask** 这个固定点。更强源/系数驻留、训练后支持结构或减少 V 实体化的新接口尚未试；本审阅不否定整个 Winograd、WINS 或有限字母表家族，也不据现有结论追加资源/参数扫描。
