# 非因果源 PSN：前沿合并、部分时间完成与四槽驻留

**真实 RTL 已完成，收益在普通 exact-code 上成立，主要收益不能归给响应 class。** 在未参与新类选择的31个训练帧、每帧固定32位置（992个P1）上，resident-frontier code 相对旧单channel code省 **9.53% ready / 7.32% BP**；相对更强 static64＋next-X预取省 **17.25% / 2.95%**。原始 plain-frontier 的 BP 对 static 只快209拍，几乎持平；四槽驻留是针对实见源码重读成本的一次有界供数适配。

[SUMMARY.json](SUMMARY.json) 给出全部计数与集合；[expanded_old_cycles.csv](expanded_old_cycles.csv)、[expanded_new_cycles.csv](expanded_new_cycles.csv) 每命令一行。没有新训练、AEE、EDA或同频PPA结论。992个P仍来自训练缓存，不是独立验证集。

## A、B、X与执行边界

A 是成熟 ROBDD、按需特征取得、请求合并、有限cache/FIFO。B 是[原 source_classifier](../source_classifier.sv)：十个时间面各有一个图消费者，每次选一个physical channel，却由10个MAC完整生产该channel的十个门。很多完整U没有当前消费者。

X 在本图边改了provider承诺：`producer_valid && producer_active[t]` 才表示有效的 `(t,producer_channel[t])`。每lane固定一个输出时间t，可在同一轮为**不同channel**计算其所需U_t。十个s仍完整有序做MAC，A行仍是各自A[t,s]，函数固定为父A16Q12/X24Q16/τ48Q28；没有时间截断、近似门或中间RNE，也不声称该整数函数与原FP32源PSN无损相同。十个独立DAG只推进实际完成的active节点。未算U是无效数据，不冒充完整原source门字；TB没有喂gate/code/分支答案。

合法性来自这里非因果T10的全部X已可读取、各输出t的PSN点积独立，DAG每条路径仍按原固定变量顺序。部署分类器需要最终投影的emitted code；本叶没有承诺外层任意消费者仍能取得被取消的完整raw-g/U。最终接口仍为6组×T10个code；本轮尚未把这个新provider接入joined FC1/完整网络。

## 固定两版布局与共同强控制

所有臂是同一个 [frontier_source.sv](frontier_source.sv)，运行时模式选择，固定D-only entropy rank、32bit节点，保留同一旧单channel child预取规则（下述边界）：

| 臂 | mode | frontier | resident | 行为 |
|---|---:|---:|---:|---|
| static64＋next-X PF | 1 | 0 | 1 | 普通消除32个共同Hamming列，完整T10生产，预取下一个已知channel |
| 旧单channel code / class＋PF | 2 / 3 | 0 | 1 | 原完整T10 provider；同样可使用4槽，但固定有序路径无重复channel，实际无槽命中 |
| plain-frontier code / class＋PF | 2 / 3 | 1 | 0 | 当前前沿按D-only rank取最多4个不同channel；每批真实读入所有所需channel，不跨批复用X |
| resident-frontier code / class＋PF | 2 / 3 | 1 | 1 | 优先前沿中的已驻留channel，再按D-rank补非驻留；同channel合并。空槽优先，否则从FIFO指针选择本批未引用槽 |

保留最初plain结果，没有把其ready正结果改写为失败。驻留模式不能淘汰已被当前batch引用的槽：先保留全部命中槽，再分配缺项；新tag在源词真实到达前无效，XFETCH完成后才置valid。每个命令清slot有效位，组间可保留，跨不同X命令不复用。4槽容量、替换策略固定，未扫参数/预测未来分支。

**独立审阅补充：本版没有完整迁移多前沿的child预取。** `pf_relevant`实际仍是`node_var[t]==channel && child>=16`，其中channel取全前沿最小D-rank；不是所有active lane。resident-first可能先选其他驻留channel，此时这个min channel甚至不在本批生产集合；其child读是有费投机读，不会推进节点。最初生成脚本中拟替换成active-mask的文本没有匹配带child条件的原句；最终SV与全部结果均为这里的单channel规则。报告不把计划当成已实现，按锁定要求未为审阅再改RTL。后续若试active-frontier预取，必须保留同128B graphcache、bank/在途上限并实测额外无用请求和驱逐，不能推定它一定更快。

原ordinary ROBDD、普通请求合并、cache/FIFO均不是新发明。当前项目新增的是把非因果源PSN的**完整T10 channel义务改成消费者需要的(t,c)义务**，同时让有限X供数与图执行真实竞争。class和code都得到此接口，不能将它作为class独有贡献或宣称文献首创。

## 共同资源与实际服务

- 唯一 **10个signed16×signed24 multiplier、10个signed48 accumulator adder**，两级signed40 product流水。lane t固定A[t,s]，每批10拍发射。active mask在流水期间稳定；无效lane不发乘法、不更新U。四channel没有复制四组乘法器。
- 所有基线均有 **128B X holding＝4×2个128bit词**，比旧单channel32B多96B。每词是5×signed24＋8 padding；每lane增加24bit 4:1槽选择，后接原s字段选择。不能把128B临时状态或mux视为免费。
- 图cache仍为 **128B＝8个128bit直接映射词**，每bank一个tag/valid；普通static的next-X预取复用这个同cache。X持久槽独立于图cache，但总容量已对全部臂开放，没有把普通next-X预取挤掉。
- 仍是 **8bank×128bit**，low3地址位选bank，每bank至多一笔pending。参数、图需求/预取、X需求共用同一请求/响应接口。四channel最多8个X词；bank冲突串行，拒绝必须保持地址/valid，不能额外开W/X口。resident hit免的是已经驻留的真实X读，原先发生的装入仍计费。
- 新槽元数据：4×7bit physical-channel tag、4 valid、2bit FIFO；batch channel4×4bit、引用mask4bit、count3bit、10×2bit lane slot、10bit active及8bit issued/received。`source_seen[96]`和统计计数器是重读监视状态，也显式在SV内。
- 其余继承A200B、tau60B、D192B、mask12B、roots24B、rank48B、U60B、两级product100B、十份node/lo/hi/var共65B、raw_word20B、code5B，以及原cache/pending tag与控制。X选择、resident-first匹配/排名/分配、图cache/预取选择的组合时序均未测，不能据周期直接声称Fmax/面积/能耗收益。
- 启动配置仍按实际用途：static30个128bit词，graph21词；新的frontier/resident是同start事务中的模式位。源矩阵每P最多192词，外部静态图区域、A/tau/D输入容量沿父接口，未添加图或权重副本。所有cold装载、请求等待、图读取、MAC、排空和code输出等待均在周期内。

## 31个未参与类选择训练帧：主结果

输入来自父[expanded manifest](../source_class_adapt/expanded_sources/manifest.json)：现有32训练帧各固定32位置，frame0参与新W″类pair选择，frame1..31未参与该选择。原W′与新W″两套source.bin拥有逐值相同X/A/tau/D、code图/roots/rank；只替换class图/canonical。两套static/code全部RTL记录逐字段相同，避免混函数分母。

| 992个P1执行臂 | ready cycles | BP cycles | ready X词 | ready scalar MAC |
|---|---:|---:|---:|---:|
| static64＋next-X PF | 1286624 | 1367968 | 126976 | 6348800 |
| 旧单channel exact-code＋PF | 1176829 | 1432418 | 110876 | 5543800 |
| plain-frontier exact-code＋PF | 1079074 | 1367759 | 139900 | 4224030 |
| **resident-frontier exact-code＋PF** | **1064671** | **1327565** | **112474** | **4224030** |
| 旧单channel W′ class＋PF | 1173106 | 1422911 | 110488 | 5524400 |
| resident-frontier W′ class＋PF | 1060942 | 1314381 | 112086 | 4213500 |
| 旧单channel W″ class＋PF | 1158089 | 1408772 | 108948 | 5447400 |
| resident-frontier W″ class＋PF | 1059105 | 1311467 | 110504 | 4188830 |

周期口径沿父TB：接受start的那拍至DONE可见，`cycles=sum(state0..state13)-1`；DONE握手后的重启没有测。ready/BP是固定两种日历，不是吞吐/Fmax分布。W′/W″ class输出是各自whole-H384响应的canonical code，不是原argmin code，也不是相同模型函数；只有exact-code可在两套模型间共享分母。各class的响应W必须与下游绑定，不能用本叶label正确性替代质量评估。

新W″ class相对自己的旧单channel class省8.55%/6.91%；相对同resident exact-code只再省0.52%/1.21%。后者BP差16098拍中，**5952拍来自既有code/class roots位于不同bank与固定BP日历下的启动参数服务差**，不能全归于class语义。ready比较与同class前后比较没有这项归因混淆；主要新收益已在普通exact-code成立。

## 小集正结果、供数瓶颈与一次适配

原2帧64P保持在结果中，但不再作为主结论：

| 原64P，W′图 | ready | BP |
|---|---:|---:|
| static64＋PF | 83008 | 88256 |
| 旧单channel code＋PF | 71509 | 87055 |
| plain-frontier code＋PF | 66394 | 84034 |
| resident-frontier code＋PF | 65584 | 81850 |
| 旧单channel class＋PF | 71385 | 86537 |
| plain-frontier class＋PF | 66202 | 82975 |
| resident-frontier class＋PF | 65414 | 80947 |

[CPU首探针](probe_frontier.json)已经预报：小集code有效MAC335900→267920（−20.24%），但向量批数3359→3048（仅−9.26%），源词6718→8282（+23.28%）。单个t的BDD路径不重复变量，但别的t以后可能请求已读过的同channel；部分时间生产后不能声称该channel完整T10已经缓存。

扩大31帧后，plain scalar MAC省23.81%，但批数55438→49558，且源channel重读14512次。resident把它降为799次，13712次驻留命中；X词139900→112474，没有额外容量。resident-first在一个病例改变了batch分组，因此总批数49558→49559，多1批；没有用更少批数掩盖驻留机制代价。

| exact-code 31帧关键状态 | one ready/BP | plain ready/BP | resident ready/BP |
|---|---:|---:|---:|
| MAC（每batch10拍，含inactive lane空位） | 554380 / 554380 | 495580 / 495580 | 495590 / 495590 |
| XFETCH供数状态 | 166314 / 350534 | 157298 / 347343 | 142877 / 308112 |
| 图LOOK | 124385 / 148446 | 123846 / 174649 | 123849 / 173725 |
| graph word真实请求 | 134441 / 134392 | 130098 / 130055 | 130098 / 130068 |

plain有效MAC降低没有按比例降低lane批数；BP下较多分散前沿还增加LOOK等待。resident主要修复XFETCH，BP节省39231拍，总代价省40194拍；它没有消掉图cache/分支预取等待。plain BP对static仅快209拍（0.0153%），resident提升为快40403拍（2.95%）。这是扩展输入揭露剩余洞之后的一次适配，而不是只报两帧positive就晋级。

## 实测和独立核验

[run.sh](run.sh) 用Verilator4.028 `--cc --exe`＋make；只用Python3.12 NumPy和C++，引用父已经3.12再生的source.bin，不重复生成训练/图。最终小集1340命令、两套扩展各12324命令，总 **25988命令**，日志均完整PASS。另保留首次plain670命令的 [plain_cycles.csv](plain_cycles.csv)；最终核的resident=0逐字段重现全部670条，普通基线也重现父原周期、traffic和状态。

- 实际 `valid(t,c)` 的完整整数U和gate各检查 **13096250** 次，最终code/class检查 **1559280** 个label。没有将未生产t的CPU金标准计为RTL检查量。
- 每个partial valid必须对应当时真实未终止且ready的node前沿，物理channel等于其var；同一(t,c)不得重复生产。inactive U/gate不被当成有效输出。TB从X/A独立重算完整十项dot，以全16code Hamming＋最小编号tie/canonical核分类。
- 外存拒绝时req地址/valid保持、每bank1pending、物理bank映射、每次X两词完整、图/源/预取请求、实际重复读取、holding命中/最多4槽及最终code背压均核验。三诊断是X全零、正signed24极值、交替signed24正负极值，均没有溢出signed48。
- [verify.py](verify.py) 从真实X独立生成完整gate，再走原DAG与固定四槽FIFO模型；逐病例核batches、有效pairs、channels、hits、refetches、X物理词、全部状态。static/code跨旧新W图的所有结果也逐字段一致。[verify.log](verify.log)保存完成状态。

本轮已关闭部分时间provider、四前沿合并、普通驻留适配及扩大训练帧验证。尚未关闭覆盖全部active前沿的图预取适配、完整独立验证序列/网络质量、与joined FC1的实际重叠执行、跨P配置驻留/无reset回收、供数选择和乘法路径的真实时序/PPA。当前producer事件为内部观察、没有独立ready；图本身立即消费active结果，最终emitted code才有背压。不能把这个事件当成已接入任意可背压下游的通用provider。

独立静态审阅见 [REVIEW.md](REVIEW.md)；该文件由审阅代理维护。本叶冻结后，root在独立frontier_joined继续集成和统一root配置bank，不能将未来wrapper结果回填为本叶已测周期。
