# Partial-time frontier：独立审阅

**当前冻结 RTL 未见阻断数值/协议错误；主要收益属于普通多路径判定与有限输入驻留。** 在992个非 pair 选择训练包上，普通 resident-frontier code 相对 static64+PF 减少17.251% ready、2.954% BP；新 class 在此强 code 臂上只再减少0.523%/1.213%。BP 后一差值含可定位的 root 参数 bank 偏差，暂不能全归类别语义。创新暂评 **3/10**，不是录用概率；没有依据“未找到先例”宣称原创。

本审阅只读最终 [RTL](frontier_source.sv)、[TB](tb.cpp)、两份 expanded CSV 与 CPU 互证报告；未改实现、未重跑 Verilator/CPU trace。独立用Python3.12聚合并复核24,648行费用/slot关系；新旧class镜像中的8,216条非class记录逐字段完全相同。作者的 [verify.log](verify.log)/[SUMMARY.json](SUMMARY.json) 给出全部25,988任务、13,096,250个实际有效U及同数门位、1,559,280个最终标号通过；只把有效 `(t,c)` 的U计入，不把未算目标的CPU答案充数。

## 数值与生命周期

- `producer_active[t]` 是输出有效性的一部分。`LOOK:229–233` 锁定本批 active/slot；`MAC:257–261` 对每个有效t完整遍历 `s=0..9`，仍计算 `Σ_s A[t,s]X[c,s]`，未跳项或借原始门。只有10个signed16×24乘法器和10个signed48累加器；active mask 在两拍乘法流水与DRAIN期间不变，`pv==0` 后才退休。未active的U/门不对消费者作承诺。
- `DECIDE:267–270` 只推进有效t且变量确为该channel的活节点。TB新增的live/ready/node_var断言、`(t,c)` 唯一性检查、实际signed48点积/门比较和最终Hamming/canonical核对都与该协议一致。每个t的源点积彼此独立；原部署rawg/U没有额外推理消费者，整T10并非算法硬约束，详见[前轮消费路径复核](../source_class_adapt/expanded_sources/RESULTS.md)。
- 四槽替换先保留**全部已选命中槽**，再为miss选未被本批引用的空槽/FIFO victim（SV:85–104），避免后续miss覆盖本批较晚命中。每槽tag含全局C96地址，跨C16组保留不会误命中；替换先清valid，两词均到齐且pending清空才提交valid。start清valid/pending/图cache，未用预先缓存的X或免费source gate。
- X地址仍为每channel两128bit词，每bank至多一个pending；同bank的多个槽请求真实串行。XFETCH收到响应才落相应槽/半字，`xreceived==xneed` 且pending为空才离开；LOOK同样等未决读取收完。DRAIN额外要求无`req_valid`，避免撤回背压中的预取请求。请求/输出保持由TB逐周期检查。每任务reset，连续DONE后无reset重启未在本批验证。

## 资源与公平边界

`xword[4][2]` 是**128B独立寄存器驻留**，相对父模块单channel32B增加96B；它没有占用或替代原128B图cache。另有4×7bit slot tag、4个valid、2bit FIFO、10×2bit lane_slot、10bit active及batch/收发状态。`source_seen[95:0]` 是重读计数的仪表状态，不能当不存在；计数器也不构成面积免费项。每t从自己slot取24bit X，增加数据选择逻辑，外部8bank口/乘法器/累加器数量保持。不能据“同ALU/端口”称与旧单槽等面积或等Fmax。

同module的static、整T10 code/class、plain-frontier、resident-frontier均具有相同128B X槽和128B图cache；code与class具有同样的frontier与resident开关，普通code的强控已实际跑过。static/整T10路径没有重复需要的channel，因此额外驻留槽对它们不产生免费命中。代价按实际X/图接受请求统计，`xwords/2 + holding_hits = channel_refs`、每channel两词完整、重读次数及4槽上界都有TB/CSV核对。

一个尚未完全迁好的A边界：`pf_relevant:117` 仍只沿旧全前沿最小rank的`channel`预取双孩子，而不是全部active t。resident-first选择甚至可能暂未选这个非resident最小channel；这时预取仍是合法投机读，但不是当前所有生产义务的预取。作者已确认并保留锁定实现，不能把它写成“所有partial请求已完整预取”。该限制对code/class共用，不是数值错误。

## 992包强对照与负例

这里只合计frame1..31各固定32包；它们未参与本次pair选择，但仍来自训练缓存，不能称validation。服务已含逐包冷参数/启动与背压，表中X是实际128bit词数，`pairs`是实际有效目标 `(t,c)`，不再用整T10通道数冒充运算量。

| 臂 | ready / BP周期 | 有效pairs | X词 | 额外重读channel |
|---|---:|---:|---:|---:|
| static64+PF | 1,286,624 / 1,367,968 | 634,880 | 126,976 | 0 |
| 整T10 code+PF | 1,176,829 / 1,432,418 | 554,380 | 110,876 | 0 |
| plain-frontier code | 1,079,074 / 1,367,759 | 422,403 | 139,900 | 14,512 |
| resident-frontier code | 1,064,671 / 1,327,565 | 422,403 | 112,474 | 799 |
| resident-frontier旧class | 1,060,942 / 1,314,381 | 421,350 | 112,086 | 799 |
| resident-frontier新class | 1,059,105 / 1,311,467 | 418,883 | 110,504 | 778 |

plain的有效MAC减少并未消除反复取X；四槽把code重读14,512降至799，是这次失败后适配的实证。每批仍固定10个MAC issue拍，减少active MAC不能直接按比例换算周期；实际靠合并不同t的当前channel请求减少批数，再用驻留抑制流量。

resident code相对static ready全部31帧为正，BP有4个负例frame[7,26,27,30]；新class相对static BP仍有2个负例[7,27]。新class对resident code各帧为正，但本轮改pair选择对旧class仅再省**0.173%/0.222%**，分别10/11帧为负。新旧函数质量尚未知；不能把原W的质量赋给W″，也不能把普通frontier的17%全部归给响应合并。[新CSV](expanded_new_cycles.csv)、[旧CSV](expanded_old_cycles.csv)。

## 唯一最小布局中性复核与先验

root参数地址由 `boot_addr:43` 决定：code读222（bank6），class读223（bank7）。当前周期式BP日历与bank相关；992包code两段boot合计86,304拍，class80,352拍，**直接差5,952拍＝每包6拍**，占resident新class/code BP总差16,098拍的37%。不同起始时刻还会改变后续请求/输出日历相位，所以不能简单减6拍就称“修正后收益”。

建议且根已在joined分支安排的唯一复核：**code/class各自的六个root统一放同一物理word/bank**，保留同992包、原图ID/容量、X地址及固定BP日历，实际重跑两臂；不扫bank/phase挑赢家，也不改变pair。新收据出来前，0.523% ready可作为较清晰的类别小增量，1.213% BP需保留此归因边界。当前源结果仍是逐包冷启动，不替代joined后参数摊销和消费者费用。

普通多路径BDD遍历、按需特征计算、相同请求合并与小容量FIFO驻留属于成熟A；可参照[Bryant的有序决策图约简](https://www.cs.cmu.edu/~bryant/pubdir/ieeetc86.pdf)及[已有代价敏感特征供数边界](../source_class_adapt/REVIEW.md)，不能把“部分时间有效”命名当新数学。本例可保留的候选增量仍是消费者完整响应商类改变上游必要判定；在普通partial/resident强控到齐后，它已明显收缩。完成同W″消费者闭环与root中性对照比再加前沿宽度参数更有判别力；当前不足以晋级独立创新标题，也不足以杀掉家族。
