# 前沿请求合并与部分时间完成

A：普通ROBDD及按需特征读取。B：当前共同source PSN选择一个physical channel后计算全T10，用10个16×24mult/10个48bitacc。X仅改变provider义务：十个非因果时间消费者各请求当前(t,channel)，取D-only entropy rank最前最多4个不同channel，合并同channel读取；每个t只计算自己请求的channel，不承诺剩余U。

CPU首探针已完成64real＋3diag：code完整3359轮→3048批，有效MAC335900→267920，但源词6718→8282；class3353→3039批、267460有效MAC、8238源词。位点需求降低不等于满10lane向量批数同比降低，也会出现不同t跨批重复读相同channel。仍完成RTL，若负则定位此供数/调度失配，不扫批宽。

同核五臂：static64＋next-X PF、原one-channel code/class＋child PF、新frontier4 code/class＋同child PF。全部给相同128B X holding（旧32B＋96B），128B图cache，8bank×128bit共用请求/响应，每bank至多1pending；参数同按用途加载。固定32bit节点、固定entropy D-only rank，代码/类表各沿原函数，不能将普通前沿并行算作class独有收益。

X holding为4×2个128bit实际字，按选中channel真实读入，128B寄存器且每lane24bit 4:1选择mux显账；保留T10单一source_t，每拍lane t使用A[t,s]和自己slot的X[channel,s]。仅10个mult和10个acc，不因4channel复制。active mask沿两拍product流水稳定，完成后只更新active nodes；未算U不对外假发完整门。最多4channel意味着每批最多8个物理X词，相同bank冲突必须串行。plain臂不保留跨批X内容作cache，重读全计；后续授权的resident臂保留4槽，见末段。

复制父core到此目录改独立module，父SV/fixtures只读。真实producer monitor逐valid(t,c)核完整10项整数dot、threshold gate；最终6×T10 code/class核argmin/canonical。引用父Python3.12已再生source.bin，不喂gate。BP检查req持有/onepending、图与源物理traffic、output稳定；普通臂须与旧全部同calendar周期复现。无新训练/AEE/EDA，class仍绑定原已标注modified response-W，不冒用原FP函数质量。

后续授权：保留plain（它实测已正，不叫失败），加resident参数的固定4槽空槽优先/FIFO，不改变容量/图/训练。普通static/onechannel也启用同槽权利；回放扩至旧W′/新W″两套1027病例，并将31未参与pair选择的训练帧前置。最后报告明确class类型、普通code控制、BP参数bank日历差与未接joined/无reset项。

独立审阅更正实际实现边界：最终pf_relevant保留旧min-channel child>=16规则，未覆盖所有active；原拟替换文本没有匹配完整原句。所有已测plain/resident结果均属此规则，SV已冻结。多前沿PF供数仍未闭，不能称完整迁移；详README/REVIEW。
