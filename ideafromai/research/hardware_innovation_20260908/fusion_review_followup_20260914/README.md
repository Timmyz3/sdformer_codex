本轮先复审上一批11项及组合结果，再继续真实RTL融合。当前本机模型配置及可读线程记录均为`gpt-6-astra/xhigh`，未发现Luna；这不是后端路由证明。两名新代理显式使用同一模型/档位，父代理独立检查算法质量及编写Kronecker RTL。[模型核查](MODEL_REVIEW.md)

|交付|状态/实际证据|
|---|---|
|上一批数据流及组合审阅|[review_dataflow](review_dataflow.md)：110条新代表命令，所有64块整数计数复现；补正D2资源表/static mask描述|
|上一批四个分解审阅|[review_decompositions](review_decompositions.md)：新编译代表与反例；发现字典冷配置分母、anchor启用税及real_6跨叶输入差异|
|上一批算法/消费者审阅|[review_quality](review_quality.md)：108条新代表命令，4臂825逐帧指标复算；未把不同AEE成本点称同质量优势|
|新R2分组字典RTL|[pair_dictionary](pair_dictionary/REPORT.md)：128命令，核心慢29.539%，停止固定布局|
|新精确四P低位/高位义务RTL|[modular_packing](modular_packing/README.md)：204命令，64块比三P10省4.1929%，保留执行接口|
|新Kronecker/完整输出组回退RTL及AEE|[kron_escape](kron_escape/REPORT.md)：128命令，小块冷省2.231%；十帧AEE1.899失败，停止当前拟合端点；修复模式切换漏载|
|精确打包与两context普通RR融合|[rr_modular](rr_modular/README.md)：176命令，64块814321→787603（省3.281%）；[独立逐拍审阅](review_rr_modular.md)通过，仍缺RR+borrow强控制|
|本轮最近邻|[literature_followup](literature_followup.md)：ULPPACK、ISCAS2025 overflow-psum、A2Q、PQS、GKPD、Hybrid KPRNN；不把借入机制当X|

原11项逐项裁决见[review_scoreboard](review_scoreboard.md)。所有新代码在本隔离目录；只更正旧D1/D2两个资源JSON的文字，旧RTL/数值不改。生产nts07、论文贡献句、docs359/H81没有修改。没有新的ASIC/PPA/整网FPS或“强接收”证据。

本轮的强控制纠偏很具体：冷配置按每臂实际需要付费；局部候选的anchor保存费不能漏进选择评分；group/rank精度不同就不得声称同AEE更优；强RR已经胜过阶段错位，新增算术必须在RR下继续比较。性能失败只停已试布局，未试接口在各报告中列明。

本轮四个新增实现点的主回归合计636命令（128+204+128+176），独立复审/反例复跑另列，未把重复构建计作新idea。保留modular作为执行接口，优先补RR+borrow控制；R2固定布局和当前Kronecker拟合端点停止。来源与功能边界不满足时，不给它们贴“强接收”标签。
