# 唯一有界适配：active前沿child预取

B：冻结frontier_source虽已让最多4channel并行生产，但child预取仍只沿全前沿最小rank的channel；resident-first时该channel甚至不在本批生产集合。该接口缺口已审阅确认，不能把旧版称为完整多前沿预取迁移。

X：一个运行时start_active_prefetch开关。0完全保留旧规则；1只对当前batch实际active且node_var==batch_channel[lane_slot[t]]的非终止消费者，预取其两个非终止child record。不是把full provider的十个active算术lane一律当十个合法图请求；single-channel控制仍限定正确当前channel。所有code/class、旧onechannel/plain/resident同权限，static next-X规则保持实际等价。

资源固定10mult/10acc、X128B四槽、图cache128B、8bank×128bit每bank1pending、同20bit pf_done；不增加容量、预测或替换策略。额外仅模式bit及active/slot匹配控制；潜在cache污染、两个分支的无用读、bank冲突和DRAIN停留实际计费。基准0必须逐字段复现父冻结CSV。

复制父SV/TB至此目录，不改父。先小，再旧W′/新W″相同expanded1027，各六臂static/onecode/oneclass/plaincode/residentcode/residentclass×ready/BP×旧/新PF。新PF效果以相同exact-code函数的0/1比较为主，class是相同权利消融；无需新CPU工作量猜周期，因为逻辑source需求/四槽计划未变，先用父独立CPU通过的counter比对两PF，再全物理请求/实际U/gate/分类验证。仅训练缓存32帧固定位置；无AEE/训练/EDA。
