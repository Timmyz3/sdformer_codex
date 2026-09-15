# 下一条有界接口建议：真实源生产摘要准入

本轮只建议，不开新RTL或长跑。旧 [NEXT②](../../consumer_transfer_20260914/NEXT.md) 的Gustav producer bank-summary/support admission仍是明确未闭接口。原NRV∩W双指针、共享W位图、dense旁路和lazy首次NRV已实接GP/PSN；不要再把有序地址交集列成缺失，也不要将旧私人索引的负结果否定整个家族。

先完成A：保留旧eager3、lazy4相同四个source bank、两tile各两W8口/单在途、双NR4、GP和全部T10 PSN，在其原图边增加真实源bank写入/打包/完成与复用握手。端点从首个实际源写入到最后T10门接受；静态program/tau配置单列相同。源未完成前的读取必须遵守同一bank占用规则，不让TB传入NRV或“bank有活动”。这不直接替换当前r0.conv2→I24：Gustav现有码字/GP→PSN是另一图边，必须各自真实gold。

再做一次有界适配X：在已接受的真实生产写入上维护每bank的seen-code摘要与generation/complete，在bank完成后依据同代decode的非空支持决定metadata准入。可从eager继承及时W支持，避免lazy在首个NRV后才补扫压缩rank；全空则不启用metadata。支持是否早于source读取、补扫消失与否必须由实际端口时间线证明，不假设所有摘要读取/生成都可重叠隐藏。

code3不能用packedword!=0判断NRV。现 `gp_slice_lazy.sv` 的真实谓词是对四个code3分别查 `decode_q[code*7+:7]` 并判OR非零。一个8bit seen-code集合可先记录出现过的码，再与8bit live_decode交集；代次确保新bank/W/program不继承旧摘要。若生产者原本输出四P的12bit码行，摘要更新可绑定这个实际accepted写入，并由真实packing器形成原64bitbank字；若生产者只有64bit packedword，必须处理3bit码跨word边界，计相应holding/解码拍，不能免费改成12bit行接口。原本更前级的量化器尚无闭合证据，补写入wrapper只能称推进了生产接口，不能称完整量化生产链已经实现。

最小probe使用现有真实dense/2:4/C16源/W/program/tau，并加入全空、首非空在最后行、同bank换代、两tile稀密不同、decode中code0非空或非零code映射空的定向例。eager3/lazy4/X三臂都从同首写入计费，ready/BP及连续换source/W/program，不扫nnz阈值。新增摘要/代次/packing状态、更新与读端口、源生产停顿逐项记录；比较metadata字节、rank追赶、末门服务以及包/消费者恒等式。若X不正，定位究竟是producer已串行占尽窗口、摘要完成过晚还是W端口不足，再只改相应一个接口，不把负结果归为Gustav全家无效。

与本轮m7剩余费用的关系应谨慎区分。m7已经在K16内只预取一次三plane、跳过全空bitmap block，并按同T行复用Z hold；这部分不能再次计作新共享。仍存在：

- 每tile外部全部1536个gate字加载，相邻2×2tile的4×4 halo常重叠；当前不同context没有跨tile源缓存/生产摘要。随后固定864个K的gather与bitmap构建仍执行，即使某些源区域全空。
- 相同静态plane跨K16重用窗口/跨tile重新经唯一W服务读取；held m7仍有9711次plane预取和17984次onehot原Q1读取。Q2每tile按output group重新填cache，held共6144次Q2 W读取。这些是本地共享W服务请求，不等于新外部DRAM加载；不能凭值相同免费广播给两个owner。
- 不同P/T出现相同16bit非零bitmap时，m7仍分别pop并更新各自Z字段；它没有运行时相同mask贡献缓存。其支持/输出地址不同，若要复用需保存8lane贡献、判key并仍付各目的加写；摘要准入本身不会自动消除它。

因此建议先在Gustav原真实GP/PSN图边完整借入生产A并检验一次摘要X；若通过，再讨论相同摘要是否能在R8 source loader给出有费bank跳过。当前没有把Gustav端点收益或这些重复机会换算为R8周期收益。
