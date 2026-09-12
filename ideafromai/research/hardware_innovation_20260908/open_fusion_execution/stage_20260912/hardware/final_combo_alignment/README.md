# 最终GPU组合与局部硬件服务身份对齐

两轴都已对齐，无需重新仿真。最终真实GPU捕获中的ordinary `original_ordered24 + onepass`、lifting `activation_whitened24 + onepass`，对应本阶段已实际执行的R24局部服务行。检查发生在同一预选帧 `zurich_city_09_a_0001`、原corner/interior窗口，未换参数或样本。

|最终GPU组合|corner已有服务槽|interior已有服务槽|对应旧执行记录|
|---|---:|---:|---|
|ordinary / original_ordered24|2,062,654|2,755,532|rank_fusion.json 同名mode|
|lifting_raw / activation_whitened24|1,898,584|2,531,584|rank_fusion.json 同名mode|

逐字段确认：ordinary的28项部署常量、34项live参数、45项原学生归档字段；lifting的44项部署常量、34项live参数、60项原学生归档字段均与硬件输入来源逐位一致。包含真实U24/V96、指数16/15、PED bias、source/consumer阈值与源结构参数。源ROM程序、frame/窗口/布局/前向顺序也相同。参数比较使用最终 `deployed_constants.npz`，没有误把保留的R32学生归档当R24部署参数。

四窗的原I24、sn1/sn2门、preview Z/raw/BN1、updated I24和projection gate均与原硬件捕获逐位相同。最终GPU实际连续PED与此前执行过的独立R24整数gold：每窗15,360值、四窗61,440值全部一致，打包signed24载荷也完全相同。gold在NumPy中用int64容器、GPU归档用int32容器，这只是存储类型差异，不是24bit线上值变化。因此保留原同Machine服务、端口和状态记录，不重复相同仿真。

旧FP边界仍保留：硬件preview raw/BN1与CUDA存在旧模型已记录的小差异，但旧模型输出的sn2门精确；本次GPU导出中的相应preview数组又与旧GPU捕获逐位相同，故没有引入新的局部FP差异。不能据此把完整GPU与硬件所有浮点中间值说成逐位等价。

onepass位于本局部计量终点之外：updated I24分成projection gate→native Conv→全域onepass BN支路，以及连续PED U24/V96支路，最后相加。现有局部CPU时间线结束于projection gate与连续PED输出，未包括native Conv、全192,000位置/通道BN统计及最终join。

|学生/窗口|onepass归一化相对旧CUDA变动值|最大绝对差|
|---|---:|---:|
|ordinary/corner|9,492|9.5367e-7|
|ordinary/interior|9,709|3.8147e-6|
|lifting/corner|10,855|9.5367e-7|
|lifting/interior|10,604|2.8610e-6|

以GPU导出中已观察到的完整BN统计检查窗口：`MUL(raw_conv,scale)`后独立`ADD(bias)`与实际onepass窗口输出逐位相同；再与实际signed24/f14连续PED做FP32 ADD，也与最终组合输出逐位相同。这只是下游接口算术核对，不是免费BN统计、不是新硬件执行，也不将之前BN大域表加到上述局部服务槽。

精度链接：两条最终组合valid825均已实际完成，ordinary AEE **1.211716274329**，lifting AEE **1.232391049037**，每臂48,152,523有效像素。分别来自 `algorithm/combinations/{ordinary,lifting_raw}/valid825/combo_summary.json`，没有继承原R32结果。`alignment.json`已链接两轴完整825与局部身份；`check.py`只刷新已有结果连接，不触发仿真或GPU。

文件：`check.py`逐字段/逐值检查；`alignment.json`包含全部比较、原硬件检查和明确服务行链接。无新训练、RTL、EDA、GPU、哈希或主稿修改。
