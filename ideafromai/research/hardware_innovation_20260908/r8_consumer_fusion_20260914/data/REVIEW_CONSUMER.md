# 独立审阅：IEEE identity 至 I24 与连续退休

只读最终 `consumer_rtl/i24_consumer.sv`、`consumer_stream.sv`、resource_contract 与最终 FP32 入口收据，不重跑 RTL。未发现冻结有限FP32输入域下的数值或计费阻断。独立 [review_consumer_counts.py](review_consumer_counts.py) 核对5,332项周期/工作量/地址守恒，全部吻合；[结果](review_consumer_counts.json) 区分256个小fixture命令、8个连续64tile命令及2个完整19200tile命令。三个检查点rawp/J/I24对应每一类均分别检查983,040、1,966,080、147,456,000个值。旧Q20入口收据不被引用为新FP32主结果。

## 数值链

IEEE32正常数满足`identity*2^20 = (1.f)*2^(e−107)`，用整数24位significand写成`M*2^(e−130)`。RTL对130≤e<138左移；106≤e<130右移并以余数/奇偶位实现RNE；e<106的有限值绝对量小于半LSB，含subnormal与±0，得到0。e≥138对应|identity|≥2048，按signed32 Q20饱和；负2048恰是可表示下界，转换饱和计数排除了这一精确端点。负数先对幅值ties-even再取负符合对称RNE。NaN/Inf明确置error，本次真实接口只声明有限数。

随后signed32 p×signed32 a产生signed64；先加已符号扩展的b<<20，再加符号扩展J<<20，避免`b+J`在32位溢出。冻结参数和任意J32的wide绝对界3,688,869,010,604,032小于signed64。最终算术floor右移26，非负低26位余数比较半LSB，奇偶由商低位判定；负ties也正确，最后饱和到signed24并符号扩展传输。宽乘、两次宽加、舍入、真实FP转换分别占状态；没有把原16×13乘法器冒充32×32。

实际数据已另用纯NumPy int64逐值重算全73,728,000个I24；实际IEEE FP32全帧也独立验证全73,728,000个J。见本目录 [full_fixture_contract.json](full_fixture_contract.json) 与 [identity_fp32_contract.json](identity_fp32_contract.json)。这是新部署函数精确，不是旧float32 BN/add或全网bittrue。

## 握手、端口、生命周期

GET有p/J各一个向量缓冲和独立have位，任一路先到均保持；只在其空时请求/接受。raw_addr核当前row，identity由当前tile/row请求取值，无隐含整帧identity存储。FP32位和J复用同8×32寄存器，转换另付一拍。每40个P/T成员按输出O8组复用a/b，组首分别两拍读取唯一256bit系数口；每tile24系数词。

SEND被阻塞时row与result保持。消费者每480个8lane值全部接受后才FINISH。wrapper要求producer done及consumer done，同时accepted==480后才退休并写下一tile的source；因此不会因rawp已经发完而提前覆盖下一tile。原生地址由C×76800+Y×320+X在RTL生成，origin=2×tile坐标−1，越界写零不发外部请求。cold静态参数12504词只加载一次；warm命令重新加载真实source/origin/identity。当前warm同mode已测，跨mode重启未测。

## 完整计费

每tile消费者必须完成480次GET、CONVERT、MUL、ADD_BIAS、ADD_IDENTITY、ROUND、SEND，加24次系数读和1次FINISH：

```
consumer_cycles = 3385*tiles + join_wait + output_stalls
job_cycles = consumer_cycles + static_words + parameter_stalls
           + 1536*tiles + origin_words + source_load_stalls + 2*tiles + 1
```

producer和consumer并行区间重叠，不能将两者cycles相加。所有无背压记录producer output stall为2419×tiles，consumer比producer晚6×tiles结束；这两项也独立逐记录核对。源加载1536词、identity/raw/I24各480词、转换/MUL/ROUND各480次、宽ADD960次均有真实计数。

最终完整19200tile每臂73,728,000个rawp/J/I24全相等；mode9总511,638,152拍，mode11总493,321,352拍。两者都从实际FP32 identity进入，包含9,216,000次转换；不能引用旧Q20输入总拍作为这个出口。4读本地mux、8个IEEE转换器、8个32×32乘法器和8条64位加法链等共同资源由resource_contract单列，尚无综合/时序/能耗结果。完整帧背压和warm未复跑，只有64tile对应覆盖；此限制保留。
