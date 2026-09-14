# 双位置 latent 打包 → 完整 FP32 identity/I24 消费者

**同函数、同资源的 mode15 相对 mode14，实际完整帧总服务周期减少3.563865%。** 每臂单go连续19200tile，全部原生K864、T10、C96/N96及真实IEEE FP32 identity都执行；raw p、J20和I24每臂各73728000个值与独立金值一致。该结果来自真实状态读/更新/写事务合并，已计固定仿射、FP32→J20转换、残差相加、I24完成、输入装入及最后消费者退休，非叶操作数推算。

[packed核心](../packed_rtl/packed_r8.sv)由另一作者实现并冻结，本目录只读实例化；[消费者](i24_consumer.sv)与上一完成版本逐字相同。旧consumer_rtl收据未改。共同资源与准确比较范围见[resource_contract](resource_contract.json)。

## 强控制与冻结函数

完整借入A为窄整数R8、原生C1源窗口、最终rank/位置支持、Q2完整R8块缓存的OS、分段SIMD加法。mode14 scalar-packed与15 dual-P共同拥有8bank×20row×26bit latent，208bit向量读/写口、单bank26bit标量读口、八条32bit加法链（ZADD时bit13断carry）、八个19×13乘法器。两者均按C1加载16个T10源字，付四路本地mask读取；第二级Q2块128B驻留，完整psum每位置只写一次，再drain。

模式14每次对一个13bit半字段执行读—加—写；15按同K、同T的P0/P1或P2/P3活动并集执行，双活时同一拍更新两个字段。source、Q1需求、第二级Q2/MAC/psum、后端精度和所有接口权限相同；位宽保证每条z部分和始终在±2592内，分段carry只是普通SIMD机制，不能单独冒称创新。

固定 `z=Q1g,p=Q2z`，无中间RNE。已逐值核对本轮data与packed原fixture的Q1/Q2完全相同；raw gold由本轮A800源重新计算，未搬旧3090输入。消费者仍为：

```
J = sat32(RNE(identity_FP32 * 2^20))          # 真实IEEE位输入，RTL一拍转换
wide = p*aQ40 + (bQ20+J)*2^20                 # signed64、MUL和两ADD各拍
I24 = sat24(RNE(wide / 2^26))                 # signed24/f14
```

a/b由本轮固定output_scale和BN2产生，不改变出口函数。a/b单256bit常量口分别读取、跨N8的40个P/T成员驻留；八个32×32宽乘、八条64位加法、八个FP转换和八个最终RNE/sat逻辑共同计入。没有把原19×13乘口免费扩大。identity非有限数置error，有限IEEE域与负ties、饱和、padding、背压已验证。

这一个资源点有更宽latent端口和cachedOS，**不能与旧mode9/11写成同面积比较**。本实验的增量分母是mode14；phase内其余普通OS改善没有算作dual-P的收益。

## 实测完整帧

|完整19200tile，cold且无外部背压|14 scalar-packed|15 dual-P|
|---|---:|---:|
|**实际总cycles**|**336199028**|**324217349**|
|consumer完成cycles，与core重叠|306648379|294666700|
|静态256bit配置词|1848|1848|
|源装入/外部有效10bit读|29491200 / 29276544|29491200 / 29276544|
|padding写/原点拍|214656 / 19200|214656 / 19200|
|核心source/本地四路gather事件|29276544 / 16550400|29276544 / 16550400|
|核心Q1+Q2向量读|8905268|8905268|
|第一阶段latent更新|23767485|19773592|
|双位置共同更新|0|3993893|
|latent向量读 / 写|24535485 / 24151485|20541592 / 20157592|
|第二阶段latent标量读 / 八lane MAC，各|53720964|53720964|
|完整psum读 / 写，各|9216000|9216000|
|真实FP identity / I24输出向量，各|9216000|9216000|
|FP转换 / 宽乘 / RNE，各|9216000|9216000|
|宽加 / 后端系数读取|18432000 / 460800|18432000 / 460800|
|core输出等待消费者拍|46444800|46444800|
|外部参数/source/identity/result阻塞|0|0|

独立从真实整帧source T10词计算同K两位置的交集popcount，得到3993893次双活。实测总差精确为 **3993893×3=11981679拍**；每次少一组latent读、加、写，不少输出或消费者。两个模式的W、第二阶段乘法、psum和消费者事务完全相同。系数向量有24/128bit不同有效宽度，词数不等价于等能耗。

两个job实际并行仿真wall149.7秒，其中14/15各149.7/144.6秒；不是按八块外推帧数，也不是硬件FPS。一次go完成整帧，参数只加载一次，没有按行启动独立实例。

## 小块、连续背压与账本

本轮16fixture×2模式×2背压×2同实例命令=128runs，三个检查点各491520值全通过；含8个当前真实块、零/一、角界padding poison、极值因子、正负乘积tie、identity tie/sat及IEEE±0/subnormal/最大有限值。真实八块14/15消费者完成110531/107036拍；第一阶段6409/5244次，双活1165，所以差3495=1165×3。与作者旧捕获1168不同，未混写来源。

固定64tile ids128…191跨行，两模式×背压×cold/warm=8jobs，三个出口各1966080值全通过。无背压cold14/15为1199181/1138971拍，warm1197333/1137123拍；只免静态1848，新source/origin/FPidentity/转换全部再执行。参数、source、identity请求与结果被阻塞时的地址/数据/last保持均检查，不能把core发完raw p当tile提前完成。

每新tile仍付1536源写、1原点、480个FP32 identity输入、480实际转换和480I24输出。wrapper须producer与consumer都done、480结果全接受才退休。状态账为：

```
consumer_cycles = 3385*tiles + join_wait + consumer_output_stalls
job_total = consumer_cycles + static_words + parameter_stalls
          + source_load_words + origin_words + source_load_stalls + 2*tiles+1
core_base = 5437*tiles + liveK + 2*activeK + 3*first_updates + second_MAC
```

core另外加其source/weight/output stalls；producer/consumer cycles相互重叠，不相加。每个合法命令全部480向量完整写入，无免费初值、TB动态支持码或latent神谕。

[verify.py](verify.py)从原生源重算184320个raw/J/I24金值，3712项事务及绝对周期通过；[verify_stream.py](verify_stream.py)另用整图word/popcount而非DUT dual counter重算coactivity，64/完整帧324项守恒与周期差通过。它们补充实际Verilator输出比对，不替代RTL。最终全部138jobs，raw/J/I24各149913600值。

## 质量与创新边界

端点数值函数及后端没有变化，可使用同一[部署825记录](../data/deployed_valid.json)：825帧、48152523有效像素、frame-mean AEE1.3276350226079938，低于历史NB01.44535253468097及同环境NB01.447936665574317。模式14/15在该冻结函数内逐位一致；这不意味着该部署函数与原FP32 BN/add或原R8任务无损，详见[质量复核](../data/quality_validation.json)。本目录未新训练、新量化或跑质量战役。

已经成立的是固定共同资源下3.56%的完整服务净增量。普通carry切分、packed格式、局部源复用或OS本身均属A；是否有可发表的新耦合X需按文献最近先验独立判断，不能把旧低秩大倍率或共享的cachedOS收益归给它。

没有宏映射、综合、Fmax、能耗或同面积证据；组合26bit scalar选择与MAC、carry分段路径尚未做时序验证。完整帧背压/warm未重跑，二者在64tile和小fixture覆盖。完整帧仅指r0整数核到r1 I24的这个实际边界，r1 As/PSN/PED及后续网络不在RTL范围。没有修改生产、旧收据或packed作者文件。

[packed核独立审阅](../data/REVIEW_PACKED.md)和[本次完整消费者附审](../data/REVIEW_CONSUMER_PACKED.md)均未发现阻断。后者只读核对2772项配置/退休/消费者守恒，并独立从整图源复算双P交集后核对72项对应关系；未重跑RTL或模型。异步scalar选择到MAC路径、共同宽口与跨核不可同频/同面积外推的限制保留。
