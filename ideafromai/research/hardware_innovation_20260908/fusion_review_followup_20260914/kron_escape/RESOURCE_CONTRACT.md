两臂同一SV，完整C96、K864、R8、N96、P4、T10，所有raw/真实FP32identity/J20/I24输出执行。mode0执行显式展开的同一个hybrid Wh；mode1执行B收缩、A广播、两组精确Wh逃逸。原Wh近似了旧Q2，所以跨旧模型比较属于有损；两新臂之间完全相同整数函数。

共同producer只有8条32bit carry链和8个19×13 signed multiplier；Q1双P13与Q2共用八ALU。Q2 expanded Wh最大36855，需要19bit寄存/存储接口，不能偷截成旧16bit；候选的B6在19bit乘数侧，z13在13bit侧；第二级S19在19bit侧、A13在13bit侧。没有第二套乘法器。shared qblock为8×8×19，两臂均缓存完整输出组的有效rank系数。

共同局部存储：z8×20×26=520B，p8×480×32=15360B；Q1 8×864×3=2592B、Wh 8×96×19=1824B；source1536×10bit。新增共同S为8×160×19=3040B，A12×4×13=78B，B8×2×6=12B、escape12bit、Slive40×4bit，以及Ahold52bit。S每次8lane向量只有一个读或写地址；构建结束后再消费，不与同bank写并发。每个P/T的四对都付SINIT/SSTORE，实际非零z才付SMAC；中间S不舍入；当前常量S界16451，p界67955134。仅这些已验证常量/明确界内合成角落属于本合同。

producer总线为256bit参数配置容器，有效Q2读取152bit；源字10bit，z向量208bit，p/raw256bit，S向量152bit。两臂同源/权重背压、输出容量、消费者资源权限。A从静态表按输出组加载一拍52bit；B12B驻寄存器，SMAC从两列选一个8lane系数；它不是每次免费的外部读。完整I24后端保留相同8×32×32乘法/8×64加法、FP32转换及RNE/sat逻辑，不借其ALU。

冷mode0实际加载1848beat；冷mode1额外12A+2B+1escape，合1863beat。mode0后无reset切mode1只补15beat，由新增factor_resident跟踪；mode1后切mode0不需重载。两种配置都保留完整Wh，未把算法压缩参数数直接当实际权重容量下降。运行期S构建/写/读、A加载、escape MAC及psum读写全部列counter。

这些是行为RTL及声明的共同物理资源预算，**不等于单独裁剪后同面积，不等于宏端口映射/Fmax已经通过**。source本地响应、组合乘加/选择器延迟沿旧同资源模型；没有DDR或布线延迟、ASIC PPA和整网FPS。CPU拟合只用训练数据，配置下发计冷启动，离线拟合时间不加到每tile周期。
