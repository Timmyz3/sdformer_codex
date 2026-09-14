从 `fixtures/*/reference.npz` 原始二值源与固定Q1/Q2重新计算，没有拿RTL导出的latent或计数器作为工作量输入。`verify.py` 的11088项检查全部通过，含零/全一/越界毒值功能输入、两种背压、无复位二次命令。

对每tile定义：K=Q1实际非零列数；Q=存在任意P/T活动的K列数；A=原始P4×T10×K活动数；U=两对空间P在同T的union数；M=实际执行的8lane Q2 MAC次数；R=非零code原型256bit读取次数；E=encoder状态数。

完整producer周期：

`5437 + K + 2Q + 3U + M + E + R + source_stalls + weight_stalls + output_stalls`。

这里VLOAD仍访问96行控制，即使v_live/rank_live关闭了物理读取；5437已含该固定控制、z清零、末态、480字psum写读/输出。消费者给producer的背压计入output_stalls，不可再把两个模块周期相加。

encoder每tile在所有40向量均非零时：exact0、group320、rank160、K4+1残差1160、zero+1残差200、temporal group296、temporal rank152。对于实际Z0个全零完成向量，前四非时域模式共同EREAD→EWRITE两拍旁路，分别再减6Z0/2Z0/27Z0/3Z0；时间模式不旁路当前0。本8tile总Z0=113。signed13 Δ编译选项利用已付EDIFF差值/共同EWRITE选择，encoder拍数不变，显式增加range/nnz/use_delta控制。各模式完整Q1活动相同；没有先算完latent再开始计时。

物理账：source=`96*图内4×4像素数`；Q2权重读取V=`count(v_live AND rank_live)`；全部权重读取`Q+V+R`；z向量读`U+40+(mode!=exact?40:0)`；z写`20+U+(mode!=exact?40:0)`；z标量读=M；psum读/写均480。

完整consumer周期=`3385 + join_wait + external_output_stalls`。每tile raw、identity、转换、乘法、RNE、输出均480字，加法960字、系数24字。顶层总拍=consumer周期+1（退休状态），producer通常比consumer提前6拍结束。`total_cycles`按握手退休实测，`cycles`是producer总拍；主比较用前者。

模式7与6、模式8与5属于相同有损函数的执行选项，gold逐值相同：t0强制全算；之后Δ可放signed13且变化rank少于当前非零rank则在prev p上加Q2Δ，否则重算。prev latent保留的是近似当前向量；Δ写入z后只用于本次Q2，不污染下一时刻的比较参考。整vector未变时两侧都直接转发prev p。每t的真实identity与BN折叠系数仍进入消费者。

固定Q1的逐rank最紧全binary源差界为Σ|Q1_rk|=[667,480,501,447,589,597,472,662]。rank迟滞每坐标保留的是某个早T真实值，故差界不是两个独立±2592相减；codebook median也处于同坐标范围。signed13溢出在此冻结合法域不可达，保留guard但没有为不可达域追加实验。`integer_bounds.json`保存min/max与不变量。SV共同signed16×14乘法器是保守规格，不宣称实际必须14bit，未据此推PPA。
