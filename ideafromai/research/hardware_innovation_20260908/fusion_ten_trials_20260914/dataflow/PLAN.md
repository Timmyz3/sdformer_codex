# 三条独立数据流接口（十项批次中的三项）

旧r8_consumer_fusion/生产树只读。固定Q1/Q2与FP32→J20→I24函数，复用现有完整金数据；不新量化、训练或EDA。

D1完整cachedOS p直接进入真实消费者。保留旧全物化作来源；主强控同一输出holding与forward权限，可在STORE同时将acc送消费者；候选取消p_mem写入。两侧同输出保持/背压与并行权限。预计强控与候选可能周期相同，但有真实事务差，不能用弱barrier分母夸大。

D2相邻横向tile复用源buffer的两列。两侧同1536×10bit源容量/1词读口和最强dual-P核；环形列相位将旧右两列解释为新左两列，候选只装右两列。真实配置地址和core读取地址均在SV映射；行首/首tile完整装入，跨行不复用。状态/接口与所有省下的写入/请求都明确计。

D3固定有限双tile交错。两个context有同权private源/z/psum/Q2缓存，单一外置8×32加法链和8×19×13乘法、单W阵列及source/W/z/psum逻辑口分类仲裁。core发明确req/grant；BASE_MAC/ZADD请求z+ALU必须一起grant。seq也有双tile容量且可在tile0计算完后启动tile1、同时drain0；交错在tile0 Q1完成后启动tile1，利用Q2/Q1空槽。共同一个FP消费者严格原序，奇数末tile完整排空。预取小端点不计本项。

每项实际小块/边界/背压/重启，随后64连续tile；值得推进则完整19200tile。多种stall、同FIFO深度、参数扫描不另计项数。core和consumer重叠周期不相加，完整装入/静态配置/最终退休实计。

最终强控补齐：D3增加mode2两context从头同时ready，完全相同RR与共同资源；mode1只对mode2评估阶段错位净差分，mode2对seq的普通多context收益单列。它是同一D3试验的公平分母，不另计新融合。
