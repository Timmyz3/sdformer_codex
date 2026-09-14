# 下次 cachedOS → 同一消费者的接入合同（只读审阅，不新增 RTL）

已读 `../partition_rtl/partition.sv` 的接口、配置、STORE/DRAIN和报告。mode12 cachedOS已是真正更强普通A：完整Q2 R8块128B驻留、位置rank支持、寄存器psum，八块98,895核心拍；mode13有限partition相对它负3.91%。这两个值仍是旧真实捕获的raw叶、19×13共同乘口，不能直接替换当前A800/FP消费者整帧分母。新packed接口由另一作者独占，本文不写第二份同功能RTL。

下一接入可以完整复用当前 `i24_consumer.sv`，无需更改冻结a/b/J/I24函数或扩大后端。partition现有输出恰是 `row=og*40+fp`（fp=P*10+T），顺序0…479，8×signed32 p、valid/ready/done，与当前消费者契合。其STORE写完整p_mem，DRAIN_READ后DRAIN_SEND遇ready才推进，保持信号，故先保留这条完整状态寿命作为可测强控制。

需要实际改动的仅是新wrapper的生产者实例和配置状态、诊断绑定：partition cfg_addr为11bit，合法0 source、3 origin、4 Q1、5 Q2、6 k_live；没有expanded和blockmask阵列。Q2的v_live在配置拍真实计算。若比较mode12/13，则共同静态Q1 864+Q2 96+k_live864+消费者a/b24=1848拍；每新tile仍1536源+1原点，identity480向量和FP转换480拍。不能让一个臂免掉静态项、另一个保留旧12504拍；若跨旧mode11比较，先定义同一联合状态/端口预算及共同配置清单。

共同后端继续保留两路一项缓冲：raw p由生产者，identity读原始IEEE FP32；先到者保持，不以缓存goldJ代输入。配置a/b的24行单256bit读口、跨同N8的40消费者驻留、FP32→J20一拍、MUL/两ADD/RNE/输出完成都原样保留。最后tile退休仍须producer done和consumer最后接受；周期不能把两个重叠计数相加。

若随后试cachedOS最终acc直接转发，应只在完整R8收缩完成的STORE边界发完整p，不能把rank partial喂消费者。顺序本来已满足消费者，无需重排；但必须保留结果holding直到ready，并与具有相同一个输出缓冲和转发权限的普通控制比较。删除p_mem store/drain的收益和由消费者阻塞造成的生产者停顿都要实际计入，不能只减480写+480读。该优化是普通最终值转发A，不能仅凭省物化改成X。

固定复用本轮data的source、FP32 identity、raw_p/J/I24 gold，再测新同模块至少mode12强控制与挑战者，先8块/边界/背压/重启后64tile及合理整帧。不需要新量化或新AEE来掩盖调度变化；但若代码改变Q1/Q2、重定RNE或接受不同整数溢出，则须另立新函数和质量，不能继承当前825。新宽bank读取/packed位格式必须由该作者给共同物理权限，当前单13bit scalar接口不能假定免费变双值。
