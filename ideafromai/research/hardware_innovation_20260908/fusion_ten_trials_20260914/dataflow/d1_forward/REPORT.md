# D1：完整输出旁路的强控制差分

**主结果为零周期收益。** 相同原生 C96/K864/R8/N96/T10 和真实 FP32 identity→J20→I24，64 个连续 tile（128..191）冷命令：STORE+forward 控制与取消 STORE 候选均 **968,747 拍**。有背压均 1,027,877 拍。两者同一 holding、同一消费者 ready 权限，所有已测命令周期/执行计数相同，唯一事务差是候选取消每 tile 480 个256bit p_mem写；64tile取消30,720词。原完整物化/末尾drain只是来源控制，不作为主要分母。

`forward_core.sv` 的 STORE 把完整8路 acc 送入共同输出holding；mode1同时写p_mem，mode2不写。DIRECT_SEND真实等待消费者，保持数据和地址；确认后继续下一个目的。消费者和producer实际并行，但两者active周期不相加。取消全物化屏障是普通完成转发A；在已经给控制转发权限后，本候选只消除存储事务，不能称新代数/新X或能耗实测。

共同module仍声明8×480×32 p_mem，方便完整追踪和同状态预算；候选不访问它，未综合证明移除面积。源1536×10、Q1/Q2、208bit z、八路19×13乘/32位加法与八路独立I24 32×32乘/64位加法均完整收费。详见resource_contract.json。

功能收据：16fixture×3模式×2背压×2命令=192条（包括8真实、全零/全一、padding poison、正负极值、负tie及FP转换/饱和）；另64连续tile两强臂×2背压×2命令8条。raw p、实际J20和I24各检查 **2,703,360值**，全绿。第二命令无reset、参数驻留；新源/origin/identity仍装入。源/参数/identity/result请求稳定性和最终tile身份均由TB检查。

结果保留在results.json、results_64.json、benefits.csv。没有把64tile负周期结果扩写为全帧试验，也没有运行PPA/EDA。当前接口以事务节省收口，不再扩展此点。
