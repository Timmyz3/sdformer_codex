# H8×T10 固定组合求值/寄存边界

B是父post_psn_subset的真实时分核：同80路48bit ALU，full285126/cert428572；省位数被BOUND_POS/NEG/LOWER/UPPER四拍吞掉。这是具体控制/数据通路布局，不是面积定律。参考只读Claude t10_rtl/cert_gate_bitl.sv的同拍nv/界判定和tail递推，但不继承其外供e/plane、initial离线LUT或逐H叶周期，也不照抄负gain翻转>=的tie语义。

技术上可一拍的原因：两组5bit LUT是组合寄存器mux，先得到lo/hi子集和，再在同一组合路径做nv=(v<<1)+lo+hi，只有最后nv寄存一次。它需要每lane两个48bit加法位置（80×2=160），不是父80ALU时间复用的免费合并。符号头用同两级add/sub做0−lo−hi，在GROUP拍寄存v；同拍初始化共同m和tail。固定这一布局，不扫H/mux/参数。

资源清单：160个48bit prefix add/sub位置（其中10个第一阶段分时构建LUT/P/N）；80×2=160个48bit下界/上界加法位置；20个48bit tail减法位置，共按RTL显式340个48bit加/减位置，无乘法器。20个tail单元在GROUP算(P<<m)−P/N同式，在每个非末plane算(tail−P)>>>1，差必偶数，移位精确。H8共同e/m所以P/N tail按t共享20值，不复制8份。两路80个signed compare/equality路径，同拍更新locked；full仅m=0采最终比较，不等判界状态。组合关键路是LUTmux→两次48add→变长移位→一次界add→signed比较→80门归约，未测时序/面积，不称同80ALU/同面积/Fmax/ASIC速度。

每组调度GROUP(sign/tail init)→每plane一个PLANE沿→OUTPUT握手，full持续到m=0，cert全80门锁定才提前退。下界/上界同时观察U包络；正gain U>=tau、负gain U<=tau、constant优先，gate-only cert的out_u无完整U承诺。保留零e哑plane规则，完整整数函数无RNE。

父Y90KiB真实存储、原单2304bit Y写/读口、T10 ybuf2880B、真实运行时e、A/tau/flags单128bit请求响应、LUT1280B单份20bank×8个读mux、cold503/ warm490词与64行LUT冷构/P-N常量初始化均保留。dot_hold480B和lower_hit/lower_gate20B可删除；额外组合单位明确付费。参数/表初始化的完整model驻留合同保留，full无逐plane判界等待。

只引用父37case binary：32real和原5diag。full/cert×ready/BP×cold/warm，唯一初始reset后连续换源；逐真实Y/指数/表/fullU/cert上下界/最后gate。与父逐病例plane/early计数必须相同。报告Claude仅1+sop/plane的叶边界与本核GROUP+PLANE/真实读/输出/配置分摊，不跨函数/H分组直接比其旧数字。正结果也仅是面积时序待证候选；完整BitL未迁移。

收口补充：按root独立审阅新增一个直接Y24全范围混合诊断，独立int64金标准，另跑8命令。包含最小负数、最大正数、零、±1、跨t翻转、正负gain/constant/tie；单独记录，32real主表不变，RTL无需修改。实际结果与资源边界以README/SUMMARY为准。
