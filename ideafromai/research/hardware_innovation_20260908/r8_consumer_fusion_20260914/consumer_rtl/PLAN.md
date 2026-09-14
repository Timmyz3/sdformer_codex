# R8 到真实 r1 I24 的共同消费者

本目录独占新实现，旧 integer_factor 文件只复制不修改。已读实际 `MS_ResBlock.forward`、`ParentNetwork` 固定 BN 安装、`LiteralForward.source_forward`。真实顺序为原生 gate→r0 conv2→固定 BN2→加完整连续 identity→I24 RNE/sat→r1 As。旧整数 R8 只完成 `z=Q1g,p=Q2z`，其输出尺度和后继消费者尚未收费。

B 是完成 raw p 后仍需全宽连续 identity、固定仿射以及不可越过的 I24 边界。强 A 保留同函数 expanded21 mode6 与完整 R8 V 驻留 mode7，两者共同拥有旧源、所有系数、latent、psum、八条32bit加法链与八个16×13乘法器。普通 BN 折叠、流水供数及完成口融合属于借入 A；本目录不声明新代数或 X。

本轮固定新部署合同：`J=sat32(RNE(identity*2^20))`，`a=RNE(output_scale*BN_gain*2^40)` signed32，`b=RNE(BN_offset*2^20)` signed32；`wide=p*a+(b+J)*2^20` signed64；`I24=sat24(RNE(wide/2^26))`。常量及 identity 范围由真实导出确认；这不是旧 FP32 BN/add 逐位等价，数据分支必须评价新 I24 出口。旧 z/p 无中间 RNE，原 I24 完成位置保持；仅本合同内要求 mode6/7 逐位相同。

共同消费者使用 raw p8×32 和 IEEE FP32 identity8×32 两个各一项的真实 ready/valid 接口；a/b 共享一个256bit常量读口，分两拍读；八个32×32乘法器、八条64bit加法链、RNE/sat逻辑列入双方资源，不把旧窄乘口免费扩宽。p 与 identity 到齐后先付一拍 IEEE-FP32→J20 转换（guard/remainder+parity RNE、sat），然后依次读 a、读 b、MUL、加 bias、加 identity、ROUND、SEND，各拍及阻塞实计。a/b跨同N8组40个P/T消费者驻留，常量每tile只读24向量。最终480个I24向量全部接受后才允许tile退休。

SV wrapper按tile序列加载真实4×4源、原点；W/Q1/Q2/k_live/mask/a/b一次驻留。source与identity来自独立外部接口；raw p由同实例整数叶产生。结果带tile身份，跨tile循环及最后完成均在SV，不由TB循环独立实例累加充当stream。先八真实tile和合成极值/舍入/边界/背压/重启；随后按实际速度固定连续64tile，合理时完整19200tile。所有常量配置、新源、identity、内部状态和stall分项报告；不做EDA、生产改动或额外参数扫描。

最终强控制补齐mode9（完整最终z支持bitmap），mode11接同一后端测试局部原生源窗口。mode11对mode7是组合收益，对mode9才是窗口单项；均归更强A。FP32入口主完整帧只运行9/11，6/7保留八块溯源；旧J20接口结果位于q20_input_snapshot，不能代替最终入口费用。
