# normalized_certificate 增量独立审阅

结论：当前实际 RTL 的 signed49 规范化判界、GROUP 状态建立及生命周期没有发现新的功能错误。它保留原 joined_fc1 的函数与周期，仅改变组合路径和状态；借入整数不等式归一化本身不构成新算法。独立创新暂评 **3/10**，可作为已有供数/判决粒度适配家族中的物理实现子项保留。

本次实际读 [normalized_bound.sv](../normalized_certificate/normalized_bound.sv)、[normalized_fc1.sv](../normalized_certificate/normalized_fc1.sv)、[implement.py](../normalized_certificate/implement.py)、两份 TB 的数值观察、汇总脚本及已有记录。没有重跑 RTL/EDA；本目录两点单 lane 映射不等于该完整核映射。

数值与生命周期

- `FPNINIT` 把 pos/neg 初始化 −1，原十列正/负各十拍继续累加，故内容准确为 P−1/N−1。`normalized_bound` 将 tau/nv 与这些状态分别 sign-extend 到 49bit，再加单 bit carry。GROUP carry 为 `!positive`，得到 tau+P/N−delta；PLANE carry 为 1，得到 nv+P/N。没有将 tau48 端点相加截回 48bit。
- `q>>>m` 为 signed49 算术右移。lower `>` 与 upper `<=` 对正 gain 的原 `L>=tau/H<tau`、负 gain 的原 `L>tau/H<=tau` 都成立；m=0 不需要例外。constant 在 FGROUP 先锁，后续 `!locked` 保护。TB 的 prefix 来自独立真实 Y 的 floor-shift 点积，而非从 DUT v 回读拼参考；实际 norm/q/shift 逐值有检查。
- q 没有 reset，但每个 FPLANE 都由同组 FGROUP 前一拍写 q，tau、P/N、hg 在此之前稳定；HG 切换只在 FSTORE，下一组先 FGROUP。观察组合网在其他状态可未定义，消费没有在这些状态采样它。warm 的旧 host 同模型承诺仍存在，config_valid/hblock 不是模型内容认证。36 个跨模式记录覆盖首次 native→full 构表与随后 cert，未发现新陈旧 q 通路。
- signed24 极值与 tau48 端点已做 16 个独立 direct-Y 诊断，480 个 q 项确实超 signed48。它们使用同一 normalized_bound，但不能说这些极端 Y 已由实际 FC1 捕获产生。其边界与真实 480 条 joined 执行记录正确分列。

同费用与资源

真实 FC1 写者、唯一 92160B Y、共享读口、397 词共同冷配置、120B H96 gatepack 和 native 路径未被本 diff 替换。新 q 共 160×49bit=980B，原 20×48bit tail 共120B删除，净增 **860B**；P/N 容量仍120B。原176个48bit prefix/共同 ALU位置保留，界位置变160个49bit，tail减法20个删除。新的160个49bit参数右移器也真实存在。336个混合位宽加法位置不能直接换成面积下降，native96乘法/96比较器仍在 union 中。

[NORMALIZED_RECORDS_REVIEW.json](NORMALIZED_RECORDS_REVIEW.json) 是本次只读独立重聚合：444主＋36生命周期的 **480条** 记录长度、逐状态周期和所有旧字段均相等，仅显式替换的旧bound/tail仪表字段除外；14,745,600个主/生命周期gate检查不变。496条含独立Y诊断的总量由原结果汇总为15,237,120个gate。未将84条重复smoke再算进最终总量。

32个真实 P32/H96 冷 ready 的 native/full/cert 仍为184408/227219/176448，冷BP为212997/255588/204698；cert相对native仍只省4.3165%/3.8963%。整命令频率 break-even 是 `f_cert/f_native > 0.9568348/0.9610370`，不能沿用独立叶82.3%，也不能拿本目录单lane的2.49370ns代入整个链。

新增与边界

借入 A 是分布式算术子集表、逐plane整数前缀以及余量包络证书；本次新增执行改动是把可变移位从 prefix 后移到注册阈值并行支路，实际生成 q、复用 GROUP/PLANE 界加法并付状态。这是有用的实现适配，当前没有独立周期收益，也没有支持投稿级新颖性的证据。

当前最有价值的后续证据是**把冻结全核相同物理边界下的实际关键路径与上述约4%的整链降频余量对上**；单lane结果只说明局部拓扑可映射，尚未覆盖真实 tau/hgroup 扇出、80门退休归约、共享存储与 FC1 路径。本次没有据此启动额外综合或改变候选。
