**真实前后链已闭合，176 个完整 H384 命令的 Y/U/gate 全部零差。给 exact code 同样的后端普通内容去重后，旧 W′ 的 early class 只有 ready +0.1103%、BP −0.1066%；根已固定的生产费用适配 W″ 得到 ready +1.6107%、BP +1.3935%。** 正号表示少用周期。这些是两训练帧各 32 个抽样位置的组件结果，W′/W″ 是不同函数，都没有本条整数链的网络 AEE。[紧凑总账](SUMMARY.json)、[输入与界](inputs.json)、[独立语义费用核验](profile.json)。

动态数据真正经过 `X24 → source_classifier 中 10×10 PSN → nearest-Hamming code/class → RTL 读 D 展开 → 320 行 g′ → support_fc1 → 完整 temporal PSN → gate`。`joined_core.sv` 实例化父级 [source_classifier.sv](../source_classifier.sv) 与旧 [support_fc1.sv](../../support_lut_execution_20260915/support_fc1.sv)，两文件均未修改。C++ 从 X 和静态参数独立计算全部答案；TB 没有向 DUT 提供原 gate、近邻码、g′、Y、U 或 final gate。[RTL](joined_core.sv)、[C++ oracle/协议检查](tb.cpp)。

源函数为 `u0[p,c,t]=Σs A0[t,s]X[p,c,s]`，`g=(u0≥tau0[t])`；A0 是 Q12、X 是 Q16、tau0 是 Q28。每个 C16 组取 Hamming 距离最小的 D 码，tie 取最小原 index；class 只改变代表码。随后 `Y=Σc D[code]W`，`U=Σs A1[t,s]Y[s]`，再按原 positive/negative/constant 字段执行 `≥tau`、`≤tau` 或常量门。A1 是既存 Q14 参数。运行时没有跨边界 RNE 或截断饱和：源任意 X24 的绝对累加界 74,524,393,472，满足 U48；W′ 的任意 spike 部分和 Y 在 [−2278,2189]、U 在 [−69,837,015,72,574,219]，满足 Y24/U48。字典响应仍是 signed INT10。源和后端 tau 是不同对象；后端复用旧 `cases.npz` 的固定整数 A/tau/sign/constant，只定义新函数，**不借旧 validation 病例的量化精度或 AEE**。[导出脚本](prepare.py)、[W″ 输入界](inputs_adapt.json)。

主表对同一 W 使用同一 D 熵序、packed32 图、128 B cache、真实预取与八 bank 服务。普通 static64 源也享有下一 X 字预取；源全部 T10 任务必须完成共同判决后才停。后端 mode4 是既存的普通响应内容去重，exact code 与 early class 都实际加载每 H96 的 3 个 class-map 配置字。每张表仅合计 train0/train1 各一次完整 H384；第二遍是无 reset 复核，不重复算入性能。

| 同函数＋相同后端 mode4 | exact code 拍 | early class 拍 | class 净少拍 | exact / class 读字节 |
|---|---:|---:|---:|---:|
| W′，ready | 112437 | 112313 | 124，**0.1103%** | 372544 / 370224 |
| W′，BP | 134201 | 134344 | −143，**−0.1066%** | 372480 / 370224 |
| W″，ready | 112437 | 110626 | 1811，**1.6107%** | 372032 / 371632 |
| W″，BP | 134190 | 132320 | 1870，**1.3935%** | 371968 / 371616 |

W″ 的 exact/class 源任务为 **3359/3266**，各任务真算同一通道完整 T10 的 100 个标量 MAC，少 9300 个标量 MAC。ready 源阶段为 72213/70402 拍，后端阶段均 40176 拍；后端系数字均 **4416/4416**、向量更新均 **5624/5624**。因此该对比没有把下游普通去重的现成收益留给 class 独占。代价也实付：X 少读 186 个 128-bit 字，图却多读 161 个字（7574→7735），总流量仅少 **400 B**；BP 对应少 352 B。BP 后端阶段相差 26 拍来自到达日历相位，不能解释成减少后端工作。W′ 同控制的系数字也均 4448，仍有“更早 class 未产生稳定净收益”的真实负例。[W′ 强控制 raw](strong_cycles.csv)、[W″ 强控制 raw](adapt_strong_cycles.csv)。

原四模式保留为执行底座/消融：m0=`static64-PF+direct`，m1=`static64-PF+LUT10`，m2=`entropy-code-PF+LUT10`，m3=`entropy-class-PF+LUT10`。这里后端 LUT10 为 mode2，尚未给 exact code 普通内容去重，因此 **m2/m3 不是最终独立 class 比较**。

| 同 W′ 的旧四模式完整链 | m0 | m1 | m2 | m3 |
|---|---:|---:|---:|---:|
| ready 拍 | 137476 | 123888 | 112389 | 112265 |
| BP 拍 | 149601 | 136214 | 134135 | 134245 |
| ready 读字节 | 260224 | 290432 | 378816 | 369840 |

另保留原 W 的前三模式：ready **137540/123888/112389**，BP **149667/136214/134135**；只说明同一整数源下原后端 W 函数的执行结果，不能与 W′/W″ 当同函数相减。全部 raw 保存在 [cycles.csv](cycles.csv)。没有将两个叶子倍率相乘，也没有用减少 MAC 数直接估算链周期。

资源与计费边界明确如下。

| 项目 | 实际配置及权限 |
|---|---|
| 外部可见服务 | **同一八 bank，每 bank 128 bit、最多一个已接受在途请求**；源、D 配置、后端分阶段路由；请求和响应受同一 BP 日历约束 |
| 地址池 | **16384×128bit＝256 KiB**，各臂同容量。32P X 占 0…6143 字，源共享参数/图映射 `5952+local`，后端映射 `8192+local`；不沿用旧 128 KiB 全池结论 |
| 新桥接存储 | g′ **6×320×16bit＝3840 B**，每次写一个组，后端一次读六组；D **96×16bit＝192 B** 单读表；额外十个码 holding **40 bit**、一个后端选择位及控制寄存器 |
| D/桥接工作 | 每 tile 经共享 bank 读 D **12 字**；RTL 做 **1920 次 D16 查询＋1920 次局部16-bit写**；全 H384 后端读 **1280 次96-bit桥接行**。局部展开可与下一组源计算重叠，实际 FSM 计时，不额外虚加/免除 1920 拍 |
| 算术 | 源 **10×16×24 MAC** 与后端 **96×16×24 MAC** 物理不同、串行活动，共 **106 个乘法单元**，以及两核各自既存累加器。不是同一个 96-MAC 设计，也没有证明同频/面积 |
| 后端驻留 | 只一个 H96 后端，四块顺序复用；Y **92160 B**、route mask **7680 B**、U **5760 B**、tau **5760 B**；系数 payload **384 B**、active coeff **192 B**、Y holding **288 B**、乘法管线 **1920 B**、FC forwarding **576 B**，其余 A/D/index/sign/class/目录/控制状态见实际 SV |
| 源驻留 | 保留现核的 A **200 B**、tau **60 B**、D **192 B**、info **12 B**、roots **24 B**、rank **48 B**、cache **128 B**，以及节点/地址/在途、X holding、U 和乘法管线；桥接 D 未借用这些寄存器 |
| 每次加载税 | 源每 P 仍冷装：static64 **30 字**、图臂 **21 字**；每 H96 后端 direct/LUT10/dedup 为 **382/394/397 字**。这些共享池读都计入，不把图、metadata 或 tau 免费供数 |

主周期从 top start 接受到最后 gate 接受，包含所有池内读、实际源生产、桥接和四个后端块。`cycles_to_done` 另记到最终 done 握手，`first_X_request_to_last_gate` 另记去掉前置配置后的区间；阶段计数统计到 done，不混称同一终点。**初始外存→共享池的装填未建模**，因此这是池已就绪、寄存器冷装的组件服务，不是整网/外存端到端延迟。gate 输出反压、八 bank 请求反压和 1…5 拍返回延迟均实测；cache/prefetch 带来的额外读也保留。没有 EDA/PPA 推断。

验证包括 2 个真实训练 tile、全零 X、signed X24 极值交替 tile，ready/BP、两个连续遍次。初版整链 112 命令，新强控制各 32 命令，共 **176 命令、21,626,880 个 Y/U/gate 标量位置**逐值核对；每个运行仅初始 reset，内部 **32 个源任务、4 个 H96 后端、跨 tile/模式及第二遍均通过握手重用**。C++ 同时逐源 MAC/U0/gate、nearest code/class、每次 D 展开/桥接读、输出顺序、持有请求、在途数和输出稳定性断言。独立 Python 从 X 重新得到 semantic routes/touched rows，复核全部 176 行的源任务、MAC、配置、系数读、向量更新、消费者 MAC、桥接计数；两遍记录完全一致。[full.log](full.log)、[strong.log](strong.log)、[adapt_strong.log](adapt_strong.log)、[语义核验脚本](profile.py)。

B 是先生产不必要的全部源通道、随后才发现码的消费者响应相同；强 A 是 static 删除公共项、D 固定排序/普通 ROBDD/预取，以及后端普通内容去重。本 wrapper 只是把这些成熟机制接成可审的实际链，**不构成新的分类或去重算法**。根 [W″ 一次适配](../source_class_adapt/EXPERIMENT.md) 将有限响应等价约束选到可减少完整 T10 生产的位置，本链证实这次在给足普通控制后仍有小幅净服务收益；它以不同 W 和额外图请求为代价，只有局部、训练来源的证据，未建立 AEE/面积优势，也不能跨仍观察原 g/U 的第二消费者。

复现：`bash run_all.sh`。该脚本重建唯一 wrapper，重放保留四模式及原 W、W′ 的两个强控制、W″ 的两个强控制，最后运行独立 profile 和汇总；原始输入/权重/父 SV 只读。小例可用 `bash run.sh 1 1 1 1`，它会写当前 `cycles.csv`，完整结果应随后用 `run_all.sh` 再生。
