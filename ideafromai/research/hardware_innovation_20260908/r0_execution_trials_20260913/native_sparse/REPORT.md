# r0 原生数据流：三种同资源调度与固定结构剪枝

2026-09-13。**真实连续输入的完整线性叶已经闭合。** 8 个原生 4×4 tile、每个完整 T10/C96/N96/K864→2×2 输出，三调度、三份固定掩码、两种背压、两个不 reset 命令，共 **288 runs、1,105,920 个 signed32 输出零差**。源码与实测见 [native_sparse.sv](native_sparse.sv)、[real_results.json](real_results.json)、[汇总](real_summary.json)。

源优先调度在本资源点胜过跨四空间消费者复用 W 的强权重优先控制。**这是借入数据流的执行收益；物理费用剪枝尚未胜过普通幅值块剪枝，不能申领新 X。** 本次不是 CPU 操作计数，也没有 EDA、频率、面积、能量或整网速度结果。

## 真实输入与主要周期

正式输入为 Python 3.12 环境新捕获的 [r0_contiguous_t10.npz](../data_and_quality/r0_contiguous_t10.npz)，matched-dense stage320、`zurich_city_09_a_0001.npy`。八个输出起点在观察源值前固定：四角 `(0,0)/(0,318)/(238,0)/(238,318)` 与内部 `(32,48)/(60,80)/(120,160)/(180,240)`。不是从旧 64 个离散 patch 拼图。θ=1 已折入 Q16 W，权重范围 −13,925…16,243。

**下表均为 Verilator 无背压实测，加上每个新 tile 的真实 source/origin 装入，单位 cycles。** 每一行只计第一次命令；第二次命令用于检查持久状态和清零，不当成免费新输入。

| 同一整数函数/结构 | 输出优先 | 权重优先，跨 P4 复用 W | 原生源优先 | 源优先 W 词数，128bit |
|---|---:|---:|---:|---:|
| dense | 1,153,440 | 798,816 | **480,384** | 41,088 |
| physical25 | 886,561 | 615,923 | **391,288** | 31,843 |
| magnitude25 | 876,794 | 607,501 | **379,099** | 30,977 |

每新 tile 的 source 1,536 拍加 origin 1 拍，共 1,537；八 tile 共 12,296 拍。上表假定静态 W/mask 已驻留；**首次批次另加 W 10,368 + mask 288 = 10,656 拍**，三个源优先数变为 491,040 / 401,944 / 389,755。完整第一次单 tile 配置为 12,193 拍。不同 tile 的执行和装入没有重叠；这些加项是实际接口拍数，不是理想 DMA 吞吐。

核心清零到 done 的总周期分别为：dense 源优先 468,088、physical25 378,992、magnitude25 366,803。含新输入装入后，dense 源优先比强权重优先少约 **39.86%**；physical25 比 magnitude25 **慢约 3.22%**。

所有 mask 的校准与 RTL 测量使用同一八 tile，属于 **calibration-window 执行结果，没有空间留出泛化证据**。每臂独立输出是 30,720 个；多模式、多背压和重复命令是功能覆盖，不能扩大数据样本量。数据身份、掩码后的精确 W、origin，以及三份算法导出中的全部 92,160 个整数 golden 已由 [verify_inputs.py](verify_inputs.py) 独立逐值核对，[input_identity.json](input_identity.json) 全部一致。

## 固定资源与真实执行边界

| 资源 | 三种模式共同合同 |
|---|---|
| native source | 单 bank、1R、1536×10bit；一个字为同 `(c,y,x)` 的完整 T10，1 拍注册读 |
| W | 8 个 bank，每 bank 1R、10368×16bit；`bank=n%8`，共同 row=`(n//8)*864+c*9+tap`；每拍最多一个 128bit 向量 |
| psum | 8 个 bank×480×signed32，完整 3840 输出驻留；row=`(og*4+p)*10+t`；PS_READ 与 ADD_WRITE 分拍 |
| 数据算术 | 8 条 signed32 加法器，WA+WB 与累加共享；WA+WB 保存 signed17，无额外数据加法器或乘法器 |
| 共用状态 | 80bit 四空间双源 mask、20bit 原生双源暂存、WA/WB/sum、psum 读寄存器、输出寄存器；三模式均保留 |
| 静态结构 | 288bit 配置寄存器，N8×Cin4×全部 3×3 tap 一组；读前跳过死 C4 循环段，无免费运行时重排表 |
| 输出 | 480 个 256bit ready/valid beats；显式 drain，背压时地址/数据稳定 |

主要数组容量：W 165,888B、source 1,920B、psum 15,360B，另有 mask/小寄存器和控制。完整细目见 [资源合同](resource_contract.json)。这是明确定义的 RTL bank/端口，尚未映射 SRAM 宏；不能用相同周期数宣称相同 Fmax 或 ASIC PPA。

`output-major` 顺序为 `og→p→channel-pair→tap`。`weight-major` 为 `og→channel-pair→tap`，先付费读四个空间消费者的源，再只取一次所需 WA/WB 并跨 P4 复用。`native-source-major` 为 `channel-pair→native(y,x)`，经同一单源口两拍读两通道，再由 RTL 生成合法 `og,p,tap` 请求并持久累加。权重优先避免了上一轮“每 p 重读 W”的弱控制。

三模式共同先判断完整 10bit 需求；某个 c 在所有当前消费者全零就不取其独立 W 词。只有实际 `(p,t)` 出现两个源同时为 1，才用共用加法器付一拍产生 WA+WB。所有时间消费者的选择、循环、清零、部分和和末 drain 均在 SV。TB 没有提供 im2col、跳过答案、调度神谕或中间和。

输入 origin 由配置提供，SV 根据 `240×320` 边界与局部坐标判断源是否越界，越界直接置零并取消源 bank 访问。五个固定功能控制中包括“越界位置故意全 1”的毒值控制，证明边界正确性不依赖 capture 先补零。完整功能控制 **60 runs、230,400 输出零差**，Verilator `-Wall` 编译无警告。

本模块的函数是 `Σ S×Wq×M` 的 signed32 原始线性累加；全 signed16 域最坏界 `864×32768=28,311,552`，无需新增中间舍入。**原 noncausal PSN 生产、时间词布局生产、全图 tile 复用管理、norm/bias/residual 和后继尚未接入。** Q16 是独立诊断函数，不能继承 FP32 学生 AEE。

## 事务账本解释了收益与负结果

对 dense，权重优先到源优先：源读 **281,088→9,600** 个 10bit 词，但 W 读 **23,580→41,088** 个 128bit 词。两者部分和更新均为 73,428 个八 lane beats，psum 读/写各 77,268 beats，后者含清零/末读。完整差额精确为：

`源优先−权重优先 = −271488 源读 +17508 W读 +624 sum +0 psum −65076 控制 = −318432 cycles`。

所以“W 取词更少”不等于该固定核更快。源优先用更多 W 事务换掉大量重复源读取和扫描。这里没有 bank 冲突消除的主张：布局使每个 bank 每拍最多一个确定请求，费用已逐拍支付；没有凭空赠送任意双源口。

两份剪枝均一次删除 72/288 块。由于原 W 已有少量零，结果系数 nnz 为 physical25=62,203、magnitude25=62,202，准确说法是**同块数和同格式**，不是完全同 nnz。physical25 以组输出贡献平方误差除真实权重优先 W 请求成本排序；magnitude25 按组权重平方范数排序，未训练或扫参。[mask_manifest.json](../data_and_quality/mask_manifest.json)

physical25 的校准输出 SSE 为 1.8258×10¹²，低于 magnitude25 的 2.4555×10¹²；但其强权重优先 W 词为 17,863，多于 17,825。在最快源优先模式下，它又多 866 个 W 读、2,664 个 psum 更新和控制，总共慢 12,189 cycles。两份 mask 的源读均为 **9,600**，没有共同关闭完整 Cin 组的源事务。**当前只得到误差与执行成本的不同取舍，未证明比普通幅值控制更好的同质量性能。** 新 AEE 由数据分支另测，局部 SSE 不替代当前“优于同协议 NB0”的门。

独立 [analyze.py](analyze.py) 从 native source、几何、mask 和目的集合推导 source/W/sum/psum 请求，另外推导各状态占用；全部 **288 份真实运行与 60 份功能运行的计数及完整 cycles 精确闭合**，含实际观察的源/W/出口 stall。[real_ledger.json](real_ledger.json)、[control_ledger.json](control_ledger.json)。cycles 来源始终是 Verilator，解析公式只作交叉审计。

## 借入范围、研究结论与复现

本次借入的是普通 loop interchange、源请求驱动、局部持久 psum、两源部分和、结构整词门。ELSA 的目的聚类与 Gustavson 累加、Phi 的有限组织是明确先验；本版没有实现完整 BAER/NoC、模式字典、多 window bank-conflict packer、ST-BIF 或全论文训练，不声称完整论文复现。[前轮文献核查](../../pro_fusion_trials_20260913/NAMED_LITERATURE.md)

因此当前已完成的是**强 A 的原生供数闭环**。新的费用目标没有赢过幅值块控制，而且它使用的 W-major 成本也不是最快 source-major 的完整成本。下一次若继续结构训练，应固定本硬件，以完整请求、psum 和控制代价比较，并保留全部普通模式权限；当前一个 25% 点既不构成新 X，也不否定其他表示、训练或真正跨消费者源词删除接口。

```bash
/opt/anaconda3/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 run_real.py --capture ../data_and_quality/r0_contiguous_t10.npz --mask physical25:../data_and_quality/physical25_q16.npz:live --mask magnitude25:../data_and_quality/magnitude25_q16.npz:live --no-build
/opt/anaconda3/bin/python3.12 verify_inputs.py
```

仅修改隔离 `native_sparse/`；未做训练、GPU 模型调用、hash、生产目录或主稿改动。独立根代理代码/计数审阅在 [REVIEW_NATIVE.md](../REVIEW_NATIVE.md)。
