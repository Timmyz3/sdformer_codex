# 从创新点推进硬件：第一版原型与取舍

2026-09-06，唯一投稿目标 TCAS-II。**已完成一条候选的三个 RTL 模块及功能验证；另保留一条需要新训练的运动残差表示。旧 C1/C2 作为实现和强对照，不再承担标题贡献。** 两条候选目前的独立新颖性判断均为 6/10，没有达到可称“稳 accept”的程度。

## 已实现：晚知分界下的发放与 θ 幅值确认

普通实现要保存宽 FC1 中间值，等全域 BN 统计到齐，再执行后继判决。这里希望保存足以确认后继结果的信息：候选发放位、K 个边界残差、丢弃值的两侧范围，以及真实 θ 幅值。最终分界到达后检查范围，只提交能证明正确的结果；不确定则请求完整 T 恢复。

这条研究的变化在于**跨全域统计屏障保存什么、何时能释放宽值**。消费者编码已有 [Gist，ISCA 2018](https://www.cs.toronto.edu/~pekhimenko/Papers/ISCA18-Gist.pdf)，估计 BN 与按通道回滚已有 [2020 年公开技术文本](https://patents.google.com/patent/US20200160123A1/en)。候选必须在这两类强先验之上，证明精确输出证据、有限残差及实际恢复组织的净价值；不能把预测、top-K 或提交控制各算一个新贡献。

```mermaid
flowchart LR
  U[宽数值区间与预测分界] --> E[有限残差编码]
  E --> S[导出记录 释放编码上下文]
  S --> V[最终分界到达后确认]
  V --> G[收齐同组全部 T 个结果]
  G -->|全部确定| C[仅一次提交 发放标志与θ幅值]
  G -->|存在不确定项| R[请求外部完整 T 恢复]
  R --> C
```

- [encoder](rtl/atlif_threshold_packet_encoder.sv)：流式保留近分界的 K 项，被逐出的项进入范围证据；记录可导出，避免整个 BN 域被一个待确认包堵住。
- [verifier](rtl/atlif_threshold_packet_verifier.sv)：支持正负比较方向、端点区间、身份核对及背压。区间仍跨分界时请求恢复，θ 的 32 位载荷原样保留。
- [group commit](rtl/atlif_threshold_group_commit.sv)：全部 T 个时间包确认后才允许下游消费，避免先消费部分时间步、随后完整重算造成重复累加。

[完整接口与成本合同](contract.md)明确了范围。默认 B32/K4/W32 的记录为 470 位，按 128-bit 字对齐为 512 位。双端点接口比旧点值包贵，旧 CPU 包的存储数字不能套用。BN/PSN 的可信区间生成、外部包存储、恢复算术和同资源净成本尚未完成。

## 验证结果

[最终结果与源码 SHA](records/functional_r3/result.json)：Verilator 功能仿真五组配置全部通过，合计 **167,928 次逐结果比对的 verifier 请求、115 个完整 T 组场景**；另有 20 项同拍替换/背压检查。默认是 B32/K4/W32/T10；小域覆盖 K0/K2/K4，另有 B5/T3 非二次幂配置。

测试包含端点相等、最小负数、区间倒置、保留项仍不确定、K 淘汰、重复/越界位置、θ/epoch/context 错配、末拍及多时间失败、reset 和恢复后的单次提交。小域点值组合穷举与随机区间分别检查，不能称全参数形式证明。

这些是**合成区间接口的 RTL 功能结果**。输入区间和完整 T 恢复值由测试端提供；没有执行冻结网络，不是 RTL 加速比、AEE、原 FP32 等价或 ASIC PPA。r1 因本机旧 Verilator 不支持 `--build` 未进入功能测试；r2 四配置通过后，按独立审查补足协议测试，形成 r3。失败与旧结果原样保留。

## 第二方向：已解释运动的实值默认状态

[机制说明与强先验](motion_default_state.md)：用因果运动预测构造当前残差，新网络传播“当前实值默认状态＋坐标化实值例外”。全域 BN 或 θ 改变大片区域的共同状态时，更新默认值；缺席位置仍有数值和稠密光流输出。

本次做了 [200 个完整统计及 8,400 个满秩时间混合/幅值判决的有理数示例](records/default_state_examples.json)，保留了“所有 gate 相同、θ 不同则输出仍不同”和“错误周期位移也能得到零残差”的反例。它尚无 RTL、训练或 DSEC 收益。

其新颖性压力来自 [Graham 等，CVPR 2018 的非零 ground state](https://arxiv.org/pdf/1711.10275) 与运动增量网络。直接“粗运动＋token 裁剪”又被 [Motion-aware Event Suppression，2026](https://arxiv.org/html/2602.23204v1) 覆盖。保留的是默认状态随完整统计及连续阈值幅值变化时的执行问题，不把这些已有工具换名算新发明。

## C1 为什么还没有晋级

ExSpike 的共同通道贡献仍值得作为强对照，但多消费者签名共享已经有本地筛查和 SumMerge/RSR++ 等先验。本轮另审“空间边界卷积＋跨 BN/PSN 的区间消费”，独立新颖性为 4/10：差分卷积已有 [Diffy，MICRO 2018](https://microarch.org/micro51/Program/main/index.html) 这个强近邻，剩余部分接近游程执行；错位脉冲能使联合区间退化成逐像素。Diffy 原 PDF 本轮未取得，不能声称已排除其全部细节差异。没有以缺 RTL 扣分，也没有为凑第二个硬件贡献再次实现它。

[独立审阅与处置](records/independent_reviews.json)保留三个方向各自的意见。下一项能决定主线的证据是第一条在实际 BN/PSN 数值与有限存储恢复服务下的净收益，以及第二条相对“运动增量＋共享默认态”强基线是否还有实质增量。当前五页论文不同时堆叠两条候选。

## 复现

在本目录运行，输出目录必须不存在，以保护已封存结果：

```bash
/usr/bin/python3.12 sim/run_functional.py --output-dir /tmp/atlif_packet_review_run
/usr/bin/python3.12 sim/check_default_state.py --output /tmp/motion_default_review.json
```

不需要 EDA license；不调用综合、训练或生产 runner。主硬件树、主稿、docs/359 和外部 GrokBot RTL 保持原样。
