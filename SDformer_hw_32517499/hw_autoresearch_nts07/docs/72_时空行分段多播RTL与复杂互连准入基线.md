# 时空行分段多播RTL与复杂互连准入基线

## 1. 设计目的

最终gate目录和普通乘法后端只减少`gate×weight`产品生成与权重读取。一个产品仍要提交到目的
bitmap中的每个token，因而多播交付和accumulator bank冲突可能成为真正瓶颈。本轮实现
`hitflow_segmented_multicast.sv`，作为以下比较的保守基线：

```text
简单当前段驻留 + bank-aware发射
        对比
层次选择、蝶形或Benes跨段网络
```

只有简单网络在真实trace上确实受互连限制，复杂网络才有实现和论文消融价值。

## 2. 精确接口语义

输入是一项已生成的产品：

```text
{tag, destination_bitmap[161:0], product_vector[OUT_TILE][16:0]}
```

输出共享一份tag和product vector，每个bank只输出独立`valid/token_id`。因此宽产品总线不按bank
复制，bank之间可以独立反压。每个目的token只允许握手一次；只有全部目的提交后才发
`product_done`，done反压期间tag保持不变。

首版bank映射为`token_id mod BANKS`。每个bank每拍最多选择一个目的，天然避免同拍同bank写
冲突。异或window、diagonal等映射必须等真实ordered trace后用相同接口比较。

## 3. 时空行分段

默认参数为：

```text
TOKENS         = 162
SEGMENT_TOKENS = 18
BANKS          = 2
OUT_TILE       = 8
```

162可被划分为9个18-token段。结合当前81个T2时间对口径，18是一个自然的时空行探索粒度；
但真实张量线性化与bank locality仍须用ordered trace确认，不能只凭形状宣称最优。

首版要求：

```text
SEGMENT_TOKENS <= TOKENS
SEGMENT_TOKENS % BANKS == 0
```

这样每个段的bank相位一致，段内offset直接决定bank。非法参数会令`product_ready=0`并拉高
`protocol_error`，不会静默生成错误映射。16-token/2-bank、18-token/2-bank和32-token/2/4-bank
均应进入后续DSE。

## 4. 被淘汰的第一版

第一版保存完整162-bit pending bitmap，并在每拍为每个bank扫描全部162个token，同时通过
动态segment边界筛选。它通过功能、SVA和Yosys结构检查，但默认参数综合为：

```text
wire bits = 16,104
cells     = 3,122
```

问题不是功能，而是全局优先编码、动态段比较和宽选择网络吞噬了稀疏多播的收益。这版不能作为
合理基线，因此被结构淘汰。

## 5. 当前段驻留重排

第二版只保存：

```text
segment_pending_q[SEGMENT_TOKENS]
remaining_q[TOKENS]
segment_base_q
```

接受产品时，把最低18位载入当前段，其余bitmap常数右移进入`remaining_q`。当前段排空后再装入
下一个18位片段，并把剩余bitmap继续常数右移。选择器只扫描18个offset，不扫描162个全局token；
全局token ID由`segment_base+offset`生成。

相同默认功能口径下，Yosys结果变为：

```text
Found and reported 0 problems
wire bits = 3,596
cells     = 294
```

相对被淘汰版，generic cell减少约90.6%。这只是同一开源综合流程中的结构证据，不能写成目标
工艺面积下降90.6%，也不能外推功耗或Fmax。

## 6. 验证结果

定向仿真覆盖：

1. 三个segment中的稀疏目的；
2. 同一bank多个目的按顺序串行；
3. 一个bank阻塞时另一个bank继续提交；
4. 共享tag和正负17-bit产品在反压期间稳定；
5. 七个目的逐项恰好一次，无重复、无遗漏；
6. done连续反压后恢复；
7. 空bitmap产品拒绝；
8. 产品、目的、issue、segment推进和bank stall计数口径。

Icarus、生产RTL Verilator lint和带绑定SVA的Verilator仿真全部通过，Verilator构建日志0 warning、
0 error。SVA逐bank检查阻塞稳定性和`token_id mod BANKS`映射，并检查done稳定与非法输入拒绝。

Erie内置启发式只对两个标准SystemVerilog参数化`for`循环报告literal warning，0 error；
生产RTL由Verilator和Yosys完整展开，故记录为方言提示。

## 7. 对复杂互连创新的结论

当前不能把蝶形网络当主创新。复旦ISSCC工作已经公开in-memory butterfly zero skipper，其他
ANN/SNN加速器也已有层次多播、Benes或crossbar分发。可辩护的新意必须来自本网络与最终gate
目的集合、时空段布局和exact accumulator语义的联合设计，而不是蝶形拓扑本身。

复杂inter-segment网络只有同时满足以下条件才进入RTL：

1. 简单当前段网络在真实trace上的交付效率低于85%；
2. inter-segment或bank阻塞占完整projection周期至少15%；
3. 高跨段fanout在多数block和p95/p99中持续存在；
4. 同库、同频、同bank数下完整projection EDP改善至少15%；
5. 子系统面积增量不超过10%，Fmax下降不超过5%。

若条件不满足，简单当前段网络就是最终架构；论文贡献应转向最终gate元数据前推、产品驻留与
full-encoder生命周期数据流，而不是强行加入复杂互连。

## 8. 下一步

1. 实现2-bank同步1R1W accumulator，消费当前多播接口；
2. 对连续同地址、正负累加、首次写、bias一次性和输出反压做逐位验证；
3. ordered profile完成后回放真实bitmap，得到每段密度、bank利用率、stall p95/p99和交付效率；
4. 扫描16/18/32-token段与2/4 banks，并把Yosys/DC互连开销计入完整projection；
5. 只有准入条件满足后再实现蝶形或Benes版本。

本轮证明的是“局部当前段驻留明显优于全bitmap扫描的可综合表达”，不证明复杂网络有收益，也不
证明G1目录已经带来full encoder净收益。
