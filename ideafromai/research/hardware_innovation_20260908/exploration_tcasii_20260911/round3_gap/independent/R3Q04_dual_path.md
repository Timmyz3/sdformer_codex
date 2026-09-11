# R3Q04 — Dual path / join / typed last-use / when the producer can die

**Status of all items:** candidate. No scores.

**Locked identity.** AT-LIF \(o=\theta s\), \(\theta\) 吸入下一层 \(W\)。吸完后网上只有两类对象：**(A)** 二值脉冲 \(\times W\) 的 GeMM；**(B)** 另一条连续残差 / PED / I24 张量。B 不是 AT-LIF 的模拟幅值，也不是“同一连续激活喂两个 MAC”。

**本题只问：** 关于 **这一对 (A,B)** 的 **一个** 机制，能否写成 TCAS-II letter。禁止写成“连续 AT-LIF 幅值被两个 MAC 共用”。

**必须抄全的 A-prior（本题用来划界，不当标题）：**

- SDT membrane-shortcut：残差加在 SN **之前**，为的是脉冲仍二值。
- Spike-IAND：删掉 ADD，整网保持 all-spike。
- FireFly-T residual port：是 encoder 的 MS 膜电位端口，不是 PED GeMM。
- DATE hybrid：dense/sparse 是 **不同层** 的核，不是同一生产者的叉。
- ESTU typed spike vs int memory：是 overlay **存储分银行**，不是“一个生产者 fork 成 gate **且** residual”。

**本题可用的观察（不是净服务结论）：**

- integer two-consumer：另一套构造，精度 −5.78%。
- source long-BP：**两条消费者都是 8088**。
- delayed V until native BN：integer 0-diff，**arithmetic_saving=0**；early_V96=55.3MB vs late_U32=18.4MB；总线占用释放 ≠ 净服务。
- Gates：15% same-port，AEE。

---

## R3Q04-I1  双类型 last-use 令牌（T2LU）：生产者死在两枚令牌都回来时

**ID:** R3Q04-I1  
**Status:** candidate

**一个机制。** 吸 θ 之后，同一逻辑生产者被 fork 成两个 **类型不同** 的未完成使用：`s_use`（A：二值 GeMM / gate）和 `r_use`（B：I24 残差 / PED）。片上只放一个 2-bit typed scoreboard `{s_done, r_done}`；SRAM 行的 free 条件是两比特都置位，而不是单一 refcount、也不是 ESTU 的“脉冲银行 vs 整数银行”。

**为什么是这一对，不是双 MAC。** 两个消费者要的数据布局不同：A 要 0/1 位图，B 要 I24。共用的是 **行的活着区间**，不是幅值，也不是乘法器。

**不是哪些 prior。** ESTU 是 overlay 分银行，不跟踪“一个生产者同时欠 gate 和 residual 各一次”。SDT 把 B 在 SN 前就加进膜，fork 消失。Spike-IAND 根本不让 B 以连续张量存在。

**用到的观察。** 两条消费者目前都挂在 **同一 8088 long-BP** 上：若令牌类型不分流 wait-class，T2LU 退化成“两条都等 8088”，G1 那种 revise-not-title。本机制的硬件要点是 **令牌带 wait-class 域**（见 I4 可拆开写；本条的最小电路仍是 2-bit + class 位）。

**信件句子。** A two-token typed last-use on one post-absorb producer: free the line iff both the binary-GeMM use and the I24-residual use have retired.

**Kill if.** 实测 `s_done` 与 `r_done` 总是同一拍（join 绑死），或 2-bit 相对单 refcount 的行占用差在测量噪声内。

---

## R3Q04-I2  吸完即物化 1-bit 视图：A 的死亡早于 B（staggered-view kill）

**ID:** R3Q04-I2  
**Status:** candidate

**一个机制。** Absorb 被做成一次 **视图分裂**：A 立刻得到打包的 1-bit 图像，此后 A 不再持有连续缓冲；B 单独保有 I24/PED。生产者对 A 的 last-use 是 bit-pack 完成拍，对 B 的 last-use 是 join/残差消耗拍。两张物理像、两个死亡时刻，一个逻辑张量。

**为什么是这一对。** 身份已经规定层间是 0/1，不是按事件存的模拟幅值。若仍让 `{0,θ}` 缓冲活到 B 结束，A 的二值 GeMM 在交税给一条它不再读的连续像。机制是 **A 可先死**，不是“幅值再乘一次”。

**不是哪些 prior。** FireFly-T 的 residual 是膜寄存器写回，没有“吸 θ → 位图视图”这一步。DATE 是层间换核，不在同一生产者上切视图。ESTU 预置两类存储，但不规定 **同地址的连续像在 pack 后可杀**。

**用到的观察。** delayed V：arithmetic_saving=0，所以本条 **不许** 声称少 MAC；可声称的只有连续像从 early_V96 收到 late_U32 的寿命。若 pack 后 A 仍因 same-port（15%）被 B 堵住，staggered kill 失败，见 I5。

**信件句子。** After absorb, materialize a 1-bit view for GeMM and let the continuous image die on the residual consumer only.

**Kill if.** bit-pack 必须保留全宽 I24 直到 B 结束（视图不是拷贝而是别名），或 pack 带宽 ≥ 连续像的剩余寿命成本。

---

## R3Q04-I3  唯一合法 ADD 在 A∩B 的 join（typed join，不是删 ADD、也不是 SN 前加）

**ID:** R3Q04-I3  
**Status:** candidate

**一个机制。** 吸完后，A 的结果是稀疏二值 GeMM 累加，B 是连续 I24。全图只允许 **一处** 把它们加在一起：一个 typed join（一边 ready 来自 bitmap/AND 累加器，一边 ready 来自 I24 流）。握手成功即消耗 B，B 的生产者可死。Join 之外没有连续+脉冲的算术相遇点。

**为什么是这一对。** Spike-IAND 的策略是删 ADD 以保持 all-spike——本网做不到，因为 B 不是脉冲。SDT 的策略是把 ADD 挪到 SN 前——那会把 B 重新灌进脉冲路径，破坏“吸完后 A 仍二值”。本机制承认 ADD 必须活着，但把它 **钉死在 join**，从而 A 全程保持二值 GeMM。

**不是哪些 prior。** DATE 的 hybrid 是不同层 dense/sparse 核，不是同一层 A 与 B 的汇合。FireFly-T residual port 汇合的是 MS 膜，不是 PED GeMM 输出。

**用到的观察。** integer two-consumer −5.78%：把 A 和 B **都** 提升成整数再加，是另一种构造，已经掉点。Join 必须保持类型：A 侧仍是 0/1×W，B 侧仍是 I24，只有 join 是 I24 add。long-BP 8088 both：若 join 的两边 ready 绑在同一 source 槽，join 不缩短寿命，只是换了个名字。

**信件句子。** A single typed join is the only legal ADD between post-absorb binary GeMM and the I24 residual; residual last-use is the handshake.

**Kill if.** 图中仍有第二处连续+脉冲相加（BN 后、shortcut 前、另一条 PED），或 join 相对“始终把 B 驻在 RF 里等 A”不减少行占用。

---

## R3Q04-I4  在 fork 处切开 wait-class：B 不再陪 A 坐 8088

**ID:** R3Q04-I4  
**Status:** candidate

**一个机制。** 当前 A 与 B **同源 long-BP 8088**。双路径若共享 wait-class，生产者的死亡时刻 = 慢的那条，fork 是假的。本机制在 absorb/fork 处给 B 单独一张短 FIFO wait-class（残差/PED 流），A 留在长 GeMM 类。B 的 last-use 可以在 A 的 8088 槽之前退休；生产者能否死，看是否还留着 **未分裂的共享底缓冲**。真分裂则 B 先死；假分裂（别名同一行）则仍等 max。

**为什么是这一对。** 问题不是“两个 MAC 抢连续幅值”，而是 **两个类型不同的消费者被编进同一个源阻塞类**。DATE 换核不管 wait-class。ESTU 分银行不管 8088 槽。

**不是哪些 prior。** 全部 listed priors 都没有“同一生产者的两条边分 wait-class”。G1 typed last-use 在 8088 未切开前被标成 revise-not-title——本条就是那一刀，单独成信。

**用到的观察。** source long-BP 8088 both（主证据）；15% same-port（若端口未切，wait-class 切了仍会在端口会合，见 I5）；arithmetic_saving=0（本条不谈算术）。

**信件句子。** Split wait-class at the absorb fork so the I24 residual is not sentenced to the binary GeMM’s 8088 long backpressure.

**Kill if.** 切开后 B 的退休拍不变（仍 8088），或短 FIFO 深度使 B 的占用大于“陪 A 坐着”。

---

## R3Q04-I5  同端口 15% 的类型锁：bitmap 包与 I24 包不许当同一种 flit

**ID:** R3Q04-I5  
**Status:** candidate

**一个机制。** Gates 有 15% same-port（AEE）。这些端口上 A 的消费者是 **门控/位图**，B 的消费者是 **I24 残差**。机制是端口级 **typed lock / 2-slot typed queue**：同一 SRAM 口上，0/1 包与 I24 包分槽，AEE 只允许在类型匹配时放行。不是把 15% 合成一个更宽的 MAC，也不是 DATE 那种层间换核。

**为什么是这一对。** 冲突的不是“两个连续 MAC”，而是 **同一端口上的两种 last-use**。门可以先退休（AEE 把口还给别人），残差仍占 B 槽——这是“生产者何时能死”在端口粒度的版本。

**不是哪些 prior。** FireFly-T 残差口是膜口，不与 PED GeMM 抢同一 bitmap 口。ESTU 的类型在银行，不在 **同一端口的 flit 类型**。

**用到的观察。** 15% same-port + AEE 是本条的存在性；其余 85% 不值得为类型锁付面积。integer two-consumer 掉点警告：不要为了端口合并把 A 提升成 int。

**信件句子.** On the 15% same-port gates, a two-slot typed queue separates binary-gate AEE from I24 residual so last-use of each type can retire without promoting both to integer.

**Kill if.** 15% 冲突在调度后消失（AEE 已把它们错开），或 2-slot 队列的面积 > 把这 15% 复制一份 B 缓冲。

---

## R3Q04-I6  拒绝“整数双消费者”：fork 保类型，整数只许出现在 join

**ID:** R3Q04-I6  
**Status:** candidate

**一个机制。** 已有构造把 A 与 B 都当成 integer two-consumer，精度 −5.78%。本机制规定 fork 的类型不变量：A 侧指令集只有 bitmap/AND/二值 GeMM，B 侧只有 I24 load/store/addend；**任何把 A 提升为 int 以便与 B 共用一条整数流水的改写都非法**。整数运算的唯一入口是 I3 的 join。这是 ISA/微码约束，不是又一个 MAC。

**为什么是这一对。** 掉点来自“另一套构造”，说明双消费者这件事对精度敏感的是 **类型**，不是少一次加。信件卖的是类型不变量，附带的硬件是 fork 处的类型标签（1 bit/tensor）+ 对提升的 trap。

**不是哪些 prior。** Spike-IAND 用删 ADD 维持全脉冲类型；我们维持的是 **异构类型对**。ESTU 有类型存储但允许整数消费者读脉冲银行的 overlay，没有“提升即失败”的契约。

**用到的观察。** −5.78% 是本条的杀伤性对照；AEE / same-port 只说明提升的诱惑来自端口合并，不是借口。

**信件句子。** Keep the post-absorb fork typed: binary GeMM must not be promoted to integer to share a pipeline with the residual; the −5.78% two-consumer integer construction is the negative control.

**Kill if.** 存在 0-diff 的整数双消费者改写（−5.78% 不可复现或来自无关 bug），或类型标签从不被硬件检查（纯软件约定则不够 letter）。

---

## R3Q04-I7  V/残差的 last-use 是 BN-complete，不是 GeMM-complete（死在晚 U，不在早 V）

**ID:** R3Q04-I7  
**Status:** candidate

**一个机制。** 把连续对象 B（V / PED / 投影后待 BN 的张量）的死亡条件写成 **`bn_done`，禁止写成 `gemm_done`**。Delayed-V 实验已经给出：等到 native BN 完成再物化 V，整数 0-diff，**arithmetic_saving=0**。因此本条不是 BN 加速器（BN 是 hygiene），而是 **B 的 typed last-use 绑在 BN 归约结束**。早 V96=55.3MB 是把还没到 last-use 的连续像提前实体化；晚 U32=18.4MB 才是 B 的合法活窗口。A 的二值 GeMM 完成 **不能** 释放 B。

**为什么是这一对。** A 先结束不等于生产者能死——B 还在等 BN 的连续归约。SDT 在 SN 前就把残差加掉，不存在“BN-complete last-use”。FireFly-T 膜端口也不等 PED+BN。

**不是哪些 prior。** 全部 listed priors 都把 residual 当成已经可加的膜或脉冲。本网的 B 在 BN 完成前不是可 join 的 addend。

**用到的观察。** arithmetic_saving=0 是诚实边界：信件 **只许** 谈寿命/占用（V96 vs U32），不许谈少乘加。总线占用释放不是净服务——若只报 occupancy、不报 last-use 差，本条自杀。8088 both：若 BN 归约与 A 同 wait-class，`bn_done` 仍等于 8088，必须与 I4 一起才有差。

**信件句子。** The continuous residual’s last-use is BN-complete, not GeMM-complete; delaying V until native BN does not save arithmetic and must not be sold as such.

**Kill if.** last-use 在 BN 前就发生（残差根本不读 V），或 late_U 相对 early_V 的占用差被别的 mandatory 缓冲吃掉。

---

## R3Q04-I8  残差口是 PED/I24 流的 last-use，不是 encoder 的 MS 膜写回

**ID:** R3Q04-I8  
**Status:** candidate

**一个机制。** 做一个 **PED-stream residual port**：B 作为 I24 流从端口进入 join，消耗完即杀；端口状态机是 `{fill → stream-to-join → last-use → free}`，没有膜电位累加、没有 MS 编码器写回。A 仍走二值 GeMM 阵列。这是 FireFly-T residual port 的 **反例实现**：名字都叫 residual port，对象从“膜”换成“吸 θ 之后仍连续的 PED/I24”。

**为什么是这一对。** 身份把 AT-LIF 的 \(o\) 吸进 W 后，层间脉冲不再携带残差。残差若还走膜口，就是把 B 错接到 A 的神经元状态上。机制是 **端口语义**：流式 I24 last-use，而不是膜 RF。

**不是哪些 prior。** FireFly-T：MS membrane of encoder。SDT：加在 SN 前，残差进膜。Spike-IAND：无连续残差口。DATE：层间核，无此端口。

**用到的观察。** 15% same-port：若 PED 口与 gate 口是同一物理口，本条退回 I5。8088 both：流式端口必须自带 wait-class，否则只是把膜口改了个线名。

**信件句子。** A residual port whose object is a streaming I24 PED last-use, not the encoder’s membrane, so the continuous consumer can die at join without touching the binary GeMM array.

**Kill if.** 本网的 B 其实就是膜（PED 不存在或可吸收进膜而无精度差），或流式口相对“B 常驻 RF”不减少峰值行数。

---

## 八条各自钉死的“一个机制”（对照表，不评分）

| ID | 机制（一对 A/B 上的一件事） | 生产者何时能死 | 明确不是 |
|---|---|---|---|
| I1 | 2-bit typed last-use 令牌 | `{s_done ∧ r_done}` | ESTU 分银行 |
| I2 | absorb 后 1-bit 视图分裂 | A 死于 pack，B 死于 join | 连续幅值双 MAC |
| I3 | 唯一 typed join ADD | B 死于 handshake | Spike-IAND 删 ADD；SDT 的 SN 前加 |
| I4 | fork 处切开 wait-class | B 可不陪 8088 | DATE 层间换核 |
| I5 | 15% 同口 typed queue | 门先死、残差后死 | 加宽 MAC |
| I6 | 禁止整数提升的 fork 契约 | 非法提升直接 trap | −5.78% 那套 integer two-consumer |
| I7 | B 的 last-use = `bn_done` | 死在晚 U，不在早 V / 不在 GeMM 完 | BN 加速器；算术节省 |
| I8 | PED/I24 流式残差口 | 流空即 free | FireFly-T MS 膜口 |

**共同禁句。** “连续 AT-LIF 幅值被两个 MAC 共用。” 吸完之后没有这个对象。
