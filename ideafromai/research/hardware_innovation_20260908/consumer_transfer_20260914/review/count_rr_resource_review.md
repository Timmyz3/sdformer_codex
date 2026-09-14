# count_rr 共享资源与生命周期独立静态审阅

2026-09-14。范围为 `count_rr/PLAN_RESOURCE_CONTRACT.md`、`interleave_stream.sv`、`rr_context.sv`、`wide_phase_alu.sv` 与消费者仲裁/握手。负责人仍在测试，本审阅没有运行实验、EDA 或修改实现；不据尚未收口的结果推断性能。行号以本次读取为准。

**结论：未发现资源复制、漏仲裁或计数覆盖待消费 raw 的功能错误。** 本实现把 count20/21 接入共同 416 bit z 服务、唯一 producer 算术及完整消费者，原 raw 叶的 208 bit 资源说法不能移用。

## 唯一算术与逐 lane 退休

| 检查 | 代码证据与结论 |
|---|---|
| producer 乘法器 | top 116–117、182–190：唯一 8 个 signed19×13→32 表达式；两个 context 只有操作数/结果接口，没有乘法表达式。 |
| producer ALU | top 203–216：唯一 8 条 32 bit carry chain；context 的 add_y 只是 shared_alu_result 切片，没有本地复制加法器。地址/统计运算另计，不能把此结论表述为整芯片只有这些加法。 |
| 多请求原子仲裁 | top161–169：资源位为 source/weight/z/psum/ALU/wide；两个请求有任一交集时只授权一个 context，无交集才同时授权。C_ADD 申请 psum+ALU，G_MAC 申请 z+ALU。 |
| lane 选择 | top171、188–193：coefficient、逐 lane scalar、lhs/rhs 和 bit13 correction 都由同一 alu_owner 选择。不存在 scalar 来自另一 context 的交叉选择。 |
| scalar 退休 | context154–159：普通 Q2 广播 scalar；G_MAC 每两条 rank lane 选同组 count_scalar，乘各自 representative 并加对应 P/T 的 signed13 z。 |
| packed 退休 | context160–164、240–250：a≤255、b≤127 时系数为 a+2048b，乘 signed3；top193 拆 product，208 在 bit13 注入 q<0 且 a≠0 的借位修正。b>127 时退 scalar，不截断高计数。 |
| 证明与主算术 | top172、194–198、373–375：Q1 配置时正负界证明也使用共同 ALU；加载在 producer 空闲阶段。断言禁止 proof 与 runtime producer 同拍双占用。 |
| consumer/borrow | consumer97 的 8 个 32×32 乘法是合同既有独立 consumer 资源；其 64 bit 主加法仅来自 wide_phase_alu。top99–108 的 consumer/borrow 选择和 request[5] 仲裁真实共用一套 8×64 链。count20/21 本轮不请求 borrow。 |

packed 退休的 G_ZREAD→G_MAC 使用保留 z_hold，G_MAC 等待 grant 时 fp/group_pending/count_hold/q_hold 不推进；获得 grant 后只写相应 26 bit 字段，再清同一 P 对的 pending。scalar fallback 只写目标 13 bit。这里既没有免费退休乘法，也没有额外并行 z 读。

## p_mem 生命周期和拒绝保持

- `rr_context` 287 把整个执行状态机放在 `resource_request==0 || resource_grant` 下。拒绝时只有周期/等待计数推进；地址、持有操作数、迭代状态及算术结果不提交。
- 所有 p 读/写 enable 都受 grant 控制；z_read_bus 未获 grant 为零；z 写在同一状态 guard 内。top 按获批 weight_owner 才返回 Q1/Q2/class/rep 数据。故另一 context 的共享总线变化不会被被拒方锁存。
- C_CHECK 用未门控的 count_live 计算 read_banks，再决定是否申请 psum，避免 request 依赖 grant 的组合环。全首次触达时 read_banks=0、所有 p_read_data=0，可本地生成零 count_hold；后续 C_ADD 仍必须争用 psum+ALU。
- 每个 context 的 p_mem 始终为 8×480×32 bit。count 在每 bank 前 160 行，每 bank 一地址、单拍只读或写；四类地址 mux 属于真实额外控制。count_live 在每次 start 清零，未清的旧 p 字不会作为有效计数读取。
- 顺序固定为 Q1 累计→完整 group retirement→ZSCAN→Q2 的 480 行 STORE→DRAIN。计数阶段不再回返；DRAIN 前全部 p 字已覆盖，不会读到 count 残留。
- top232–235 同批启动两个 context，但只给当前 consumer_select 的 raw_ready。另一个 context 可读入自己的首个 result_data 等待，DRAIN_SEND 时 raw 地址/数据保持，无后台计数写。
- top509–534 只有在当前 consumer 完成 480 个 I24 beat 且 producer raw 已完成后才切换消费者；整批两个消费者都退休才装下一批 source/重新 start。check_owned 也禁止消费者退休前重配或重启。
- parameter_valid/source_valid 为零时加载地址保持；compute_source_allow/weight_allow 通过 eligible 撤销 grant。consumer 的 raw/identity 各有 have_p/have_j holding；64 bit add 被拒时 wide_hold/state 保持，输出 BP 时 result_data/row 保持。

相关现有断言覆盖资源双 grant、proof/borrow 双占用、未获批存储访问、p 单 bank 读写冲突、count 地址/溢出、480 行覆盖和 context 所有权。它们不是对所有状态位 holding 的形式证明；本结论来自静态 guard 核对，动态场景以负责人完整结果为准。

## 配置、端口与比较公平性

| 资源/费用 | 当前合同在代码中的落实 |
|---|---|
| z | 每 context 8×10×52 bit=520 B；共同 vector 服务 416 bit。count 两 P 只更新其中26 bit，仍独占同一 z grant，不能主张半宽面积收益。每 bank 单一授权读地址/表达式。 |
| psum | 每 context 15360 B，不扩容；count 暂用5120 B。两个 context 的物理状态独立，但全局 psum grant 每拍至多一 context。 |
| 源 | 每 context 1920 B；compute source 服务全局仲裁。外部 SOURCE 装载与本批执行不重叠，不把装载带宽当免费。 |
| producer 权重 | top 仅一份 Q1=2592 B、Q2=1536 B；context 只有 q_hold/qblock 和支持位。 |
| class/rep | top 唯一 class 864×24 bit=2592 B、rep32×24 bit=96 B、ngroups6 bit。MREAD/G_QREAD 与 Q1/Q2 竞争同一 weight grant，不存在额外 metadata 数据读口。 |
| 配置 | 普通冷加载 864+96+864+24=1848 拍；count 额外864+32+1=897 拍；mode21 首次再加1拍。static_words 在每个实际 parameter_valid 拍增加。 |
| warm/跨 mode | resident/class_resident/permutation_resident 分开记录；先普通再count需补897拍，先20再21需补1拍。状态保留只适合本模型不变的复用；代码没有运行时换模型失效接口。 |
| 固定排列 | top 唯一40 bit；context21 原生门字按 π 选位，p 输出读地址用 π⁻¹，result_addr 保持原顺序；全部 Q2 物化后才逆序读。 |
| count 额外状态 | 每 context 640 bit count_live、256 bit count_hold、24 bit class_hold、group/pending控制等仍存在；共同 union 实例包含这些成本，不等于逐臂裁剪后同面积。 |

consumer 的系数表和 READ_A/B 属于原有独立消费者资源，不要把“producer 权重共口”扩大为所有消费者系数也共该口。source/weight/z/psum 在此是明确的仲裁服务合同，物理 SRAM 映射、端口 mux 与 Fmax 尚未 EDA 验证。

公平比较应使用此 top 的模式0/1/2/3/4/20/21、同源流、相同冷暖状态及完整 I24 wall cycles；sum(core_cycles) 含两个 context 的等待，不能再与 consumer_cycles 相加充当 wall time。原208 bit raw叶结果可以解释迁移起点，不能代替当前416 bit消费者强控制。

**唯一需保留的接口边界：** 本实现以整批 I24 退休为 context 再利用屏障，因而覆盖安全；它尚未尝试更早回收已被消费者接收的 p 行。不能为提高重叠直接提前 count 写入，除非增加逐行所有权与端口仲裁。

证据路径均相对 `consumer_transfer_20260914/`。本次未审阅正在修改的 Gustav，也未修改 count_rr。
