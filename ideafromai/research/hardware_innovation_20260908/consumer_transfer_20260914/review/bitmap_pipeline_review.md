# bitmap_pipeline mode11–13 有界静态审阅

2026-09-14。独立逐项核对当前 `bitmap_pipeline/decomp_core.sv` 相对旧 `fusion_ten_trials_20260914/decompositions/q1_bitplanes/decomp_core.sv` 的全部差异，读 PLAN、TB、现有 results/SUMMARY；未运行实验或 EDA，未修改实现。源码行号以本次读取为准。

**结论：未发现新增功能 bug。** 三臂分别去掉独立构造拍、重叠系数读与上一 pop、跳过空 bitmap 块，作用点和收费可分辨。它们是旧位平面机制的接口适配，不能称完整 BISMO 迁移或三个新算法。

## 逐项核对

| 项目 | 核对结果 |
|---|---|
| 原生输入映射，SV166–179 | 令 i=10p+t，新索引 i/20=p/2、(i/10)%2=p%2，恰好等价旧 src_masks[p] 的四位置 gather；i%10 保留原时间。每个 k 向 40 bank 的 k/16 字、k%16 位各写一位。 |
| 边界与 dead k，SV160–179 | L_LOAD 把图外值置零，再进入 gather；k_live 为零也实际写零，未把外部污染或旧 bitmap 遗留带入。最后一次 local_source 写在此前时钟边沿完成。 |
| mode11 融合 | bitmap 写的数据直接来自已有 local_source 四个 10 bit 选择，与当拍 src_masks 非阻塞写无依赖；无额外源口。仍有每 k 一次 aux_writes，并非免费提供预排 source。 |
| mode12 valid/权重 BP，SV209–225 | 本拍执行旧 pipe_valid 对应的 bpq_hold，再决定是否接收下一 plane。权重不允许时不改 bm_plane/bpq_hold，旧任务执行一次后 valid 清零；后续等待不会重复执行。weight_allow 控制本地数组读取，并非异步返回通道。 |
| plane 标签/符号，SV78–80、99、109–117 | 消费用 pipe_plane 而非已推进的 bm_plane。plane0/1 加 pop×1/2，plane2 通过共享 ALU 补码减 pop×4，包含 signed3 的 −4。pop16 输出 5 bit 可表示 16。 |
| 最后 plane，SV217–231 | plane2 接收时 bm_plane→3、valid→1；下一 BM_PIPE 拍先提交最后 bm_acc 再转移，BM_STORE 下一拍读取更新后的 bm_acc。plane2 被跳过时也正确排空更早任务。 |
| live 首次覆盖，SV174–176 | 每 16K 的首 k 直接赋值该块 live，其余 15 次 OR；遍历全部 864K，54 块×16 位完整覆盖。无需 reset 清 2160 个 live 位。 |
| live 枚举，SV34–40、81–85、190–198 | BM_POS 复制当前 fp 的 54 bit live；降序循环最终选最低置位，与 pending&(pending−1) 一致。空行直接 BM_STORE 写零；消费期间 bitmap/live 不再更新，不存在并发失效。 |
| 跨命令/切臂 | 新命令仍从 k=0 完整构造后消费，故旧 live/bitmap 都先覆盖；每个非空 BM_SCAN 清 pipe_valid，空行不执行遗留 valid。mode14/15 的原状态路径与原符号运算保持。任意未支持 mode 不在该结论范围。 |

配置仍为先在 IDLE 完整加载，再 start。bp_live 的“每块首 bit 清、其余 OR”依赖原有有序 Q1 配置契约；乱序部分更新并不是本次支持的新接口。当前 TB 两次无 reset 命令使用相同 source/gold：验证重复启动和 BP holding，**尚不等于实测 live 非零→全零或跨 mode 重配**；上表跨命令结论为完整覆盖的静态依据。

扩到 Q1=−4 后，z 保守界为 [−3456,2592]，仍可用 signed13；raw 绝对保守界 8×3456×32768=905969664，仍可用 signed32。旧资源表的 ±2592/679477248 仅适用于旧 [−3,3] 契约，不应直接当新范围。

## 资源与实际省拍来源

- 现有 4320 B source bitmap、2592 B Q1 位平面副本、8 棵 pop16、8 个 32 bit 加减 ALU 保持；BM_PIPE 一拍至多读一个 128 bit plane 向量，并用旧 128 bit bpq_hold 执行一组 pop。重叠不需要第二份权重 hold 或第二权重读口。
- BM_PIPE 与 BASE_MAC/ZADD 互斥，同一组 ALU 没有同时承担两项算术；z 的 208 bit 口、p_mem 输出写/读未增加。pop树与 shift/sign/add 仍是单拍组合路径，新增 overlap 未缩短该路径。
- mode13 新增 40×54=2160 bit（270 B）live、54 bit pending；mode12 新增 1 bit valid 和 2 bit plane 标签。bm_after/first_live/next_live/execute_plane 为组合量；需计入编码、选择、扇出逻辑。
- gather 当拍仍同时写 40 个 bitmap bank 的单个位；live 另有 40 个选定位的读改写，并在消费时读选中行的 54 bit。它们可由寄存器/位使能实现，不能无条件解释为普通单口 SRAM 免费 bit 更新。没有新增源数据端口，不代表没有新增元数据访问。
- 无 BP 时，一个非空 bitmap 块的旧 plane 部分需 3+L 拍（L 为活系数 plane 数），BM_PIPE 需 4 拍，净省 L−1。若 L=0 反而多一拍；L=1 不省。固定 drain 槽仍收费，流水没有跨 bitmap 块接续。

现有 SUMMARY 的真实八块无 BP core：mode14=87599，旧15=103908，11=96996，12=90988，13=76712；这些是现有记录，本审阅未重跑。

| 相邻消融 | 由计数精确解释的 core 差额 |
|---|---|
| 15→11 | −6912 = 8×864，合并 BM_PACK |
| 11→12 | −6008 = 9012 次 pop−3004 个非空块，各块均三活 plane |
| 12→13 | −14276 = 17280 个扫描块−3004 个非空块 |

上述五臂 Q2 MAC 均为 17556，13 并未省略消费者计算。八块 SUMMARY 的 service 还计入各块完整配置及 start，不能与另一连续流只配一次静态权重的冷 service 直接混比。当前 results 有 400 条命令、20 个 fixture、5 mode×2 BP×2重复；逐 raw 校验在 TB，尚未把未出现的独立检查报告当已完成证据。

## 借入 BISMO 的准确边界

BISMO §II 给出按权重位展开、AND/popcount、移位和符号累加；§III 的体系结构是 fetch/execute/result 三阶段，具备各自指令控制、共享缓冲、同步 FIFO、DMA 和可伸缩 DPU 阵列。原文也明确软件可跳位和重叠计算/传输。[BISMO FPL 2018，§II–III](https://www.sjalander.com/research/pdf/sjalander-fpl2018.pdf)

本叶借入的是上述算术及“供数与执行可以重叠”的局部原则。固定二进制源×signed3、16K×8 rank、单响应寄存器的 BM_PIPE，没有搬入完整 ISA、DMA、同步 FIFO、通用矩阵双操作数调度或 result-stage 重叠；完整 source bitmap 构造完才开始消费。mode13 的 live 是本地生成的源块元数据，不能据此声称首次发明 sparse skip。

本次最具体仍未接上的接口是**跨 bitmap 块供数/执行衔接**：现有单 bm_hold 下，下一 BM_SCAN 不能直接覆盖尚在消费的旧 source word。若后续要移除每块 drain/scan 间隙，需要明确第二 source hold 或等价标签生命周期，并按真实数据口/状态收费；这是已定位的未接接口，不是要求本轮再做实验。

证据路径：上述 `bitmap_pipeline/*` 相对 `consumer_transfer_20260914/`；旧源码及 REPORT/resource_contract 相对 hardware_innovation_20260908。未读取正在编辑的 count_rr/gustav 实现。
