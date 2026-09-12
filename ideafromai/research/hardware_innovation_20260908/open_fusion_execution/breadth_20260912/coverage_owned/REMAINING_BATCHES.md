# 可执行剩余批次

每批都先保留最强普通对照，明确一个新接口，才开始执行。这里是覆盖补漏的具体任务清单，未标“完成”的条目不算实验已发生；root/训练代理的并行结果应由它们的实际输出更新。旧布局负结果只停止那个布局。

| 批次 | 具体最小接口与输入 | 普通强对照和共同资源 | 本轮状态/下一项证据 |
|---|---|---|---|
| B0 身份与指定正文 | 修正 FireFly→SCNN 对应；分别核 CSA/COMPASS/ASNA/ERAFT/SPARTA 正文，补 L07–09 | 原题名、原任务、位宽与器件不能混；不因缺原文阻塞通用控制 | 已逐行分离身份/外部/不适配；未修改原目录 |
| B1 NRV×W 与 native→BN | 固定16line cache/无cache合并；完整外部gate/PED→native→全域BN→join及两块目录复用均已跑；下一步真实I24前段整层 | dense/index同缓存/合并/队列；采用两块目录复用的普通fused强后缀，新增W流量和raw/PED外存真实收费；不同端点不能相加 | [NRV前端](NEW_INTERFACE_RESULT.md)、[native链及复用审阅](native_chain_review.md)已完成。仍缺原生I24生产、NRV/NR4前段到消费者全层，以及新训练参数整链 |
| B2 多上下文与交织 | 固定F_live=2已实跑；源/连续PED同96RF真交错已完成首P2及完整A窗口两边界 | 同总RF/状态/ROM/端口/issue/背压；普通完整CSE、P1留Z立即消费、低状态普通控制同权限；真实单DMA缓冲 | [F2的352次](flive2/README.md)原布局负结果保留；[交错40例](../hardware/actual_interleave/full_window/README.md)完整A的dense/lifting/34净少0.68%/1.90%/2.61%，作为公共底座。下一改源→preview生产接口，不扫当前保存/轮转布局；不关闭家族 |
| B3 同预算可训结构 | 新dense/普通34/lifting已完成同父320步GT及十帧；新常量源RTL和matched局部六例CPU已执行；下一步质量与真实全链 | 同恢复步数/样本/seed、原逐半步RNE；参数量不同明确记录；普通源小RF重算/溢写同权，不以61RF当dense下界 | [匹配训练](../algorithm/matched_training/run.json)、[新源执行](../source_execution/)、[matched局部](../hardware/matched_local_chain/)为动态证据；[valid825](../algorithm/valid825/run.json)及三组GPU halo核对已完成。PIT/新pairing/HiNM/VENOM开放 |
| B4 表示与低位真实执行 | affine/diag/full等两学生×五表示十帧、CR及紧凑RF5+5解码、25条W4/W8 code+scale局部链均已做 | 普通affine、缓存条件加、同函数expanded16为强对照；位数/row-scale/原RNE/解码/共享端口实际收费，父版本分开 | [表示质量](../representation/aee/run.json)、[解码器](../hardware/prediction_decoder/README.md)、[packed执行](../hardware/PACKED_RESULTS.md)。尚缺完整消费者压缩残差/step、不同系数响应解码及MiLo/ReverB/joint sign+rank恢复；不继承父825 |
| B5 因果帧间与证书 | 用已连续 4 帧资料，上一完成预测/原事件 wake 后做完整尾部；新学生再采连续帧；证书改用真正不同参考轴才算新接口 | 恒等/普通 delta/不动预测、固定历史预算；过去信息，无当前或未来 flow oracle；扣除 exact/zero 普通命中，连续消费者也必须正确 | 原 4 帧机会已做；完整数值/动态 BN/恢复/端口未闭。Pro§6.2严格equal-a/水平P2已在9/11实试；本轮独立复放同305760lane/38220SIMD8命中0，不能计新增接口 |
| B6 受限公共计算 | 固定 lifting 子图的小匹配表，有限父值跨 K 生存；至少一个完整 K 与后继排程 | 无 online 全表发现的普通 CSE、同匹配/元数据/状态/更新费 | M20/K16/H8 原布局负结果已做；受限子图与原 Prosperity/Phi/ExSpike 全接口未做 |
| B7 其他精确数字接口 | 分开执行当前学生 attention 行 memo、三 pop 项门控、字转置；只在必须 spill 处 codec；bit-PE 需同面积可比较 | 保留分母/非零默认值、相同物理字总容量和布局转换；普通 MAC/压缩宽度作分母 | 旧叶/字串行/广播/codec 已做，不能据此判所有新布局无效；一次固定布局，不无边界扫参数 |
| B8 网络边界替换 | 先一个固定 attention/block 或 tokenizer/decoder 层，再等预算恢复；原 GT 与原 head | 同任务原 NB0、最强当前学生、同训练预算；IAND 等会改变连续分支，不称无损 | 多数候选原网络确实未试；网络替换是新学生，与旧叶硬件证据分开 |
| B9 可迁性筛选 | 平台/CIM 只取有明确数字地址/调度接口的一项；工具只服务已选 kernel | 原宏/工艺/FPS 不能搬，数字替代要重新计成本 | 库、综述、器件应用不需要为“全部试完”造空实验；新可迁数字接口出现再执行 |

B1已补NRV前端、完整外部gate/PED起点native链和固定两块目录复用，B2已补固定F2与两种真实96RF交错边界；[原设计与完成链接](INTERLEAVE_NEXT_INTERFACE.md)。B3三臂匹配训练825/GPU halo及两项新源的独立825均完成；B4的表示、解码、packed执行已完成限定范围，不能继续列“affine未试”或“压缩码未执行”。训练公平性见 [代码审阅](matched_training_review.md)。B5同水平P2证书是复放，不计新增接口。其余批次仍是明确剩余工作，不把旧布局失败当家族终止；真实I24前段整层、因果帧间完整数值和新训练参数全链均保留。

B3/F05/F20另完成[固定两项signed-PoT常量接口](../source_constant_probe/README.md)：两函数独立825现为1.209053834/1.235243345，均优于NB0；四局部链及真实GPU整数端点匹配，不继承父825。其[同函数低RF普通控制](../source_constant_probe/dense_low_state/README.md)229字/13工作RF＋门RF合法，已与普通34的固定13RF编译一起进入首P2交错消融，仍输最强CSE立即消费基线。低状态不是第三个AEE臂，也不等于PIT/DeepShift完整训练复现。

收口更新：native新W8已实际完成唯一旧ordinary同父十帧AEE1.149276778；真实首帧两个输出窗和全域统计对CPU0 bit差。B1完整后段仍从外部gate/PED起，不是上游I24全层已闭。B3三新学生825为1.208330/1.421461/1.225495；两项新函数另已完成自己的825，W8仍只有十帧。当前已排GPU任务均完成，没有遗留后台训练。
