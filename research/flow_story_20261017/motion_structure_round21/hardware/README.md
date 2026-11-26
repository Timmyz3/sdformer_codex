# 未拟合结构候选的运行时义务

[SOURCE_OBLIGATIONS_V2.json](SOURCE_OBLIGATIONS_V2.json) 绑定最终设计快照`d05c1d5a…42eb`及`structure_models.py efbfa8f5…7528`，由 [source_obligations_v2.py](source_obligations_v2.py) 仅按源码和形状生成，零模型前向、零拟合、零GPU/RTL。结构候选仍暂停；这份表不能被当作精度或净硬件胜利。旧精确 Conv/Linear/MAC清单保持原件，此次补齐输入scale、context统计、bias/ReLU、pool/repeat、edge gather/diff、输出反标准化和selector义务。

每帧完整T均值读4,608,000个FP32值（18,432,000B），4,147,200次加和460,800次缩放；输出1,843,200B，若分别落盘再读合3,686,400B。全部方法共同的generic context仍需384→16组均值及全图均值/std，后者按当前源码包含两个均值归约、方差和16次runtime平方根。G2/F共同另付921,600MAC的q、2400个bias加、真尾3×3残差与相同向量尺度。统一向量尺度的平方根属于离线fit，推理只使用已冻常数；literal scale仍对6通道逐元素减/除，未把xy的0/1或残差零均值自动当免费。各普通方法享同FP32、PE和融合机会，不强迫沿旧FP64串行评分器。

额外 head MAC分别为 generic_raw4,815,840、generic_group2,949,600、同q普通CNN990,048、结构头170,976；只包括声明的矩阵乘法，不含共同前置或此次补列非MAC。结构头保留62,208次dense membership MAC，不能把one-hot矩阵的零静默删掉并仍声称相同浮点求和。每种方法的scale后空间数组、context、节点/边/卷积输出、q/z残差及常数都列元素数；这是逻辑操作和存储对象清单，不是一次内存事务或每个元素一周期。

原服务模型总activation SRAM是2MiB、目标producer峰值1,605,632B；均值数组1,843,200B不能与该producer峰值无代价同时驻留。需要证明分阶段复用、spill/reload或producer融合及端口时隙。原JSON中“existing2MiB target working set”指资源配额，不是测得的目标峰值，此处明确两者。尚未给head有限端口日程、FP实际数值实现、旁路面积或整网周期。内部TVLSI/TCAS-II结论：这些是未来公平对照必须支付的义务，作用是避免预测器免费；是否值得实现仍由真实可行动作域及动作gain信息实验决定，不能让低MAC头替代光流增量证据。
