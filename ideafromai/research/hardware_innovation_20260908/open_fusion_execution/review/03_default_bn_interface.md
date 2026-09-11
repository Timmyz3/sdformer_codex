# code0 穿过完整动态 BN：可直接试，不能把次数当浮点等价

**试后更新：** 下文保留试验前的接口推导；上述四臂现已全部执行，见[实际结果](../default_bn/README.md)。完整零叶复用在补齐公共−μ后相对强控制仅0.47–0.55%；不是仍未开工，也不是新标题已成立。

**B 与费用。** 已有完整 BN 域是每通道 `T10×120×160=192000`，共 18,432,000 个 FP32 值。相同机器的密集树归约为 203,297,841 槽，其中均值 46,352,385、中心方差 57,872,265、规范化及输出 99,072,000；外存三遍读 221,184,000 B、写 73,728,000 B。这是整域 BN 算子费用，不能加到两个 4×4 局部窗口当整帧时间。依据是旧树的 [global_bn_pairwise.json](../../algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain/preview_sn2_chain/global_bn_pairwise.json)。

**实际依赖。** 上游 `SpikingPEDLayer.forward` 是 `x_res=conv_res(x)` 与 `sn(x)→Conv3×3→BN` 两路，最后才相加；当前 fixed helper 将前一路实现为保留原始 I24 的 U/F 残差更新及连续 PED U/V。因此 native Conv 的 code0 只证明该分支原始值为 +0，不证明原始 I24、连续 PED 或最终输出为零，也不能在 BN 后直接共享一个门字：当前投影门在 BN **之前**，下一层门必须看 `FP32(BN_default + PED[p])` 及自身时间算术。

**A 与已停边界。** 普通空源证明、Finch 默认值表示、BN 分阶段/融合均是公共底座。旧“同非零向量＋全键字典”在别层净状态薄/负，当前完整 K864 非零 LRU16 命中仅占全部 0.34–0.38%；不重做它。[bn_zero_interface.md](../../algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain/preview_sn2_chain/bn_zero_interface.md) 实际只完成来源普查与载荷情景，**尚未执行 code0 压缩 BN**：两学生 110365/110961 个完整零向量均由真实 K864 空源推出，随后全部 BN/PED 非零。恢复后的新模型不能继承此旧全图分布。

**浮点边界。** 完整分母仍为 192000。实数公式中的 `n0×μ²` 不等于原顺序重复 FP32 FMA；重新按类别/次数归约也可能改变均值。第一原型保留每个逻辑位置、四条交错部分和、256 值叶块和原树次序：tag 只抑制零载荷搬运，方差仍原 SUB/FMA；精确 +0 的均值加法和默认规范化可另作同 tag 上的增量。每域默认值须沿原 `0×scale+bias` 运算生成，不能擅自只拿 bias。位等价目标是既有 FP32 Engine；它本身与 CUDA 仍有约 4e−6 规范化差异，不能顺带宣称 CUDA 位等价或沿用 AEE。

**一个尚未试的严格接口。** 本次只读检查发现：ordinary/lifting 的 750 个 256 值叶中分别 **209/210 个整叶为 code0**，所有被证明的零均为 +0。μ 就绪后，可以用**原 64 次 FMA 从 +0 出发**生成每 H8 的完整默认支路，再只在完整零叶上加载四份相同支路、照旧树归约。这样复用的是同初值、同次序的完成支路，非计数乘法；缓存生成、384 B 状态、读写、tag 检查必须收费。先只试完整叶，混合叶/任意次数不扩展。

**立即执行与裁决。** `default_bn/` 独立比较 dense、保序 tag 压缩、同 tag 上均值/默认规范化省算；输出仍全量物化并收费。随后唯一增加完整零叶支路复用。完整两学生、原 μ/var/rsqrt 和全部 18,432,000 输出逐位比较，另用一个固定背压。相对最强 tag 控制才算增量；若仅省传输，该项保留公共实现。先不删除 PED 加法/原始 I24，也不把这项通用默认表达写成已证明的新颖性。
