# 真实一页结果：精确连续数据codec机会（2026-09-12）

**结论：I24普通块位宽已吃掉几乎全部本轮codec增量；FP32默认值＋指数编码可省字节，但只是更强普通底座，未证明标题级X。** 未用GPU/EDA、未改生产树或模型、未改BN算术。

数据：真实 `000_zurich_city_09_a_0001` 一帧，ordinary/lifting_raw两臂。FP32是capture的全域projection BN输入/输出 `10×96×120×160`，以及corner/interior preview切片；I24是权威capture_full_producers的 `full_continuous_q24`，signed24/f14，真实生产者布局spatial,T,C，并另列native布局及小source窗口。I24未重新量化：按24位打包作为原分母，不能拿NumPy int32文件当32位流量。

| 张量 | 原始B | 强简单对照B | 另一mode/组合B | 全普通mode选择B |
|---|---:|---:|---:|---:|
| ordinary FP32 BN输出 | 73,728,000 | fill 33,682,028 | fill+exp 29,472,273 | 普通全模式 29,209,312 |
| lifting_raw FP32 BN输出 | 73,728,000 | fill 33,489,620 | fill+exp 29,282,468 | 普通全模式 29,018,745 |
| ordinary I24 PED | 55,296,000 | signed-width 42,263,200 | signed-delta 43,072,616 | 普通全模式 42,219,037 |
| lifting_raw I24 PED | 55,296,000 | signed-width 42,156,016 | signed-delta 42,975,088 | 普通全模式 42,110,467 |

FP32 fill+exp相对fill再省12.50%/12.56%，但全普通mode选择还省约0.89%/0.90%；这不是一个超越普通组合的X。I24全模式相对块signed-width仅省0.1045%/0.1080%。I24在native布局另有数值，但转换并非免费，不能跨布局挑最好数当已部署增量。全FP32源输入两臂也已测，见JSON。

实现：64值一块；模式/位宽字节、基值、默认值、指数基、掩码、escape原字及byte padding计入。FP32保留1+23符号/尾数位和完整指数，NaN payload、±0也可还原。FP32共43,008个mode-block、I24共20,480个mode-block实际序列化/解码为0 bit mismatch；另23个边界组合通过。全域费用是从每个真实块的整数长度计算，不把抽样压缩率外推。编码只支持顺序流，没有免费随机访问索引；如每块加32位索引，两个全域张量每臂各另加1,152,000B。FP32当前帧通道直方图/default选择是乐观编码端扫描，未给其免费实时预测身份。

边界：这是CPU功能/字节机会测试，非端口周期/面积/功耗，也未重跑AEE。只要原字按位恢复并保持消费算术，表示本身不改函数；尚未接入真实消费者。未完整移植ZipServ/Atalanta/Shannonic/EBPC，故也不能声称打赢这些原作。未知编码/解码/双消费者缓冲费用可能吃掉字节收益。更直接的本地反证是：已实现ordinary BN→PED融合不物化BN输出，针对该输出的当前新增可避免流量为0；离线capture大小不是融合后仍在搬运的字节，不能为codec恢复已删物化。

复现：`/opt/anaconda3/bin/python exact_codec_probe.py` 与 `... i24_codec_probe.py`。数据源路径、所有12个FP32和8个I24结果在 [exact_codec_results.json](exact_codec_results.json) / [i24_codec_results.json](i24_codec_results.json)。脚本与两份run.log、[edge_roundtrip.json](edge_roundtrip.json)同目录。主试验实际wall约10.72s与5.01s，仅用于复现，不当硬件性能。首跑系统Python3.6/缺numpy的系统3.12后已改用现有Anaconda3.12，不做环境安装。

下一决定：停止把I24自适应codec当主创新；把FP32普通fill+exp保留为所有候选共有的强对照。只在普通融合后仍必须spill的其他源上证实事务和decoder预算，才考虑孤立codec RTL。已有算法精度优于NB0不能替代这些新表示的端到端验证。
