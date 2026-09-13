# 本轮实际整网 AEE（探索性 diverse10）

共 32 个完整十帧运行、320 次前向，0 次新增训练更新。NB0=1.4546028611；dense父=1.1597366283；lifting父=1.1653433024。各臂相对自己的结构父比较。

同帧名、同有效像素已逐帧核对。新臂用共同粗头preds.2与FP64 EPE求和；NB0用原最终头与FP32归约。比较任务质量，不声称同网络执行或逐位同数值协议。第一帧参与部分候选的源统计校准；CSV另列剔除它的九帧均值。所有帧都是项目中已看过的验证样本，没有把十帧评估写成新的 valid825。

|批次 / 运行|AEE|相对对应结构父|低于NB0|
|---|---:|---:|---|
|aee / parent|1.1597366283|+0.0000000|是|
|aee / r0_spatial_nm8|1.1840618202|+0.0243252|是|
|aee / r0_flat_nm8|1.1633597872|+0.0036232|是|
|aee / r0_ordinary34|1.1750728784|+0.0153363|是|
|aee / r1_shifted24|1.1824568954|+0.0227203|是|
|aee / r1_shifted14|1.1735405844|+0.0138040|是|
|aee / r1_ordinary24|1.1646512257|+0.0049146|是|
|aee / r1_ordinary34|1.1784827774|+0.0187461|是|
|aee / spatial_plus_shifted24|1.2060245415|+0.0462879|是|
|aee / flat_plus_shifted24|1.1825259807|+0.0227894|是|
|aee / spatial_plus_shifted14|1.1739504090|+0.0142138|是|
|aee / flat_plus_shifted14|1.1637546217|+0.0040180|是|
|count_aee / r0_signed_pair|1.1633027787|+0.0035662|是|
|count_aee / r0_unsigned_pair|1.1512492510|-0.0084874|是|
|count_aee / r0_flat_nm32|1.1909979110|+0.0312613|是|
|r0_sparse_aee / r0_ordinary24|1.1591833472|-0.0005533|是|
|r0_sparse_aee / r0_ordinary34|1.1860722596|+0.0263356|是|
|r0_sparse_aee / r0_shift14_magnitude|1.1777446144|+0.0180080|是|
|r0_sparse_aee / r0_shift14_request|1.1762471512|+0.0165105|是|
|r0_sparse_aee / r0_request_plus_r1_ordinary24|1.2036039540|+0.0438673|是|
|r0_sparse_aee / r0_unsigned_plus_r1_ordinary24|1.1838750236|+0.0241384|是|
|control_aee / r0_shift14_gram|1.1949790382|+0.0352424|是|
|control_aee / r0_pair_h8|1.1966744037|+0.0369378|是|
|control_aee / r0_ordinary34_h8|1.2144504382|+0.0547138|是|
|control_aee / r0_flat_svd_r8|1.3479650409|+0.1882284|是|
|control_aee / r0_activation_svd_r8|1.4030194175|+0.2432828|是|
|control_aee / r0_spatial_r16|1.2691297866|+0.1093932|是|
|control_aee / r0_tucker_r8|1.4170152223|+0.2572786|是|
|temporal_fusion_aee / lifting_parent|1.1653433024|+0.0000000|是|
|temporal_fusion_aee / lifting_r0_ordinary24|1.2099646387|+0.0446213|是|
|temporal_fusion_aee / lifting_r0_unsigned|1.1820587633|+0.0167155|是|
|temporal_fusion_aee / lifting_r0_request|1.2227245036|+0.0573812|是|

融合组合均实际重新运行，没有把单项质量或周期比相乘。新的融合不因小幅相对父退化淘汰，但过 NB0 本身不足以证明优于普通2:4/3:4，也不代表硬件费用通过。
