# bitmap mode10 增量静态复审

2026-09-14。仅补审 mode10 相对已核 mode13 的 onehot 路径，原 `bitmap_pipeline_review.md` 保留。未运行实验或改实现。

**未发现功能错误。** 对 16K bitmap 只有一个置位的块，原有三 plane 计算精确等于读取该 k 的 Q1 signed3 向量，加到共享 bm_acc。

- `BM_SCAN` 在当前 bitmap 字上判断非零且 b&(b−1)=0，同一拍锁存 bm_hold。下一 `BM_NATIVE_QREAD` 使用 hold 的位号，地址=16×bm_block+onehot_index；首末块、bit0/15 均在0..863。
- weight_allow=0 时留在读取状态，bm_block/bm_hold/q_hold 不改变，地址保持；批准后仅锁存一个24 bit Q1向量，再进入 BM_NATIVE_ADD。
- BM_NATIVE_ADD 设置 lhs=bm_acc、rhs=Q1 的32bit符号扩展；sub_alu=0且不启用 ZADD 的13bit carry cut，经原8条32bit链完成精确加法，signed3=−4 正确。
- BM_SCAN 已清 pipe_valid；onehot路径不会执行遗留 pop。BM_NATIVE_ADD 提交 bm_acc 后才通过 bm_next 前进；若为最后live块，下一拍 BM_STORE读取更新值，pending只清当前块一次。
- mode10 与13共用完整源构造、live元数据、同一bitmap/q_mem/q_hold/bm_acc、原z/psum接口，无新数据数组/乘法器/完整加法器。新增的是onehot判断、位号编码、Q1地址与共享ALU输入选择，另有统计计数器。
- 当前 q_mem 分别在普通 QREAD 和 BM_NATIVE_QREAD 两处读，状态互斥且写同一 q_hold，RTL语义每拍至多一次读，可用地址mux实现单读口；代码尚未显式收敛为唯一读表达式，不能从静态状态互斥直接保证综合后的SRAM端口数。bitmap onehot判断使用与BM_SCAN相同地址，未要求额外不同地址读。
- 无BP时每个onehot块把 mode13 的4拍 BM_PIPE替换为1拍原生权重读+1拍加，共省2拍；原BM_SCAN仍收费。Q1流量需分清24bit native读和128bit plane读，不能直接把weight_words当字节数。
- first_issues 合计native+pop；bitmap_native_reads/issues独立记录，aux_issues仍仅pop。汇总不应把aux_issues下降直接当全部算术消失。

该路径是源局部密度驱动的精确执行选择，不是新的乘法表示。验证与资源结果以负责人随后收口的实际记录为准；这里不把静态审阅写成已通过动态测试。

证据：`bitmap_pipeline/decomp_core.sv` 的onehot组合逻辑、BM_SCAN、BM_NATIVE_QREAD/BM_NATIVE_ADD以及共享ALU输入选择；路径相对 `consumer_transfer_20260914/`。
