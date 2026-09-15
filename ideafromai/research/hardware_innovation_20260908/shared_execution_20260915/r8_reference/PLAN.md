2026-09-15。只写本目录；旧bitmap_rr和R8数据只读。先核图边与前缀，再重算R8 gold，最后在共同接口与资源许可下重跑2/native4P、4/borrowRR、21/count、7/bitmap强控制。它们均为已冻结flatR8函数，不能用空间phase3的raw/I24作gold；同源但不同有损函数的质量另列。

两网络评估器实际导入同一data/model_access.py，加载同matched-dense前缀、相同r0.conv2.0及r0 identity。改动仅在该conv2和r0输出之后；空间初始输入直接来自旧R8全帧捕获。prepare.py独立验证首帧seq128/9664 source/id及R8输出，与旧fullgold一致，新的36tile只借原始source/id，按旧q1/q2/a/b计算R8 latent/raw/J/wide/I24并用expanded-W核raw。

以旧bitmap_rr唯一8×32 ALU/8×19×13mult、唯一consumer8×64链/8×32×32mult及原子RR为底座，保留mode4真实宽链争用。只改变外部fixture加载与存储许可，不改计算枚举/仲裁。每context Z实际声明8×40×52=2080B，总4160B，仍一个真实416bit服务；R8原算法使用前10行。cache扩为24×8×16=384B容量（R8使用前8向量），和空间共享这份上限；允许同holding预算，unused容量不能当PPA面积实证。静态W/plane/class及bitmap/pop等额外项全部单列，不能只比较ALU数量。

统一source_tile15bit＋source_address11bit(local0..1535)请求，全部1536个源字都真实加载，含padding/poison；核内依据独立origin过滤边界。ORIGIN有单独valid/request、source物理原点32bit（高sx/低sy）、origin_stalls计数。TB虚拟tile索引与真实origin无关，方便连续换序列/row；每batch最多两context加载完再同时launch，整个batch所有I24退休后回收。cold模型一次加载、warm保留模型，source/origin逐tile付；原count class/rep/固定排列加载仍计费，不重校准。

BP沿旧parameter/source/identity/compute source/W/output日历，新origin n%17为4/5时拒绝；停顿时所有请求地址与输出保持。初始小集，再两64与18seq36tile，两遍无reset，额外跨mode/非连续source检查。raw/J/wide/I24实际RTL逐值验证，独立源profile核工作量和全部服务恒等式。源码只从旧只读复制到本目录，Verilator4.028 --cc --exe+make；无GPU、EDA、训练、生产/Git。
