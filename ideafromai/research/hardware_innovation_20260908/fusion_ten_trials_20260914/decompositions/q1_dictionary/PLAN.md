# D3 Q1重复向量静态字典：实现前合同

B：不同K可能具有完全相同8rank signed3列。固定选出现最多、至少2次的前32个非零向量（平频按24bit码升序），编译每k的class与代表列索引；其余K直接走原双P更新。对字典类先累加非负整数源计数，类退休才做 `z += Q1_rep * count`，与上一轮动态z同幅分组完全不同。

完整A：静态weight repetition/CSE与UCNN式sum-first归约。强控制14为最新native窗+双P packed/cachedOS；15用相同资源的32类有限字典。额外8bank×96row×20bit计数（两个独立unsigned10字段，最多864）、字典metadata、8×32bit ALU可carry切在10或13、8×19×13乘及所有读口两側共同预算。计数一次按同k的8个P/T pair行更新，最多三块覆盖20行，费用包括块扫描/读/写；没有无穷并行归约。

为每个实际字典类清3个八bank行，源k类别lookup并已计静态cfg载入。被选类的count输入累积按bitmask更新，原生source/window/gather照旧；类活性随真实输入产生。退休读代表Q1、扫描3个计数向量，逐个非零P/T count用共享乘法及z读改写完成，全K之后仍最终z支持scan与完整Q2 cachedOS。完整psum和输出背压不减义务。

失败数：重复列多但同k的T/P稀疏时，三块计数调度和类退休可能抵消weight reuse。此固定32类端点必须实跑，实际字典较小时按真实类数计clear/retire；不扫描容量。14旧fixture112runs，无有损训练。
