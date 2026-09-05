# M2241 技术评阅

结论：值得继续做小型消费者释放 RTL；目前的 1.1805× 只能称“逐 beat 流水候选相对现有整行填充 FSM 的周期建模收益”。还不能把它全归因于取消 row-cache 或新的生命周期机制。无需增加合同体系，补一个关键对照即可判清。

当前证据可信的部分：3,840 个基线轴/窗口对上既有 VCS；4320 个 cold G48 chunk 的 TSBG 1-row/4-row 均为 11,928,718 cycle、7,519,968 次 bank read。两槽候选为 10,104,872 cycle，read/update 数不变；多加一级响应寄存仍为 10,184,330 cycle。候选尚未经过 VCS，continuation chunk 没有独立的逐周期校准，故不能把全部 4320 chunk 称为 RTL 校准结果。

端口与寿命上可以实现。模型只有串行的八 bank refill，允许下一 beat 写另一 response slot，同时当前 slot 给一个 Acc24 更新口供数，没有免费增加 SRAM 并发读端口。现有 M803 可以在保持 `core_rsp_ready=0` 时固定当前响应并接收别的 slot；释放必须发生在最后一个有效 context 真正接受之后，不能在“SRAM 数据已到”时释放。需要保持 slot/generation/group/half/slice 和 pending-context mask 配对，空 consumer beat 也要明确退休。模型把 slot 计为包括正在消费的 beat，这一点正确。

数值边界需要用新小 RTL 的 scoreboard 补一遍。DSE 只比较 update 数，没有执行 signed Acc24 数值。按当前遍历，每个 context 内 group 仍递增，跨 context 交错本身不改变独立累加器；保持每个目标 lane 的 group/half 顺序、原来的 K8 signed fold 和一次最终 commit，即可沿用 G48 的绝对和界 98,304。不能仅靠“update 数一样”宣称数值等价。

最少缺的对照是 **TSBG row-cache + 相同两槽逐 beat streaming**，并将同一优化也给 ordinary。具体保持相同请求开始时刻、refill/consume 重叠、context 选择收费、响应寄存级数和 commit；row-cache 在 miss 时边填边经 bypass 消费，hit 时正常读 cache。现有 `ordinary_slots2` 完全不保留 row-cache，无法替代这个对照。冷的 group-major 顺序中，这个带 cache 的 streaming 版本应能达到相同周期；重复热窗口还会保留 cache 命中优势。若实测如此，新结构的卖点应是省 payload 复制/驻留状态和相关能量，1.18× 则归入通用 streaming 的消融。

还需注意，一个 group 只付一次选择费的候选与每 context 付一次选择费的旧 FSM 不完全同轴：11.929M→11.581M 的单槽收益已混合“提前消费”与 context 选择合并。关键对照同时统一这笔费用即可，不必再扩成大扫描。已有 1/2/4 热 group 反例很好，应保留；它说明本结论依赖当前 cold chunk 边界，不能外推真实跨窗口 locality。

建议下一步只做：①上述 streaming-cache 对照；②消费者持有/释放 RTL，在响应乱序、最后消费者 stall、generation 复用和全空 half 上核对数字与顺序。通过后再看综合。6144 B row-cache、1024 B 现有 M803 slots、1152 B Acc24 的容量账正确；“借现有两槽”不会自动消掉 M803 的其余六槽，也不等于已经获得面积/能量改善。
