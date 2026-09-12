# 唯一放置替代：跨行紧凑RF表

固定原5+5、64行/行32B的coefficient image，不重训、不改D/c或表划分。RF16..95缓存全部640个signed24表scalar，共80向量；RF0..9是T10输出，RF10存8个门字，共91个活RF。原源程序/其它消费者在该decoder叶中不并发，不能将剩余5RF解释成可放完整PED。

不新增gather ISA。不同lane的索引先按原8bit predicate分组，仍逐组串行执行。对一组的一个t，现有 `IPRED_MASKED_CACHED_ADD` 读取一个acc向量和一个表RF向量，从该表向量选一scalar广播给匹配lane；两RF读、一issue。所有lane不同时读取不同RF地址。

本次采用明确的单步地址控制：每个t为 `(row*10+t)` 计算RF地址/内lane，单独收一个controller slot，然后发已有加法。冷装表从真实CR32B行取数据，越过2B padding时分段收集到原24B暂存，每段收费，再用已有packed24解码写80RF。它不是免费cache预置，也不增加非线性算术或任意跨lane读取能力。

同原四窗及ordinary/interior固定压力；与既有完整常量缓存seq比较，保留先前结果。这个控制较严格；若后续将索引增量并入共同控制器，需要新实现/明确共同权限，不能在本轮把该费用悄悄抹去。
