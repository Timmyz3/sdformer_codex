# M1134C｜M1132C production-scale 第一性原理规模审计

结论：**当前 M1132C 三个全量 set 与 M1130C batch consumer 均 STOP；只允许另起 additive O(axes) streaming successor source。**

## 规模与当前资源

- 冻结 weight count：70,853,184/axis，三轴共 212,559,552 event。
- 审计快照：MemAvailable 420,684,592 KiB；Commit headroom 123,397,960 KiB，即 117.681 GiB。
- 当前 CPython 3.10.18：二元 tuple 56 B、四元 tuple 72 B、64-char SHA hex string 113 B、大整数 28 B、set entry 16 B。
- 1M live set 的实测大小与 3/5 fill、power-of-two capacity 模型逐字节吻合。

## 为什么当前链必停

每个 set 在 212,559,552 entry 下至少需要 536,870,912 slots，即约 8 GiB table；三个 table 合计 24.0 GiB。三类 key tuple 再占 36.425 GiB，结构下界已经 60.425 GiB。

当前 set 还会长期保留唯一 exact-ID 字符串和至少一套不同 ordinal 对象，因此 producer retained floor 为 **84.642 GiB**。M1130C 仅第一份 `rows` 的 object shell + dict + list reference 下界又是 **57.013 GiB**，两者合计 **141.655 GiB**，已超过 commit headroom **23.974 GiB**。

该下界尚未计算第二份 scheduled rows、M1130 的三类 identity set、每个 write 最多 8 个 occupied keys、event list、payload、allocator fragmentation 和输出封存。因此不是“可能慢”，而是现有 batch 形态在当前 commit 约束下不可准入。

## 时间下界/乐观代理

硬操作量至少包括 212,559,552 次 producer call、validate、sink、上游 ID 构造与 validate 内 ID 重算，以及各 637,678,656 次 set membership/insert。

100,000-event 受控 benchmark 在关闭 GC、无 downstream、小 set 条件下达到 105,152.8 event/s；外推 producer-only 为 2,021.4 s。叠加冻结 M1102 已测 494.2 s，乐观总计 **2,515.6 s（约 41.9 min）**。这不是 full replay 测量；cache/resize、rows、scheduler、validator、I/O 和 seal 都只会增加工作。

## O(axes) 能否保持 exact-once

可以，但必须同时成立：

1. 每轴 beat ordinal 严格连续且等于 next beat；
2. 每轴 transaction ordinal 是全局连续流；
3. 每轴 producer 顺序已按冻结 scheduler key 非递减；
4. exact-ID 仍按五元身份重算；
5. 17 字段 validate 不变；
6. 每轴终值严格为 70,853,184；
7. 对所有字段做无歧义 binary streaming SHA，并与独立封存 expected digest 比较；
8. sink 失败不得提交 ordinal/counter/digest/scheduler state。

这些条件把 duplicate、gap、reorder、transaction collision 和 forged ID 全部保持为 fail-closed。若顺序保证或独立 expected digest 缺失，则 O(axes) 不准入，只能用外存 sort/bitmap 或保留 O(N) 状态。

最小 successor 只能是新 namespace 的 streaming validator + streaming 1RW sink source，状态限于三轴 ordinal/counter/SHA 与 3×24 个 next-free-cycle；不得直接接 real hook。

本审计没有修改 M1132C/M1130C/M1016/M1102/docs359，没有打开 canonical row，没有运行 full replay、EDA、GPU 或 remote。
