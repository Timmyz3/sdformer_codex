# M1118 Table-A component annex r12 作者交接

结论：**r12 作者验证通过；只授权不同作者做静态打铁，不授权 EDA、production 或全系统 Table-A 行。**

r12 没有修改 M910，而是在其已封存 C2 行旁新增两个强类型组件证据行：

- C1：M1114 准入的 frozen H67 四层 bottleneck Conv、10-sample、812160-task raw CPU same-ledger 机会。候选 `434242823` cycle，对 strongest-zero 和 same-coordinate bit 均为 `763908050` cycle，即 `1.7591725401987818×`。`214912 B < 245760 B` 只表示容量账本算术；它不是 RTL/mapped-gate speedup，不是 SRAM 宏 PPA，不绑定 final checkpoint，也不是 system speedup。
- C3：M928 准入的 Fixed-T10 logic-only pre-macro DC setup/area 点：TSMC 28 nm、3.000 ns ideal clock、ZeroWireload、0 macro、`62433.503388 µm²`、最差已报告 setup slack `+0.0003 ns`。hold 未闭合，也没有 PT STA、power、energy、throughput、speedup 或 system 证据。

继承的 C2 行逐字来自 sealed M910 preview。最终 annex 有 3 个 component row，但 full-system Table-A production row 仍为 0。

作者验证通过 21/21 单元测试和 8/8 定向攻击；duplicate key、NaN、C1 RTL/final-checkpoint 升格、C3 第二新增行面积/速度篡改、全系统行升格及额外组件行均 fail-closed。没有运行 EDA/GPU/remote/production。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
