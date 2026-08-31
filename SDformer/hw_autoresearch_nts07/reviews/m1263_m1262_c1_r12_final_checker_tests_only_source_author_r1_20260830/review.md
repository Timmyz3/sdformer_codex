# M1263｜R12 最后一轮 checker/tests-only source GO

结果：**PASS 100/100，P0/P1/P2 = 0/0/0；等待独立 hammer，不授权 release/VCS。**

R12 TB 保持 `e13d630f...ad302`，M528/M935/M1162/SVA 与 `docs/359` 全部冻结。只新增最终 checker 与 tests：

- phase/PASS 只从可执行 `$display` 调用中提取，普通 string 与 `$fatal` 不再计数；
- phase/PASS 的首 token 必须精确相等，`_SHADOW` 被拒绝；
- force/release 按 task、语句顺序、重复次数、target 与 RHS 全量精确比较；
- 未授权 task 或 task 外的任何 force/release 均被拒绝。

原 18 tests 与新增 12 个近邻攻击合计 30/30 PASS。M1262 的六个 unexpected accepts 全部转为 reject。此包仅允许一次 fresh independent source hammer；通过前不得制作 release，不得启动 VCS/EDA，也不再扩展 checker 范围。
