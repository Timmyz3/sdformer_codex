# M1262｜M1261 R12 checker/tests-only 独立打铁

结论：**FAIL-CLOSED，92/100，P0/P1/P2 = 0/4/0。禁止 release 与 VCS。**

冻结身份均正确：R12 TB 为 `e13d630f...ad302`；M528、M935、M1162、SVA r3 与 `docs/359` 均保持冻结 SHA。M1261 作者包内外双封通过，原有 18/18 tests 复跑通过。`valid_shadow`、注释掉或复制 normal call、`claim=false` 加注释诱饵均能正确拒绝，普通注释诱饵保持惰性。

但独立近邻攻击发现六个意外接受，属于四类 P1：

1. checker 收集所有可执行字符串，却没有证明字符串是 `$display` 的实参。把 phase 放进普通 string，或把 PASS 放进 `$fatal`，仍会通过。
2. phase/PASS 使用 `startswith`，因此追加 `_SHADOW` 的非精确 token 会通过。
3. helper inventory 使用 `set`，重复同一 allowed force 会被去重后漏过。
4. 全局检查只限制 target 名字，没有限制 exact helper placement/cardinality；helper 外额外添加一个 allowed force 会通过。

这不是 TB、DUT 或 SVA 的功能失败，只是 release checker 尚不能抵抗近邻伪造。允许制作一个 additive checker/tests-only successor，只修四点：解析真正 `$display` 调用、精确 token、保留 statement multiplicity、限制全局 helper 位置与次数。R12 TB、DUT/M935/M1162/SVA 必须继续冻结；修复后复跑原 18 tests 和本包六个攻击，再做一次独立 hammer。不得继续扩 checker 范围。

本评审没有运行 VCS、simv、EDA、GPU 或远端任务，也没有修改任何受审源文件。
