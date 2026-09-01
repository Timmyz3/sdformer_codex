# M1630｜C1 residual-hold guardband DC source 作者交接

状态：**SOURCE-ONLY PASS；必须先经 M1631 不同作者审阅，当前不授权 DC 或 attempt。**

M1614 已把 C1 从原 M993 的大量 hold 违例收缩到 3 条，最终 setup WNS `+0.001718520 ns`、hold WNS/TNS `-0.000353523/-0.000401557 ns`、面积 `152834.995973 µm²`、DRC 0。M1630 不使用该失败网表；它重新读取 exact M993/M1006 原 DDC，只在唯一次 hold-only incremental mapping 时把 hold uncertainty 从 `0.050 ns` 临时收紧到 `0.051 ns`，随即恢复 `0.050 ns` 再生成所有最终报告和 SDC。

冻结点保持 `3.000 ns` clock、`0.200 ns` setup uncertainty、9 个 exact SRAM macro、ideal clock 与 `ZeroWireload`。无 false/multicycle/min/max-delay exception、disabled arc 或 case analysis；无 generic/ultra/second-pass/retry/降频。`current_design` 用 `get_object_name` 比较，DC 启动冻结为 `-no_home_init -no_local_init -no_gui`，不重定义 `HOME`。runner 对 `Error:/Fatal:/LINK-*/unresolved/TIM-209/OPT-150/loop` 任一证据 fail closed。

Python 3.6.8 和 Python 3.10 均为 15/15 PASS，`bash -n` 和 contract 内外封印通过。M1630 result/attempt 路径均不存在；本次没有执行 DC/VCS/PT/Formality/PTPX/GPU，也没有改 `docs/359` 或 `ucli.key`。

下一步只授权 M1631 对 exact Tcl/runner/test/contract 进行不同作者静态 mutation 审阅。只有 M1631 P0/P1 为零后才可封 M1632 one-shot release；未经这两道门不得创建 attempt 或运行 DC。
