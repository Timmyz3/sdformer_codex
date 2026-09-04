# M565｜M559 PBR4 immutable CPU analyzer/runner source author handoff

日期：2026-08-28  
状态：`SOURCE_ONLY_AUTHORED__RUN_AUTHORIZED_FALSE__FRESH_STATIC_HAMMER_REQUIRED`

M562 的 100/100 只允许 N2 immutable runner source admission。本交接因此只新增 Python analyzer、shell
runner、source contract 与 fresh static request；没有建立 N4--N8、没有读取真实 M511/weight payload、没有
创建 result/attempt，也没有运行 RTL、EDA、训练、GPU 或远端。

Python 源码冻结四个 exact point（A1-SC8/A1-ISO8/A1-OSG/PBR4）、literal T10/block-outer 顺序、typed
binary `+1`、M523 phase-major bundle、M218 六 slice resident-hit sequence、固定 A1-STRONG、公共 terminal
FSM、closed result/failure schema。输入 bitpack 是 seek+chunk 的 little-bit-first 流式扫描，不把 926.88M-bit
replay 展开进内存。

N0--N9 DAG 保持单向：runner 只知道 N6/N8 canonical path，不含后生 SHA；未来 N6 必须绑定本次两份 source
SHA 和 N3--N5 的既有字节，N7/N8 再完成 wrapper self identity。直接调用必须在 result/attempt 前因 read-only
FD、parent PID/starttime/cmdline 或 terminal wrapper review 不匹配而拒绝。

本 author 只运行了 Python 3.6 内置小 golden（两份 terminal SHA、M523 9-tap/8+1 bundle、SC8/ISO8
18-cycle resident-hit）与 `bash -n`。它们不是正式 analyzer run，也不形成性能结果。

下一门是 fresh independent source static hammer；P0/P1 必须为 0/0。即使 PASS 也不授权执行，只允许后续
N4 launch-candidate review authoring。
