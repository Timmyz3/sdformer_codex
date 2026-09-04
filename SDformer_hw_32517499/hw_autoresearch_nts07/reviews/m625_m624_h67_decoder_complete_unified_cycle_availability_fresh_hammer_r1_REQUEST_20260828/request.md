# M625 fresh hammer request：M624 decoder-complete unified cycle availability

请由新 reviewer 对 M624 做只读独立打铁。目标不是产生性能数字，而是确认它是否正确地在数据/执行语义不足时拒绝运行统一 simulator，并给出了最小可执行补数清单。

PASS 门：score >= 95，P0=0，P1=0。允许 CPU 只读复算与临时目录重放；禁止 GPU、EDA、远端、生产 simulator 和对 M624 文件的修改。

必须独立检查：

1. M624 contract/result/receipt/manifest/outer seal、analyzer SHA 和 `docs/359` SHA。
2. ordered trace 恰为 10 samples、1840 rows、790 operator、930 ATLIF、120 attention、0 ConvTranspose；79 个 operator module 与 160 Conv2d/630 Linear 行算术一致。
3. M51 manifest 恰为 310 records；本地 present/missing 为 160/150、748,800,000/564,480,000 bytes，所有 present payload SHA/size 零 mismatch；缺失项为 Linear 140、Conv2d 10。
4. M511 期望 40 records/87,030,000 bytes；M578 四 tensor 合计 7,140,096 int8 bytes；decoder 全局 order extension 需要 40 metadata rows。若评审时新 M511 包已经出现，把它记为 M624 后续进展，不得反向把 M624 的 null 性能列改成结果。
5. M590 r6 仍受 M596 `P0=3/P1=2/formal_cpu_execution_allowed=false` 禁止；M510 只允许 analytic projection；M522/M523 只允许 support evidence。
6. B0/B1/B2/B3/Ours 五行的现有数据路径与 blocker 是否准确；尤其不得把 M216/M518/M519/M528 切片相加或相乘。
7. 所有 cycles/traffic/stall/fixed-numerator/speedup 字段必须为 null，`cpu_simulator_runs=0`，GPU/EDA/remote=0。
8. 评价 R1--R6 是否为最小且足够的下一步；不得借 fresh review 启动 capture 或 simulator。

通过后只授权：接收/生成缺失包、另行修复统一 CPU source，再建立新的执行 contract。M625 本身不授权性能运行或论文 headline。
