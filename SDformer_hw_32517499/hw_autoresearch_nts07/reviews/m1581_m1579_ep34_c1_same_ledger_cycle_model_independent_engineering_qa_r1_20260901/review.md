# M1581｜M1579 ep34 C1 same-ledger cycle model 独立工程 QA

日期：2026-09-01（Asia/Shanghai）  
裁决：`NO_GO_M1581_M1579_PRODUCTION_RELEASE__CORE_SAME_LEDGER_MODEL_ENGINEERING_PASS__RELEASE_REUSABLE_AND_TOCTOU`

## 核心模型：通过

QA 固定 M1579 source/test 与 M1524/M528/M505/M504 SHA，仅运行 source audit、
作者 6 项测试和缩小后的 synthetic fixture，没有解码生产 support plane，也没有执行
51.84M 行 replay。final ep34 checkpoint、capture manifest 与 ordered-record SHA
一致；生产几何静态绑定为 10 samples × 4 operators × 432 partitions × 3000
rows，即 51,840,000 行、466,560,000 ledger bytes。

缩小 fixture 精确验证 `support → sample,operator,partition → timestep,y,x`
行序，交换 operator record 会拒绝。zero/bit/product/all-write/dead-write/PVRF
均从同一个 ledger-derived array set 调用冻结 M528 `cycle_row`；worker recurrence、
M505/M504 transitive SHA 与 conservation 字段保持复用。synthetic ratio-of-sums 为
2.2222，而逐 sample ratio 的算术均值为 2.0833，证明二者没有混写；sample-major
和 operator-isolated 分布也分别保留。

output/ledger 必须同一 canonical result directory，workers>3 会拒绝。成功路径先在
output 同父目录 staging，写齐 ledger/result/两张 CSV/RUN_COMPLETE，完成双封后才
`os.replace(stage, output)`；篡改封存成员可被独立验证器发现。claim boundary 明确
是 operator-level CPU cycle model，不是 RTL、wall-clock、全网、系统倍速或能量。

## Production release：拒绝

`cpu_runs=1` 目前只是 release JSON 字段，不是可消费能力。QA 用 tiny synthetic
execute 发布第一份结果、把它归档后，同一 release 能再次成功发布第二份结果；源码
没有 canonical attempt marker、`O_EXCL` 或锁。更严重的是 release 在验证后到结果
写出前仍可被改写，M1579 最终记录的是改写后文件的 SHA，而不是已验证字节的 SHA。
因此它没有把一次 CPU 权限或 release 身份绑定到长时间 replay。

最小 successor 应在物化 ledger 前原子消费一次性 attempt，并把失败也视为永久消费；
release 必须作为 regular non-symlink immutable byte snapshot 一次读取、验证和哈希，
结果绑定该 pre-execution digest。修复后再做独立 source hammer，才允许唯一一次
51.84M CPU execution；生产结果仍须独立 hammer 后才能引用。

本 QA 未运行 GPU、RTL、VCS、DC、PTPX 或任何 production cycle replay，也未修改
M1579 作者文件。
