# M1584｜M1579 one-shot / TOCTOU 修复增量独立 QA

日期：2026-09-01（Asia/Shanghai）  
裁决：`NO_GO_M1584_M1579_CPU_PRODUCTION__RELEASE_SNAPSHOT_LACKS_PRE_READ_REGULAR_FILE_GATE`

本次只审 commit `2b53c147` 相对 M1581 的 release 增量，没有重跑模型数学，
也没有执行 51.84M 行 production。

主体修复是有效的。release 的 parsed value 与 SHA 来自同一 opened byte snapshot；
replay 中改写 pathname 后，attempt marker 和结果仍绑定最初 verified SHA。marker 使用
`O_EXCL`，tiny fixture 证明它在 materialization 前已经存在。成功后归档 output 再
执行会以 `FileExistsError` 拒绝；materialization 主动失败后 marker 仍保留，retry
同样拒绝。output/ledger 精确绑定、ledger 必须位于 output、workers≤3 均未回归。

但 secure-open 仍差一个必须的类型门。`read_release_snapshot` 在 `os.open` 前没有
对 named inode 做 `lstat + S_ISREG`，在读取前也没有对 opened inode 做
`fstat + S_ISREG`。独立 mutation 创建 FIFO release 后，子进程在 JSON/attempt gate
之前阻塞；这会让唯一生产命令无限挂起且尚未留下 consumed marker。目录和 symlink
最终会拒绝，但不能替代 pre-read regular-file gate。

最小修复是：open 前要求 named path 为 regular non-symlink；使用 `O_NOFOLLOW`
打开后，再要求 opened fd 为 regular 且 dev/inode/size 与 named identity 一致，随后
才读取字节。除此之外，2b53c147 的 snapshot SHA、O_EXCL、失败消费、二次拒绝、
≤3 workers 与路径绑定都应原样保留。

因此 M1584 仅授权这一行级 secure-open successor，不授权 exactly-one CPU
production。修复并通过新的独立增量 QA 后，才可授权一次 ≤3-worker 生产执行，
其结果仍必须独立 QA 后才能引用。
