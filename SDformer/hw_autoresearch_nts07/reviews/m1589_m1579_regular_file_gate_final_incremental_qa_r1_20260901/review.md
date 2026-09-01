# M1589｜M1579 regular-file gate 最终增量 QA

日期：2026-09-01（Asia/Shanghai）  
裁决：`PASS_M1589_M1579_FINAL_RELEASE_GATE__EXACTLY_ONE_51840000_ROW_CPU_PRODUCTION_AUTHORIZED_MAX3_WORKERS`

本次只审 commit `842da3aa` 对 M1584 唯一 P0 的修复，并用 M1584 已封 hammer
重放 one-shot/snapshot/path 最小回归；没有执行 production replay。

release 现在先 `lstat` named path，要求 regular non-symlink 后才 `os.open`，并保留
`O_NOFOLLOW`；打开后要求 `fstat` 仍为 regular，再次 `lstat` current pathname，
named/current/opened 的 dev/inode/size 三方一致后才读取。独立 inode-swap mutation
被拒；FIFO、目录和 symlink 均立即拒绝，不阻塞，也不生成 attempt marker/output。

M1584 主体修复没有回归：parsed value 与 SHA 来自同一 opened byte snapshot；
attempt marker 在 materialization 前以 `O_EXCL` 创建；成功后第二次执行拒绝；
materialization 失败后 marker 仍保留且 retry 拒绝；replay 期间改写 release pathname
不会改变 attempt/result 绑定的 verified SHA。output/ledger 绑定和 workers≤3 仍有效。
M1581 已通过的 same-ledger/M528/ratio/distribution 核心模型结论按固定 SHA 继承。

因此授权一份新的、精确的 regular non-symlink release，在 fresh canonical attempt
namespace 下执行唯一一次 51,840,000 行 CPU cycle-model production，最多 3 个
worker。第一次成功或失败都会永久消费该 attempt，不允许自动或人工 retry。

生产结束后必须另做独立 result QA，核对 final ep34 身份、ledger 行数/行序、cycle
conservation、ratio-of-sums、分布、路径与双封。通过前不能把结果写成论文数字。
授权仅限 operator-level CPU cycle model，不代表 RTL、wall-clock、全网、系统倍速、
能量或 PPA。本 QA 未运行 GPU、RTL 或 EDA，也未修改作者文件。
