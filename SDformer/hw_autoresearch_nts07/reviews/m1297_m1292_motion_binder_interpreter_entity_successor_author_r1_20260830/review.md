# M1297 author-side interpreter-entity/TOCTOU successor

结论：**source-only PASS，12/12；等待异作者 receipt-blind hammer。**

M1297 冻结 M1292，只修解释器实体和 prepare-to-exec TOCTOU。生产常量完整绑定远端 `/usr/bin/python3 -> /usr/bin/python3.12` 的 dev、inode、mode、size、mtime 秒、SHA-256、Python 3.12.3 与 memfd/all-seals 能力。

解释器在任何 checkpoint/artifact snapshot 之前，经 realpath + parent dirfd + `O_NOFOLLOW` 打开并保留 FD。版本和能力由 `/proc/self/fd/<fd>` 启动的该实体自行测量，不再接受调用者标签。不可逆 O_EXCL attempt 前再次核验 logical path 与保留 FD 的完整身份；attempt 内容绑定解释器实体摘要，child 通过同一保留 FD 执行，因此 attempt 消耗后不再按 `/usr/bin/python3` 路径重开。

合成测试覆盖字段/类型漂移、symlink retarget、FD-bound command/pass-fd、attempt identity digest、persistent O_EXCL、失败无重试、零参数生产入口。未连接远端、未消费生产 attempt、未选择 checkpoint，也未运行 GPU/EDA。

本回执不授权传输、远端 preflight 或 one-shot execution；必须先由不同作者对精确 source/test/contract 做盲打。

