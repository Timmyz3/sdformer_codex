# M914：M518 r4 Fixed 解释器调用静态打铁

结论：**PASS 100/100，P0/P1/P2 = 0/0/0**。本审阅没有运行 runner、DC 或任何许可证查询。

冻结 runner 的权限是 `0664`，因此原来的直接路径命令会由内核在进入脚本前以 126 退出；该失败不消费 attempt，也不产生 preflight、work、canonical 或 quarantine。以固定 `/usr/bin/bash` 将同一 runner 文件作为脚本参数解释是最小修复，且无需修改文件内容或权限。

这不会绕过 runner 的安全合同：`BASH_SOURCE[0]` 仍指向冻结 runner，runner 自身的 `realpath`/SHA pin、Fixed admission pin、一次性 attempt、三样本资源门、全 UID EDA 冲突门、运行期监控、失败隔离和最终发布条件都保持原样。

唯一授权命令记录在 `review.json` 和 M914 overlay 中。运行前仍须重新确认 `/usr/bin/bash`、runner 和 admission 的 SHA，确认 C1 队列终止，且所有 M518 Fixed attempt/result/work/preflight/quarantine 路径为空。原 M905 的总 attempt 上限仍为 1；M914 不是第二次尝试授权。

边界：这里只准入调用形式，不准入 DC 面积、时序、PPA、能耗、系统加速或论文 headline。
