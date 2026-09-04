# M915：M518 r4 Fixed 子进程碰撞误杀独立法证

结论：旧 r4 Fixed 身份维持 **FAILED_OR_INCOMPLETE_DO_NOT_CITE**。失败不是资源不足或外部 EDA 撞车，而是 runner 的进程树边界错误：运行期扫描只排除主 DC PID 3022873，把其直接子 worker 3206061 当成 external collision。两者 UID、可执行文件和 NUL-safe 命令行完全相同，且 `runtime_7` 明确记录 `PPID=3022873`。

monitor 因此终止主 DC；worker 被重新挂到 PID 1 后继续写 `dc.log`，产生 EPIPE 尾流。旧 runner 在该 worker 排空前就封存 quarantine，所以当前 `SHA256SUMS.seal.sha256` 仍能校验历史 manifest 本身，但历史 manifest 的成员校验恰有一个失败：`fixed/dc.log` 从历史 `1a527e...` 变为当前 `e97921...`。M915 另行冻结当前完整树，绝不回写或“修复”旧 quarantine。

所有八个 runtime 样本的 commit headroom、MemAvailable、SwapFree 和 cgroup OOM 计数均通过原门；前六个 gate 为 none，第七个唯一触发项就是误判的 child。r4 attempt 三件套双封有效且已消费，不能复用原身份。DC 启动还记录了缺失 `HOME` 导致的 Synopsys GUI 初始化错误，虽非本次 rc11 主因，后继必须显式使用工作区内私有安全 HOME。

只允许一个 additive Fixed-only 后继：新 canonical/attempt 身份；collision 扫描排除经 `/proc` 父链证明属于精确 DC root 的所有子孙；DC 用独立 `setsid` job 启动，任何失败都在封存前排空该 session/process-group；显式 `HOME=<work>/safe_home`。原 RTL、Tcl、filelist、SDC、库、3 ns、单次 `compile_ultra`、无 hold fix、logic-only 和所有 claim 边界不变。该法证本身不授权 DC/VCS/PT/Formality/PTPX 或许可证查询。
