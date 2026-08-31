# M1090r3/M1091r3：C2 frozen-history validator repair source receipt

M1093 正确发现 M1091r2 把 live-input 的严格 direct-symlink 拒绝误用于历史 M1080 VCS quarantine，导致合法的 manifest-listed `_2931510_archive_1.so` symlink 在 launch gate 前必停。M1091r3 仅修复这一处：新增 `verify_frozen_history_flat`，且函数第一条边界要求参数必须等于精确 M1080 quarantine 路径。

历史 validator 校验 exact manifest、exact outer 和 manifest 列出的全部 113 个成员。唯一一个 symlink 只有在解析目标仍位于该 quarantine 内、解析目标为 regular、且通过 symlink 读取的 bytes 与原 manifest SHA 一致时才接受；当前 113/113、1 个 symlink 均通过。该函数不能用于 live source、tool、library、model、SDC、TCL、memory、TB 或 filelist。

所有 live identity gate 均从 r2 原样保留：21 项项目输入与 8 项外部身份继续 exact SHA + `lstat` regular/direct-symlink reject；`dc_shell` 仍是唯一明确例外，只允许 exact `snps_shell` link 并校验 regular target SHA。caller-selectable expected-hash env 仍为 0，固定 argv/launcher/receipt gate 仍在 attempt 前。future launcher 与 receipt 当前不存在，所以 engine 在 source 阶段仍不可能消费 attempt。

本作者只执行 AST/JSON/hash/lstat 与历史 manifest followed-byte 静态检查，没有导入或执行 engine，没有运行 DC/VCS，没有创建 attempt/result。M1091r2 保持 DO-NOT-RUN；下一步只允许不同作者 M1093r2 hammer 新 engine，不能直接生成 launcher 或执行 EDA。
