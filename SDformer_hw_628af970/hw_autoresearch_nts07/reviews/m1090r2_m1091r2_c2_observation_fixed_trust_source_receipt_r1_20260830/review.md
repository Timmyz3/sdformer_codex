# M1090r2/M1091r2：C2 observation fixed-trust source receipt

M1090r2 保留 M1090 已通过结构审阅的 22 路纯观察 wrapper 与 128-cycle first-X TB，但使用全新的模块、filelist、contract、release、engine、attempt/result/work/quarantine/lock namespace。旧 M1091 runner SHA 保持 `fade26d...`，因 M1092 STOP 明确列为 DO-NOT-RUN；旧、新 attempt 均不存在。

M1091r2 engine 已删除全部 caller-selectable expected-hash 环境变量。它在源码中硬编码 M1090r2 contract/release、M1092 STOP、M1088、M1080 attempt/quarantine、docs/359，以及 21 个项目输入身份。Python 3.10、VCS、lmutil、slow/fast DB、cell model 均要求 `lstat` 为 regular 且 SHA 精确；`dc_shell` 是唯一例外，只接受固定文本 `snps_shell` 的 symlink，并继续校验 regular target SHA。SDC、DC TCL、memory model、TB、filelist 及 filelist RTL 全部逐项 regular/direct-symlink reject 与 SHA pin。

为避免 contract↔engine 的循环哈希，contract/release 先双封，engine 再硬编码二者；最终 engine SHA 由本 receipt 给出，并必须由 M1093 独立 hammer 与后续 launcher 绑定。engine 只接受固定 argv `--authorized-launch`，而且先要求固定路径 launcher、双封 launch receipt、M1093/M1096 评审链与真实 parent launcher，再允许到达 `ATTEMPT.mkdir()`。这些未来 launch 文件当前故意不存在，因此 source 阶段直接调用必在 attempt 前关闭。

本作者仅用 Python 3.10 对源码做 AST/JSON/hash/lstat/DAG 顺序检查，没有导入或执行 engine，没有运行 DC/VCS，也没有创建 attempt/result。下一步只授权不同作者 M1093 审 engine；M1093 通过后才可由另一作者生成硬编码 engine SHA 和 M1093 outer 的 launch wrapper，且仍须 M1096 hammer 后才能执行。
