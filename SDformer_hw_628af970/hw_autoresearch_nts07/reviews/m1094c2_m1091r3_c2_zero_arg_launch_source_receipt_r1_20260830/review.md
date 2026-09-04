# M1094C2：M1091r3 C2 零参数 launcher 作者收据

结论：**只冻结 launcher 源码和 launch receipt；必须由不同作者完成 M1096r2 final hammer，当前不得执行 launcher、engine、DC 或 mapped VCS。**

launcher 固定在 engine 已声明的路径，接受零参数，硬编码 engine SHA `41b78990...af04`、engine source receipt outer `8bc6f725...edd9e` 与 M1093r2 outer `d6fa5ecb...f9cc`。它不会从调用者 argv 或环境读取 expected hash、路径或授权值；child 环境从常量新建。最终命令使用 `env -i` 清空 Python 启动环境，同时不在 Python 与 launcher 之间插入 `-I`，从而保持 frozen engine 要求的父进程坐标。launcher 启动 child engine 时才使用 `-I`。

作者静态自检通过 73 项，包括 AST、JSON、hash、lstat、双层 seal、sealed predecessor、精确 child argv、环境通道和 namespace freshness。更新后的 source contract file/side/outer 全部重新计算，旧 outer 会被检查器拒绝。

本作者没有导入或执行 launcher/engine，没有运行 DC/VCS，没有创建 attempt/result/work/quarantine。该证据仍是 source-only，不能写成 mapped 定位、时序、功耗、能量或系统加速。

M1096r2 必须独立 pin launcher SHA、launch receipt outer、engine SHA、M1093r2 outer，并对 argv、环境注入、seal 漂移、父进程 cmdline 与 namespace 做攻击。只有 M1096r2 GO 后，root 才能外部 pin 完整 tuple 并执行一次唯一命令。
