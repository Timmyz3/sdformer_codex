# ideafromai 已直接纳入主仓

2026-09-11 起，实际目录是 /home/zhumd/work/sdformer_codex/ideafromai/。
整个原目录已移动到这里，包括环境、捕获和未入 Git 的本地大文件；这里不再是同步副本。
旧 /home/zhumd/work/ideafromai 为指向本目录的兼容链接，两个入口访问同一批文件。

直接在本目录编辑并通过主仓 git add / commit / push 管理，不再运行单向同步。
旧 tools/sync_ideafromai.py 保留兼容入口，只显示新位置，不复制或删除文件。

Git 追踪代码、RTL/TB、配置、研究文档、结果文本以及已准入的小型参数／复跑数据。
.gitignore 保留环境、原始捕获、训练包、检查点、构建和波形；既有超过8MiB的生成轨迹／大DAG
逐路径排除，外部论文PDF保留本地。所有这些文件已随真实目录迁入，不是被删除。
新增普通文档和代码无需同步器选取；需要增加参数附件时直接调整Git忽略例外即可。
因此Git是研究源码与可审阅结果的版本库，完整训练／捕获数据仍需本地或专门的数据备份。

活动代码、文档入口、INDEX／MANIFEST、两个虚拟环境的启动路径及Grok隔离树链接已更新。
历史结果JSON、封存原稿和box原始镜像保留当时记录的路径，由兼容链接承接；生产RTL与docs/359未改。
旧筛选快照仅作为迁移备份保留在 /home/zhumd/work/.ideafromai_migration_20260911/previous_snapshot/，
它不再是编辑入口，也不再运行同步。

[想法总入口](README.md) · [当前硬件研究线](research/hardware_innovation_20260908/README.md)
