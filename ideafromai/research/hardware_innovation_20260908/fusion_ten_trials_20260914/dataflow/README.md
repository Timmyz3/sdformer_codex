# fusion_ten_trials：dataflow独占三项

[总结果](REPORT.md) · [机器可读汇总](SUMMARY.json) · [原始假说/公平控制](PLAN.md) · [root独立实现审核](../REVIEW_DATAFLOW.md)

独立三个目录各有可编译SV/C++、准备脚本与实际JSON收据；各README给重现命令。先完成对应run，再在此目录运行`/opt/anaconda3/bin/python3.12 audit.py`与`/opt/anaconda3/bin/python3.12 summarize.py`。不运行一次性源编辑脚本；它们已移除。build与可由prepare重新产生的fixture binary按本地.gitignore忽略，最终SV/JSON/CSV/文档保留。
