# 创新优先：C1/C2 逐文件复审

日期：2026-09-06。唯一投稿目标 TCAS-II。已精读 42 份原始文件，另复审上一轮研究入口，形成 620 条定位记录；含重复机制、文献背景和工程史料，不是 620 个独立创新。

首选阅读 [可检索逐条审阅台账](C1C2创新逐条复审.html)。可搜索机制、文件或先验，按主机制潜力筛选并展开完整论证。

- [结论与重构路线](synthesis.md)：五个新候选、独立评分、反证与最低证据门。
- [完整文字报告](report-source.md)：结论加逐文件逐条审阅，是本包的规范报告源码。
- [机器可读台账](audit-ledger.json)：每条记录的定位、四维评分和迁移判断。
- [来源账本](claim-source-ledger.json)：先验链接、访问范围及独立审阅快照。
- [质量检查](QA.json)：覆盖、哈希、字段、链接与结构检查；未声称浏览器视觉全检。

当前第一优先是浅层两个 FC1 的二值源统计先行；两个独立评审均给研究潜力 6/10。C1 尚无通过主创新筛选的方案。评分不是录用概率，当前没有稳 accept 或可投稿准入。原 C1/C2 缓存、重算、重排结果保留为实现探索，不再作为主创新推荐。

本轮新增数值仅为预先选定 sample_id=0、全部 12 个 FC1 完整 BN 域的 CPU 源统计，详见 [结果](records/v2_sample0_moments.json) 与 [脚本](scripts/screen_source_moments.py)。不代表 S40 分布、valid825 精度、RTL 周期或 ASIC PPA。该脚本避免覆盖封存输出；没有重新采集或训练。

报告重建需 Python 3.12 和 markdown 包：

```
/opt/anaconda3/bin/python3.12 scripts/build_audit.py
```

构建核对原始 inventory 和四份逐文件审阅。根 README/INDEX/MANIFEST 仅修订导航优先级，审阅前版本保存在 original_snapshots/，其前后 SHA 见 catalog_updates.json。原提案和旧实验数字未改。

没有修改主稿贡献句、生产 RTL、docs/359 或 H81；没有 EDA、新训练或结果准入。后续电路验证仍受候选合同与用户选定生产方向的边界约束。
