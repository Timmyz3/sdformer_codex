# M2272 接手协调与新增实验

先读 [取舍与实测结论](report-source.md)。本目录补充已有 GH 包，不覆盖其主报告或此前审阅记录。

主要产物：

- [新增 idea 逐条审阅](records/new_ideas_and_g12_review.json)，[M2271/更早筛查协调](records/m2271_independent_review.json)，[ExSpike 原文复核](records/exspike_c1_prior_review.json)。
- [固定槽与目录分区实测](records/fixed_slot_compression.json)，[独立审查](records/fixed_slot_independent_review.json)。固定80B方案淘汰。
- [晚阈值回放服务及失败位图](records/repair_sector_sample0_r1/result.json)，[同q4组织的隔离控制](records/repair_same_q4_controls.json)。保留研究，但没有净周期/能量准入。

本轮均为 CPU 研究。冻结 checkpoint、训练、生产 RTL、主稿和他人改动未修改。

重跑入口使用 Python 3.12；repair 必须指定不存在的新目录，防止覆盖已有结果：

```bash
/opt/anaconda3/bin/python3.12 scripts/screen_fixed_slot_compression.py --output /tmp/fixed_slot_recheck.json
/opt/anaconda3/bin/python3.12 scripts/screen_repair_sectors.py --output-dir /tmp/repair_sector_recheck
```

`scripts/derive_same_q4_controls.py`只读取本目录已有失败位图与源列计数，不执行前向。`scripts/screen_selective_signature_service.py`是发现重复后取消的探索代码，无完整结果，不作为后续实验入口。
