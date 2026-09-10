# 创新优先重构：本轮结果

日期：2026-09-06。结论与复现入口：[report.md](report.md)。

**用户后续方向修正：** C1 继续以完整 Prosperity 为底座，融合其他论文机制进行重构；不因已有先验而停止。最新落实见 [C1 Prosperity 融合路线](../c1_prosperity_fusion_20260906/README.md)。本目录的负结果仍有效，但不构成对 C1 整条方向的否决。

本轮的关键修正是加入按 hidden channel 分块、只暂存最终位 G 的强基线。它实质削弱了深层晚门限压缩包的收益依据。源共现统计是已有 V2 路线，本轮补充实际 checkpoint 的均值、方差、PSN 判位数值验证，不能当新发现重新命名。

| 文件 | 本轮实质内容 |
|---|---|
| `screen_prefix_recovery.py` / `sample0_r1/result.json` | 8/12/16 位单调前缀；未决消费者；共享 Y 与直接时间查表恢复的算术对照 |
| `screen_directional_packet.py` / `directional_r1/result.json` | 原 B32K4 表示只恢复失效一侧；记录确切消费者时间掩码 |
| `screen_source_moments.py` / `source_moments_r1/result.json` | 旧 V2 的完整源矩与输出矩、最终判位对照；在线权重二次式成本 |
| `report.md` | 强基线、去留理由、独立评审与一手文献 |

命令在 `/home/zhumd/work/ideafromai/` 下使用 `/opt/anaconda3/bin/python3.12`；每个脚本接受 `--output-dir` 指向新目录。已有结果无需重跑。脚本复用此前二层 sample0 的源加载器和 checkpoint NumPy 读取器。

这些是研究数值与运算量实验，无新增生产 RTL、VCS 周期、PPA、全网 AEE 或投稿贡献句。现有 `hardware_mechanisms_20260906/rtl/` 的三个区间接口模块属于另一份已有工作，本轮未修改，也不借其验证支持这里的点值算术。
