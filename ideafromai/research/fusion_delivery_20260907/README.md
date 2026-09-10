# C1 / C2 多论文融合：本轮可检验结果

2026-09-07。阅读入口：[机制说明、例子与独立评审](report.html)。

**优先推进C2有界时间共享归约；C1继续研究联合替换旧捕获。** 本轮未建立“稳接收”结论。原Prosperity、直接完整T FTP、完整RSR++均保留为强对照。

- C1固定Phi选基：新192个tile、11,904行，C128无驱逐仍输原Prosperity；换缓存不足以建立优势。
- C1新图改写：交集虚拟父有算术机会，但要用取消旧捕获支付新增读写。联合替换反例已验证；真实同cohort的有界联合门为6662→6584次加法，端口5584→5583，当前净余量薄。
- C2真FC2：全签名平坦/递归归约有显著机会，成员链和类顺序会损失bank并行。
- C2源顺序8槽参考：保留20.78%/42.21%的加法减量，省去成员链；所有源系数仍读一次。并发同槽更新、排空和输出背压尚未闭。

完整数值为`z=theta*g`，连续值阈值幅值不丢；上述正确性为非1阈值有理数诊断，非冻结FP32等价。无新生产RTL、EDA、PPA、AEE或系统FPS。

| 文件 | 角色 |
|---|---|
| `*_plan.json` | 每次执行前的假设、样本、成本和范围 |
| `screen_materialization.py` / `materialization_r1.json` | 固定路由：分区、二次准入、无驱逐诊断与原Prosperity |
| `screen_virtual_parents.py` / `virtual_parents_r1.json` | 保留原森林，单步插入虚拟交集父 |
| `screen_joint_exchange.py` / `joint_exchange_r1.json` | top8中有界成对替换、取消旧捕获的结构证书 |
| `screen_fc2_signatures.py` / `fc2_signatures_r1.json` | 真FC2全T分组、完整RSR++及后序消费 |
| `screen_source_order.py` / `source_order_r1.json` | 原通道序、有限部分和、驱逐后完整分发 |
| `implementation_boundaries.md` | 端口、生产者顺序、元数据与数值边界 |
| `independent_reviews.json` | 独立概念/代码评审；评分不是录用概率 |
| `source-ledger.json` | 一手论文来源与未解决访问缺口 |
| `WORKLOG.md` / `SHA256SUMS` | 研究状态与文件核验 |

Python固定`/opt/anaconda3/bin/python3.12`。脚本拒绝覆盖既有收据；复现需复制完整研究包到新目录并保留相邻先验路径，不能删除旧结果重跑。原封存校准包保持只读。
