# M1804｜M1803 C2 two-VCS 结果独立打铁

结论：**PASS，99/100，P0=0、P1=0、P2=0。**

- M1803 release、M1801 source contract、M1802 source review 与 M1803 result 的双 seal 全部通过；`docs/359` SHA 未变。
- 执行命名空间只有 1 个 attempt marker、1 个 canonical result、0 个 quarantine；结果内恰有 2 个 compile log 和 2 个 sim log，四个 rc 均为 0，无自动重试。
- unit PASS token 恰 1 次：`legal_case0=1`、`invalid_payload_cases=1`、4 类 attack、4 次 reset recovery，public fault 全程 binary。
- full PASS token 恰 1 次：精确周期对 `51/53, 131/133, 486/499, 1231/1246, 14/14`；5 次 protocol attack；numeric/tuple/weight mismatch 均为 0；stall、full-8、candidate/baseline OOO cover 均非零。
- assertion failure 为 0。`full/assert.report` 中 48 条 `0 match` 是 cover property 未命中，不是 assertion failure；未发现 `failed at`、`Offending`、fatal 或 error 记录。

准入口径：M1803 只闭合 **M1801 RTL 功能与 registered-public-fault 根因确认**。它不构成 mapped functionality、PPA、功耗、能量、系统倍速或物理论文数字。

下一门已授权：在**同一 M1801 新 source identity**下，对 K8 与等带宽 K1×8 运行 matched DC、Formality、mapped VCS、SAIF、PTPX；两轴必须使用相同 library、约束、时钟、端口、workload、activity window 与统计边界。旧 M1661 PPA、M1777/M1779 energy 仍不得复用。
