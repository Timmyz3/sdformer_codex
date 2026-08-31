# M1155R6｜M1154R6 C2 dual-DUT source 独立停止打铁

结论：**M1154R6 的 fail-closed source 合格，但旧 M1133R6 冻结网表不能继续扩 observation；不跑 VCS/DC，只保留 M903 logic-only 证据。**

## 稳定 tap 门

source、作者回执、contract 三重封及冻结网表身份均独立对上。直接扫描 exact mapped-netlist 的 declaration prefix 得到：

| 语义 tap 类 | 需求 | 稳定存在 | 缺失 |
|---|---:|---:|---:|
| retained fault-Q | 5 | 5 | 0 |
| paired req/rsp accept | 4 | 0 | 4 |
| consistency fault now/q | 2 | 0 | 2 |
| core/adapter protocol error | 2 | 0 | 2 |
| 合计 | 13 | 5 | 8 |

同一 declaration prefix 中存在 120,128 个匿名 `n*` 名称，但它们没有稳定语义。猜匿名网、用层次通配或只观察 5 个 retained fault-Q 都不能形成可复现 root-cause 绑定，必须在 attempt/VCS 前拒绝。

## bounded mock 与第一处边界修正

bounded synthetic census 能在 13/13 显式名字时通过；valid-qualified endpoint 模板也正确约束了：只有 `valid && payload_known` 才能 ready/accept，未知 payload 单独拉起 diagnostic，且没有 `force`、`initreg` 或 X coercion。

不过当前 `dual_dut_probe_template()` 只生成精确 tap 路径及 atomic-first-X 的**注释规格**，没有 DUT instance、always block 或 bitmap 寄存器。因此作者回执中的 `paired_accept_and_fault_atomic_first_x_bitmap=true` 只能解释为设计意图，不能解释为“已实现/已仿真”。这不削弱 STOP 判定；反而意味着即使未来 tap 门通过，也还必须补一个真正可执行的 dual-DUT/first-X TB。

## 执行与攻击

独立 checker 共 60 项检查，M1154 namespace 保持 fresh，真实 attempt/VCS/simulation/DC 都是 0。五类攻击全部拒绝：匿名网猜测、通配层次、仅 5 tap 仍授权、force-X、重跑已消费的 M1146。

唯一合法的未来重开方式是：另行授权新 RTL，在综合前显式 preserve 全部 13 个语义 tap，生成新网表，再用真正可执行的 dual-DUT/first-X TB 和全新 one-shot namespace。旧 frozen netlist 不再继续挖 observation，也不得由此升级 mapped functionality、activity/power 或 paper PPA claim。
