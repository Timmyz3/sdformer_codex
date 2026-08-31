# M729/M533 r10 source-static fresh hammer

**PASS，100/100，P0/P1/P2 = 0/0/0。** 本次只做源码、身份、封存和控制流静态审阅；没有执行 runner、VCS、simv 或任何 EDA，也没有修改作者文件。

## 结论

- r10 runner SHA 为 `dd601184...4646d`，source contract 与双 seal 通过，TB r4 SHA 为 `320901a0...c82`。
- r9 consumed failure 与 M726 独立失败评审均保持 byte-exact、双 seal 有效；M726 只允许一个新 r10 identity。
- 将新 TB 在内存中精确反向变换后，得到旧 TB SHA `72a6cef7...345ff`。唯一功能相关差分是把 `legal_parent_data` 从 automatic task storage 改为 module-scope `force_parent_data_static`，计算值、force 目标和 oracle 均未变化。
- top RTL、SVA、macro adapter 和 macro binding 的 SHA 全部保持冻结值。
- runner 新增 canonical `RUNNER_PATH`，并让 receipt-writer 的 Python 失败显式返回；初始门与 pre-mkdir 终态门都重新验证 r9/M726。
- wrong-r9-runner、module-scope force、跨 `cd` canonical path、receipt non-masking 四个静态负例/复现均通过；新 r10 result path 不存在。

## 授权边界

本评审只允许后续创建新的 launch candidate 与独立评审链，不授权现在运行 VCS/simv/EDA。launch candidate、candidate hammer、launch-now true release 和 final-release hammer 当前均不存在；在它们全部独立双封存且 live preflight 通过前，r10 仍是 source-only。
