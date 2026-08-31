# M874：M863/C1 R21 conditional one-way true-release handoff

M874 已生成 runner 固定路径要求的 R21 conditional release，并精确绑定 M864 source hammer 与 M871 candidate hammer 的 review / manifest / outer-seal 三重身份。release 的 `launch_now=true` 只是冻结 runner 所需的结构字面量；当前 `authorization_effective_now=false`。

release 只有在两个条件同时满足后才可生效一次：由不同 fresh reviewer 在固定路径发布 M875 PASS100、P0/P1/P2=0/0/0 的 final hammer；随后 caller 独立复核该包双封并 pin 其实际 outer-seal-file SHA。此前禁止 live runner、VCS、simv、license query 或任何 EDA。

作者侧在 Python 3.6.8 与 3.10.16 下完成 11-key typed authorization 正负例、source/event/closure 三负例、timeout fake suite 和 rc86 pre-mkdir 零副作用测试。未修改 runner、TB、RTL、SVA、macro、binding、foundry model 或 `docs/359`，未创建 R21 result/attempt/quarantine。

即便未来 functional VCS PASS，也只证明该 island 的功能和冻结 cover/attack gate，不得提升 cycle、speedup、timing、PPA、energy、system 或论文 claim；M528 1.746753× 仍是 CPU same-ledger 数字。
