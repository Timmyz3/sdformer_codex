# M1143R6 frozen-netlist mapped-VCS successor source author receipt

结论：**PASS；只授权不同作者继续做 source hammer。** 本回执不授权 launch、
attempt、VCS、mapped VCS、DC 或其他 EDA。

只读 preflight 精确核验了 M1142R6、M1141R6 checker/contract/author receipt、
原 M1133R6 attempt/failure、mapped netlist、foundry Verilog library、memory model、
case0 TB 与 VCS binary。原 failure 目录没有 SDF，因此 future command 保持原
case0 合同，不添加 SDF 选项，也不重跑 DC。

冻结 netlist 上重新执行 337-bit structural gate：12 个 active-low clear nets，
75 个 direct inverter register bits，262 个 buffer-then-inverter bits，最大链长 2，
全部单驱动、恰好一次反相并回到 `rst_core`。

controlled fake runner 验证 future one-shot 只产生一条 14-argument compile command
和一条 128-cycle case0 command；成功路径要求唯一 PASS token 且无
`M1112_FIRST_X`，随后双封并原子发布。compile failure、PASS token 缺失/重复、
X token 和 result collision 均 fail-closed；所有 post-attempt failure 只生成一个
双封 quarantine，attempt 保持 consumed，重试被拒绝。

共 266 checks、10 attacks。真实 subprocess/VCS/DC/EDA 调用为 0，新 attempt/result
均未创建；原 failure、netlist、subject 和 `docs/359` 身份保持不变。
