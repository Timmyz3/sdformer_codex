# M1761 release 作者自检

`M1761` 仅授权一次 `M1753` 执行；作者侧没有启动 EDA 或查询 license。

- 精确绑定 M1753 runner/checker/test、source contract 双封与作者收据双封。
- 精确绑定 M1760 review `98/100` 与 review/manifest/outer 三元组。
- 预算固定为 K1/K8/K1x8 三轴，各五例：3 次 VCS compile、15 次 simv、15 个 SAIF、15 次 PTPX，禁止自动重试。
- 必须先完成全部 15 个 checked SAIF，才允许第一次 PTPX；任何部分轴均不可引用。
- workload 只能称 `DIRECTED_COMPONENT_NOT_PRODUCTION`。
- PTPX 只报 whole mapped logic 的 internal/switching/leakage/total；排除 weight SRAM、testbench memory、IO PHY、clock tree 与 post-layout parasitics。
- `1.016728x` cycle 与 `4.562720x` throughput/mm2 必须同表同句；K8 对单 K1 禁止 headline。

M1760 的唯一 P2 是 M979 内部没有 K1 cycle assertion；M1753 exact post-sim checker 会在 SAIF 计数准入前绑定唯一 K1 PASS/cycle，因此该项保持 nonblocking，不修改已发布 source 身份。
