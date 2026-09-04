# M571｜M548 waterfall r2 fresh independent hammer request

日期：2026-08-28  
状态：`AUTHOR_REQUEST_ONLY__NOT_A_REVIEW__NO_EXECUTION_AUTHORIZATION`

请由非 r2 author 的 fresh independent reviewer 对
`contracts/m571_m548_m528_prosperity_phi_style_waterfall_contract_r2_20260828.json`
执行严格只读、source-only hammer。禁止运行 EDA、VCS、runner、训练、远端或大型
CPU 任务。

必须完成：

1. 验证 r2 contract、contract 双封、author handoff 双封、M569 review 双封和不可变 r1 的全部精确 SHA。
2. 用 duplicate-key-rejecting parser 解析 r1/r2/M569/M528/M535。
3. 独立复算 `213376 / 245760`，要求 r2 ratio 精确为 `0.868229166667`。
4. 结构化比较 r1/r2：除 contract identity、date、status、新增 M569 repair provenance，以及上述唯一 ratio 修复外，所有整数、`2.038776477138`、`1.741232213066`、scope、claim policy 与 literature boundary 必须冻结。
5. 独立重算 M569 列出的全部 waterfall 整数守恒式，并检查 r2 没有把 work、cycle、external artifact 或 system 分母混用。
6. 确认 r1 未被改写且继续 `DO_NOT_SHARE`；r2 只有在 fresh review 的 P0/P1 均为零后才能 shareable。
7. 验证 `docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

请将审阅输出独立双封。该请求不授权任何工具执行、RTL/PPA/energy/system/headline
claim，也不允许 author 自我准入。

