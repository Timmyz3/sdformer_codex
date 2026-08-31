# M742｜M519 R11 source-only repair author handoff

状态：`AUTHOR_SOURCE_ONLY_R11_READY__FRESH_INDEPENDENT_STATIC_HAMMER_REQUIRED__NO_EDA_AUTHORIZED`。

本包按 M740 的唯一 successor 边界创建全新 R11 身份，R10 没有被修改。唯一继承语义修复是删除 M576 `jq` 单引号程序中 `score_out_of_100 == 100` 后的字面反斜杠；`PASS`、100 分和 `p0/p1/p2=0/0/0` 断言均保留。R11 另增加一个完整 no-EDA 自测模式，它使用双封存的 `launch_now=false` 候选，走完 admission、contract、历史证据和全部 jq 校验后，在资源 preflight、attempt 发布和第一次 `dc_shell` 前明确退出。

冻结源码：

- runner：`dc_handoff/scripts/run_dc_m519_r11_setup_area_three_axis_exact_sha_r1.sh`，SHA256 `7c588b1a95a0afb075de97d148b5a07bad9dc2040ab890c7eb00f6c507ff6692`
- contract：`contracts/m519_r11_setup_area_three_axis_recovery_contract_r1_20260828.json`，SHA256 `6d9f30852e4afec80384417fa8bd01d561101846a6b88079cff6ea8088e11334`
- candidate：`contracts/m742_m519_r11_setup_area_three_axis_dc_launch_admission_candidate_r1_20260828.json`，SHA256 `9e6b5de45d26a133a08b05caa60889a10c34aa497af426d8bc3bd35580e1da1b`

作者自测 `static_no_eda_full_path_test.sh` 已通过：修复后的 M576 jq 表达式返回 0；完整候选 admission/contract 路径返回 0；R11 canonical、attempt、work、preflight 和最终 `launch_now=true` admission 均不存在；没有运行 DC/VCS/Formality/PT/PTPX/remote。候选中的 `authorization` 只冻结可能的未来 release 形状，`launch_now=false` 和 `source_only_authorization.*=false` 是当前权威边界。

下一步只能由 fresh independent static hammer 审阅本包。即使通过，也仍须另建 `launch_now=true` release 并再做 final-release hammer，才可能发布一次性命令。本包不授权 EDA。
