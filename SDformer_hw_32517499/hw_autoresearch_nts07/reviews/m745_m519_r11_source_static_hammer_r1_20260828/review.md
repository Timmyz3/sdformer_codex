# M745：M519 R11 source/static 独立打铁

结论：**PASS，100/100，P0/P1/P2=0**。R11 source/static 包可以进入“另建 `launch_now=true` release”这一步；本评审自身不授权 DC、VCS、Formality、PT、PTPX 或 remote。

## 核心发现

- 最终封存身份一致：runner `7c588b…`、contract `6d9f30…`、`launch_now=false` candidate `9e6b5d…`。作者 handoff 与 review request 的外层 seal 分别为 `8f2bda…`、`95391f…`。
- R10 的实际失败原因被精确修复：M576 的单引号 jq 程序中那个字面行尾反斜杠已删除；修复后原有 `PASS / 100 / P0=P1=P2=0` 断言返回 0，断言没有放宽。全 runner 没有第二处同类 jq 字面续行符。
- 除新 R11 身份、R10/M740 因果绑定和两条 NO-EDA self-test 外，执行后半段没有改动。把 R10/R11 名字和版本化结果路径正规化后，从进程识别、资源门、preflight、attempt、tool launch 到 cleanup 的 runtime tail 完全相同；资源阈值、三轴顺序、pass gates、工具/库/RTL/Tcl/filelist/SDC 身份均未放宽。
- `static_no_eda_full_path_test.sh` 独立复跑成功。runner 在完整 candidate/admission/contract/JQ 校验后，于第 517–527 行退出，早于第一项 preflight（791）、attempt 发布（907）和 `dc_shell` 调用（1255）。
- 负向测试通过：坏 runner pin、坏 candidate pin、非法 full-path 模式、未知/缺失 identity 或 provenance key、以及 `launch_now=true` 变异均 fail-closed。注入的 helper failure 只在 `/tmp` 形成双封存回执，没有污染仓库 identity。
- 评审结束时 R11 canonical、attempt、work、preflight、reject、仓库 pre-attempt receipt 和最终 release 全部不存在；`docs/359` 仍是 `dedde7ce…`。

## 准入边界

本结论只允许主代理另行创建一个精确 pin 本 runner/contract 的 `launch_now=true` admission。该 release 还必须经过 fresh final-release hammer，之后才可能发布最多一次 DC-only 命令。当前没有面积、时序、功耗、能量、吞吐/mm²、完整 FC2 或系统加速结论。
