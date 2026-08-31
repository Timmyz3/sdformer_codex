# M819 fresh independent source-hammer request for M818/C2 R18

请 receipt-blind 审查 M818，不信任作者 PASS。必须冻结 M803 RTL/SVA/TB/filelists 与五档 exact 周期门，并重点打 M814-P1 的实际修复边界。

必须动态注入两类相反结果：一，attempt 发布前失败或 no-replace collision，canonical 未由本次 move 建立且 stage 仍在，双封 failure receipt 必须为 `attempt_consumed=false`；二，rename 已成功、shell latch 尚未建立或 canonical post-verify 被注入失败/损坏，canonical 已出现且 stage 已移走，双封 receipt 必须为 `attempt_consumed=true`。还需检查 publication/postcheck 显式排序、strict duplicate/nonfinite JSON、flat attempt、result renameat2、failure quarantine primary collision fallback 与 future exact launch chain。

本请求严禁 VCS、simv、license server、DC/Formality/PT/PTPX/EDA，严禁 true release、final hammer、formal attempt/result，严禁修改 `docs/359`。PASS100 也只授权另建一份 exact true release 与 final-hammer request，不授权立即运行 VCS。
