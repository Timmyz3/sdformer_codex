# M814 fresh independent source-hammer request for M813/C2 R17

请对 M813 做全新、对抗性、source-only 打铁，不信任作者 PASS。重点不是重审 M803 算术，而是证明 M811 的三个 launch-boundary 缺陷真正关闭，且 M803 RTL/SVA/TB 与五档 exact 周期硬门没有变。

必须亲自注入：result 在 precheck 后碰撞、attempt 目标污染/碰撞、duplicate `status` / nested `authorization.launch_now` / identity SHA、PRE_STAGE 失败、POST_STAGE 失败、failure quarantine 目标碰撞。任一攻击导致覆盖、嵌套、非封存 attempt/failure、或权限 JSON 歧义，均不得 PASS。

评审严禁调用 VCS、simv、lmutil/license server、DC/Formality/PT/PTPX/EDA，严禁创建 true release、formal attempt/result，严禁改 `docs/359`。PASS100 也只授权新建一份精确 true release 和 final-hammer request，**不授权立即运行 VCS**。
