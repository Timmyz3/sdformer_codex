# M616 fresh true-launch hammer request

请对 M615 authored M614/M612/M593 parent-scratch energy production admission 与 true release 做一次 fresh、independent、read-only hammer。评审期间不得调用 runner `--execute`、正式 analyzer，不得创建 result/attempt/consumed，不得运行 GPU、EDA 或 remote job。

必须验证 admission SHA `0e194055d4a6ac396b091d6c3d0dba61b94d28d0936ecf89352c96e95a23f630` 及 outer-seal-file SHA `7cd77b75b5439fe46140b3e8d4889f2b57c1de720200f9b029500a62d4fa9e51`，并确认它恰好采用 runner 接受的 10 个顶层 key、`launch_now=true`、`release=true`。只读 authorization validate-only 与已绑定 source preflight 可以执行。

必须验证 true release SHA `9f465b9a091ded283bdddb2a37dc596b2cbfed83e48b4f0567ba9297819e8fa2`、outer-seal-file SHA `a474c48cad9650d994de25f6fc9e016ed21df8764ab342f0f7593973511225ee`、`max_attempts=1`、`still_not_executed=true` 和 `namespace_collision_acknowledged=true`；验证 M615 handoff manifest SHA `0832285e9c350dc4d49d74ef0e8998dd701bdaa12bf227043d493c7a7dd48d91` 及 outer-seal-file SHA `b2e574e102241e6dc73d0f69f3a3e99d38c6c0d56c4307350fca1f3117a02871`。

必须逐项核对 M612 candidate/shell/Python/adapter identity、M613 PASS100 review SHA `83e545b3b9b8e5069626ef842af5de5b85d1f7b7d76896813ce2f32d653c3109`、manifest SHA `e2329f6bb12a8aebe28ce8b9645e2d3ffe1639b92fb935d46c7229d02204531d`、outer-seal-file SHA `1a8c834023f5c8782a73c2ae1178c234e5152658eb6420009205cf20c6d74d9f`，以及 M597 source contract/analyzer、冻结方程和 component-only claim boundary。

用 `lexists` 语义确认唯一 canonical result、attempt、consumed、runner staging 及 quarantine raw/staging/final 全部不存在。重新采集三次、间隔 2 秒的资源样本并核对门限；runner 不实施该策略，因此即使 M616 PASS，任何未来唯一 invocation 之前仍须 root 紧邻执行 fresh live resource/cgroup/collision recheck。

数字前缀 `m614` 的共享已显式确认：既有 PAFT 完整 ID/path 是 `m614_m579_paft_control_single_port_product_capture_r4_result_hammer_r1_20260828` / `reviews/m614_m579_paft_control_single_port_product_capture_r4_result_hammer_r1_20260828`；energy admission 完整 ID/path 是 `m614_m612_m593_parent_scratch_energy_true_launch_admission_r1_20260828` / `contracts/m614_m612_m593_parent_scratch_energy_true_launch_admission_r1_20260828.json`。两者不存在 exact collision；必须确认 PAFT manifest SHA `b77df8fc9bb44f3cb39731991b0b621c40f848aa7b465d1ff192c3a0b39ce439` 与 outer-seal-file SHA `3b6676cc215817c23bcf1f1d758071996b09b71ed4e87b2532eabafab8765215` 未变。

PASS gate 为 score ≥95、P0=0、P1=0。输出到 `reviews/m616_m615_m614_m612_m593_parent_scratch_energy_true_launch_hammer_r1_20260828` 并双封。即使 PASS，也只授权在 root fresh live recheck 后的一次 exact invocation；任何 raw result 仍需 fresh independent result hammer，冻结诊断数字不得提前作为 admitted result。
