# M772 / M533 r14 VCS environment read-only preflight

PASS。严格 clean environment（env -i）下未设置 HOME；以实际编译同架构的 `vcs -full64 -ID` 确认 `V-2023.12-SP1_Full64`。FlexNet 只读状态查询确认 `VCSCompiler_Net` 和 `VCSRuntime_Net` 均为 99 issued / 0 in use。没有 HDL compile、simv、seat checkout、result mkdir 或 r14 attempt 消耗。

bare `vcs -ID` 会把安装根错误解释到 `$VCS_HOME/linux`；该诊断不是本 PASS 回执的一部分。r14 必须用与冻结 compile `-full64` 一致的 identity probe。
