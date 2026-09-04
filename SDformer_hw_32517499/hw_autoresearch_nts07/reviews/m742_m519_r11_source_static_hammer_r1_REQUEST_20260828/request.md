# M742 / M519 R11 source-only fresh independent hammer request

请对 R11 做 receipt-blind、read-only 静态评审；不得运行或修改 DC/VCS/Formality/PT/PTPX/remote，也不得创建 `launch_now=true` release。

必须完成：

1. 校验作者 handoff、R11 runner、contract、`launch_now=false` candidate、R10 失败回执与 M740 的精确 SHA 和双封存；确认 R10 runner SHA 仍为 `7dc7d79c...`，`docs/359` 仍为 `dedde7ce...`。
2. 独立证明 R11 对 R10 的唯一 jq 语义改动是删除单引号程序内的字面反斜杠，且 PASS/100/P0=P1=P2=0 断言未放宽。扫描整个 runner 是否还有同类单引号 jq 字面续行符。
3. 重跑 `static_no_eda_full_path_test.sh`，确认完整 admission/contract jq 路径确实在第一项 preflight、attempt 发布与 `dc_shell` 前退出，且没有产生任何 R11 run identity。
4. 检查所有 canonical、work、attempt、preflight、reject、receipt 路径均为 R11 唯一；R10/R9/R8 不得重用或改写。
5. 攻击 candidate：`launch_now=false` 必须是 fail-closed；缺少 full-path 自测变量、runner/admission SHA pin 不匹配、任意未知/缺失 identity/provenance key都必须失败，且不得触发 EDA。
6. 检查资源、collision、tool identity、RTL/Tcl/filelist/SDC、三轴公平性和 claim boundary 未被放宽。

PASS 只能说明 source/static package 可进入下一道 release authoring；不得授权执行。通过后仍须新建一个精确 pin R11 runner/contract 的 `launch_now=true` admission，再经 fresh final-release hammer 才能发布最多一次 DC-only 命令。
