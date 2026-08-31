# M794：M784/M533 R15 pre-mkdir 未定义函数失败审计

结论：**失败原因闭合，PASS_FAILURE_AUDIT 100/100；P0/P1/P2 = 1/2/0。** R15 没有进入 VCS，也没有消费硬件 attempt；但已经被调用过的 R15 release 控制流不健全，现永久撤销，只授权新建 additive R16 source，不授权运行 R16。

## 失败与边界

- R15 runner SHA 为 `0bff3424...`。第 412 行只定义了 `verify_r13_failure_m770_m782_and_author_preflight_prerequisites`。
- 第 876、907 行正确调用完整名称；第 1102 行最终重校验残留旧名 `verify_r13_failure_m770_and_author_preflight_prerequisites`。旧名定义数为 0、调用数为 1，故 shell 报 `command not found`。
- 失败点 1102 早于 live `vcs -full64 -ID` / `lmutil` 入口 1104，也早于原子结果目录 `mkdir` 的 1116。真正的 VCS compile 在 1201，simv 在 1213，均未触达。
- 结果目录及同名前缀 sibling 为 0；`/tmp/m784_m533_r15_unit_delay_vcs_preflight.*` 在 EXIT cleanup 后为 0。按冻结合同，atomic result mkdir 才消费 attempt，因此 R15 attempt 未消费。
- 工具二进制和 license file 在失败前只被读取做 SHA 校验；没有执行 VCS identity probe、没有查询 license server、没有 VCS compile、没有 simv。

## 为什么 M787/M792 漏检

`bash -n` 在这个坏 runner 上仍返回 0，因为未定义命令在 shell 语法上合法，只会在运行到该行时解析失败。

M787 明确没有执行或 source runner；它实际执行的是抽出的 M770 Python heredoc，其他部分是 SHA、字符串和语法检查。M792 同样只执行 `execute_real_heredoc()`，对 runner 的其余审查是 edge/token 检查。两者都没有做“全部自定义函数定义—调用闭包”，也没有用无 EDA stub 完整走到 pre-mkdir live-probe 边界。因此两份 100/100 launch admission 被本审计 supersede；其余 SHA/UNIT_DELAY/协议/边界检查不被反向判假，但不能继续支持 R15 release。

## 裁决

- **M794-P0-01**：R15 最终 pre-mkdir 调用了未定义函数，release 不可执行。
- **M794-P1-01**：M787 的 `bash -n` + 局部 heredoc 测试没有覆盖函数闭包。
- **M794-P1-02**：M792 继承同一盲区，未对 exact pre-mkdir 路径做 stub dry-run。

R15 release `6c3d4a1...` 与双封保持原样，作为历史负证据；状态改判为 `PERMANENTLY_WITHDRAWN_DO_NOT_EXECUTE_DO_NOT_CITE`。禁止重跑、resume、改名或引用 R15。

## R16 的硬门

R16 不得只修第 1102 行。新 source 必须同时做到：

1. 静态解析所有自定义函数定义与调用；pre-mkdir 原子 `mkdir` 前每个可达调用必须恰有一个定义，无 stale/prefix 变体，无重复定义；注入一次 R15 旧名的负测试必须失败。
2. 对 exact runner 入口到 live VCS/license probe 前的完整控制流做 dry-run。仅把 live probe 边界替换成测试 stub；stub 发出唯一 sentinel 并在结果 `mkdir` 前退出。此前所有真实 prerequisite 函数必须实际执行，且 VCS/lmutil/license/VCS compile/simv/result mkdir 全为 0。
3. 测试源码与输出都要 SHA 绑定；source hammer 和 final hammer 必须各自重跑上述两项，不能只信上一层布尔字段。

本评审未改 RTL/TB/SVA/runner/release，也未调用 EDA；`docs/359` SHA 仍为 `dedde7ce...`。
