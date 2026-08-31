# M770 / M533 r13 VCS_HOME 失败 fresh 独立审计

结论：**PASS（100/100，P0/P1/P2 = 0/0/0）**。M758/r13 的双封存失败包完整，但该 attempt 已永久消耗且必须保持 `FAILED_DO_NOT_CITE`。它没有进入 HDL 编译，没有生成 `simv`，对功能、时序、PPA、能量、周期和加速均无结论。

## 现场证据

- `SHA256SUMS` 与外层 seal 全部通过；终态回执为 `phase=vcs_compile`、`runner_exit_rc=1`、`child_rc=vcs_1_tee_0`。
- `compile.log` 只有两行：VCS 在 `/bin` 下找不到 `vcsMsgReport`，并明确要求正确设置 `VCS_HOME`。
- r13 的独立发布命令用 `env -i` 只保留 PATH/LANG/LC_ALL；runner 虽然绝对路径调用 `/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs`，却未设置或验证 `VCS_HOME`。实际 `vcsMsgReport` 在 `/opt/synopsys/vcs/V-2023.12-SP1/bin/vcsMsgReport`，SHA256 为 `b34e06a92b05856532f868d32c0c81f1708506096856ad9a97bd27e2bd60215b`。因此这是已证明的直接根因，不是 RTL/TB/SVA 失败。
- 资源 monitor 完成 final synchronous ack，三次 collision 均 PASS，因而失败不是 OOM 或同 UID 工具冲突。

## license 边界

`env -i` 同时删掉了站点的两个 license 变量。r13 在 license checkout 前就退出，所以 license 缺失不是这两行日志的已证直接根因；但 r14 若只修 `VCS_HOME`，会在下一阶段继续暴露不可复现的 checkout 风险。因此 clean environment 必须同时精确 pin：

- `VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1`
- `VCS_ARCH_OVERRIDE=linux`
- `SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo`
- `LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat`

本地 license file 与 `lmutil` 也应按 JSON 中的路径/SHA 封存。

## r14 最小授权

仅允许新的 additive r14 身份和路径，增加上述精确环境、`vcsMsgReport`/license file/`lmutil` 字节绑定，以及 attempt 消耗前的只读 preflight。原 52 条 `require_regular_sha` edge 的值与目标必须字节不变；RTL top r2、TB r7、SVA r2、foundry `+define+UNIT_DELAY`、全部 watchdog/resource/collision/coverage gate 必须冻结。

preflight 必须在 r14 原子 result mkdir 前完成，不能启动 compile：

1. 在完整 pin 的 clean env 下只读执行精确 VCS binary 的 `-ID`，验证 2023.12-SP1 身份。
2. 用 SHA 绑定的 `lmutil` 做不占 seat 的 status query，至少核对 `VCSCompiler_Net` 和 `VCSRuntime_Net` 可达且有空闲 seat。
3. preflight 回执双封存后，再走新的 source/candidate hammer、true release 和 final-release hammer。

本审计不授权现在运行 VCS/simv/EDA。`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
