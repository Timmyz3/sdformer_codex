# M759：M519 R12 license-env source/static fresh 独立打铁

结论：**PASS，100/100，P0/P1/P2=0**。R12 source-only 包通过，主代理可以另建一个精确 pin 的 `launch_now=true` release；本评审自身不授权也未运行 DC、VCS、Formality、PT、PTPX、remote 或 license-server query。

## 关键结论

- 三个冻结身份精确匹配：runner `fd53e3…`、contract `7c8445…`、candidate `6f99e4…`；candidate 保持 `launch_now=false`。两份 JSON 严格可解析、sidecar/outer seal 均通过，17 个 `exact_files` 全部逐字节匹配，`docs/359` 仍为 `dedde7ce…`。
- R11 失败链被不可变绑定：quarantine 为 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`，唯一 attempt 为 `CONSUMED_AT_FIRST_DC_LAUNCH`，M752 为 100/100 且 P0/P1/P2=0。R12 使用新的 runner/contract/release/result/attempt 身份，不复用或重释 R11。
- exact environment 被关闭为 `SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo` 与 `LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat`；本地 license file 和 `lmutil` 的路径及 SHA 同时在 runner/contract/candidate 绑定并在 live launch 时逐字节复核。
- fail-closed 顺序成立：K1 资源 preflight → status-only `lmstat` server/`Design-Compiler`/`DC-Ultra` → 双封存 raw stdout/stderr/rc 与 parsed receipt → 仅当 server 可解析且两个 feature 都明确有 free seat 才发布唯一 attempt → 首次 DC。unknown、unreachable、unparseable、issued<in-use 或无 free seat 均在 attempt 前拒绝。
- license probe 明示只是瞬时状态：不是 checkout、不是 reservation、不是 DC/PPA 证据。

## NO-EDA 验证

- 完整 candidate/admission/contract 路径 self-test 返回 0，并写出 `PASS_M519_R12_FULL_ADMISSION_CONTRACT_PATH_NO_EDA`；其退出点早于资源 preflight、license query、attempt 与工具调用。
- early helper 返回 0；注入 failure 返回 86，只在 `/tmp` 形成可验证双 seal 回执。
- 缺失 license env、错误 `SNPSLMD_LICENSE_FILE`、错误 `LM_LICENSE_FILE` 均返回 3，且未形成仓库 attempt/result。
- 从冻结 runner 原样抽取的 parser 用离线 fixture 验证：有 free seat 返回 0；无 free seat、格式不确定及 in-use>issued 均返回 1。该测试没有连接 license server。

## 准入边界

评审结束时 R12 final release、canonical result 与 attempt sentinel 均不存在。这里只允许另行 author true release，并要求再做一次 fresh final-release hammer。没有观察 license 可用性，没有运行 DC，也没有面积、时序、hold、功耗、能量、吞吐/mm²、完整 FC2 或系统加速结论。
