# M1907：M1906 TSBG `VCS_HOME` clean-env successor 独立静态打铁

## 裁决

**FAIL，91/100，P0/P1/P2 = 0/1/0。不得启动 M1906 的 license、attempt、VCS 或 simv。**

M1906 已精确绑定 M1898 的 consumed-attempt 与 failure quarantine 双封，也精确绑定失败日志 `e6c82b04...`。该日志证明 M1898 唯一可观察失败是 clean environment 未提供 `VCS_HOME`，VCS 因而在 `/bin` 找不到 `vcsMsgReport`；failure 目录中没有 `simv.log`，没有仿真执行。

但 M1906 没有实现本轮唯一允许的对象差：

- 第 155--158 行把 `VCS_HOME` 传给了 `lmutil` license preflight；
- 第 160--164 行正确传给唯一 VCS compile；
- 第 166--168 行的唯一 simv 环境反而没有 `VCS_HOME`。

因此实际接收者是 “license + compile”，而合同要求的是 “compile + simv only”。这不是新治理阻塞，而是一个明确、局部且可机械修复的 P1 对象差偏离。

## 已通过且不得重新发明的检查

相对已通过 M1899 的 M1898 基线，以下项未改变并继续通过：clean direct shebang、固定绝对路径、attempt 在 license/EDA 之前且先双封、success/failure `publish_no_replace_checked`、failure 非 best-effort、same-UID EDA/common-shell 截断门、无自动重试、唯一一次 license/compile/simv、`-assert svaext`、冻结 RTL/filelist/SVA/TB 身份。

M1906 的 attempt/result/failure/lock namespace 在本审阅期间均不存在；本审阅没有运行 license、VCS、simv、DC 或 PT，也没有创建 attempt/result。

## 最小后继

采用新 additive namespace，保持 M1906 其余源与治理不变，仅做两处环境行修正：从 `lmutil` 删除 `VCS_HOME`，并给唯一 simv 添加同一固定 `VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1`；compile 继续保留该变量。随后重新做独立 source review、different-author release 与 release audit，才可授权一次 attempt。

本裁决不授权性能、面积、能量、系统倍速或论文准入主张。
