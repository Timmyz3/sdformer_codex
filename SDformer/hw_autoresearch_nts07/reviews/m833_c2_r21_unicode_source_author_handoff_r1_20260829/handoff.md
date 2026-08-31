# M833/C2 R21 Unicode-safe source author handoff

R21 是对失败 M826 live invocation 的 additive successor。唯一功能性源码变化是：runner 中所有 12 个 guard 调用和 1 个内嵌 Python writer 均通过 runner-local `LANG=C.UTF-8 LC_ALL=C.UTF-8` 启动 Python 3.6。外层 caller 仍可固定 `LANG=C LC_ALL=C`；runner 没有全局 export，`license_gate` 与 `compile_and_run` 和 M826 逐字相同，因此 VCS/simv locale 语义未变。

本机 Python 3.6 已实证 `PYTHONUTF8=1` 无效，R21 合同明确禁止把它当修复。真实绝对中文 `docs/359...` 路径在未包装 outer-C control 下稳定触发 `UnicodeEncodeError`，在 runner-local C.UTF-8 下通过完整 source map。

M803 RTL/SVA/TB/filelists、五档 exact cycle、四类 atomic receipt 分类以及 15 键 exact authorization 全冻结。Python 3.6 与 3.12 均通过原 12/12 atomic、原 8/8 authorization、新 5/5 Unicode、source closure 和 outer-C dry-run。dry-run 是 wrong-SHA rc3 与 positive rc86，VCS/license/attempt/result/quarantine 计数全部为零。

M826 release 已调用失败且不可复用；M826 durable attempt 未消费。R21 使用新的 runner/result/attempt 与未来 release 路径。目前仅授权 fresh source hammer，不授权 release、VCS、license 或 EDA。
