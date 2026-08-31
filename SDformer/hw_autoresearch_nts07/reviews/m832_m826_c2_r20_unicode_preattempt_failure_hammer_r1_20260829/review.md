# M832：M826/C2 R20 中文路径 pre-attempt 失败独立审计

## 裁决

**PASS 100/100（失败原因与无副作用边界收口）**。

M826 的一次 live invocation 已经失败，原 M826 release/final hammer 不得复用。失败发生在正式 attempt 之前，因此 durable attempt **未消费**；VCS、license gate、simv 和任何 EDA 都未启动，正式 result、attempt、stage、work、runner log 与 failure quarantine 均不存在。

这不是 C2 RTL 或 exact-cycle 失败。根因是 exact clean environment 设置 `LANG=C, LC_ALL=C`，而本机 `/usr/libexec/platform-python3.6` 的 filesystem encoding 因而为 ASCII。冻结 source map 含真实中文路径 `docs/359_DATE终局冻结_20260813.md`，`validate_source -> _contained -> Path.resolve -> os.readlink` 在读取该路径时抛出 `UnicodeEncodeError`，runner 尚未走到 failure arming。

## 独立复现

源级只读复现得到：

- `LANG=C LC_ALL=C`：filesystem encoding=`ascii`，`validate-source` rc=1，同一 traceback；
- 再加 `PYTHONUTF8=1`：仍为 `ascii`，rc=1；本机 Python 3.6 不支持以此修复 filesystem encoding；
- 改加 `PYTHONIOENCODING=utf-8`：只改变 stdout，filesystem encoding 仍为 `ascii`，rc=1；
- 只在 Python 子进程上覆盖 `LANG=C.UTF-8 LC_ALL=C.UTF-8`：filesystem encoding=`utf-8`，同一真实绝对中文路径通过，40 个冻结 source 全部校验，guard self-test 通过。

因此 `PYTHONUTF8=1` **不是**这台机器上的最小正确修复。正确最小边界是 runner-local `C.UTF-8`，外层 clean environment、RTL/SVA/TB/filelist、五档 exact cycles 和所有 atomic/claim 语义都不动。

## 唯一授权的后继

仅授权新建一个 additive C2 R21 source identity：将 runner 中每个 platform-python3.6 调用（包括 source dry-run self-test、failure receipt、attempt/result guard、inline receipt writer 与 sealing/publish）统一包在 runner-local `LANG=C.UTF-8 LC_ALL=C.UTF-8` 下。必须新增真实绝对中文路径负回归：外层 `env -i LANG=C LC_ALL=C` 的未包装 control 仍应失败，而 additive runner 的所有 Python 入口均须报告 UTF-8 并在 source-only boundary 通过。

后继必须重新经过 fresh source hammer、true release 和 final launch hammer。当前审计不授权 VCS/license/EDA，不创建正式 attempt/result，也不产生任何周期、面积、功耗、系统加速或论文 claim。

`docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
