# M641｜M640/M511 本机 capture 信任链 r4 fresh static hammer

最终结论：`NO_GO__PRE_BODY_SHELL_BYPASS_AND_REVIEWED_IDENTITY_DRIFT`，89/100，P0=0、P1=2、P2=2。`authorized_literal_command = NONE`，不得启动 one-shot capture。

本轮只执行了静态审计、全量重哈希、`bash -n`、Python `compile()`、21 个冻结输入重哈希，以及保证在 attempt 创建前退出的负控。没有运行 production capture、CUDA/checkpoint/model load、production payload verifier、GPU 查询、EDA、DSE 或远端任务。审查前后 canonical capture、fixed attempt、verifier output 均不存在，`docs/359` SHA 仍为 `dedde7ce...`。

## P1 阻断项

### M641-P1-01｜`#!/usr/bin/env bash` 与 `BASH_ENV` 都在正文清洗前执行

M640 冻结的 runner/wrapper 首行均为 `#!/usr/bin/env bash`，正文第 3--5 行才设置并锁定 `PATH=/usr/bin:/bin`。内核先启动 `/usr/bin/env`，而 `env` 会使用调用者的原始 `PATH` 查找 `bash`；因此伪 `bash` 可在正文第一行前执行。安全夹具负控以只打印标记并退出的假 `bash` 证明：victim body 未运行，返回码 77。

同理，非交互 Bash 在读取脚本正文前处理 `BASH_ENV`。hook 可以先执行，再 `unset BASH_ENV ENV`；正文第 6--10 行随后会看到两个变量均为空。安全夹具实际输出同时包含 `M641_BASH_ENV_EXECUTED_BEFORE_BODY` 与 `M641_BODY_SEES_EMPTY_STARTUP_HOOK_VARIABLES`。hook 还能定义未导出的 shell function，避开仅扫描 `BASH_FUNC_` 环境项的检查。

影响：M640 合同的 `path_exported_and_readonly_before_first_external_lookup=true`、`bash_env_must_be_empty=true` 和“启动钩子不可绕过”均不成立。攻击者可以在 host/GPU observation、receipt 和 producer 启动前获得执行权，并清除痕迹后转交真实 Bash。

最小修复：runner/wrapper 使用绝对、privileged shebang（例如已在下一代草稿出现的 `#!/bin/bash -p`），以使 Bash 忽略 `BASH_ENV` 和继承函数；再用全新 exact SHA 更新 verifier、contract、request 并 fresh hammer。不能在 M640 旧合同上口头继承该修复。

### M641-P1-02｜M640 请求冻结的三份 exact identity 已不存在

M640 request/contract 要求 runner `ebf27829...`、wrapper `612fccdd...`、verifier `1054231d...`。最终重哈希时磁盘已是未审下一代 runner `fddf6a0f...`、wrapper `60d12cfd...`、verifier `5f71b537...`；M640 contract 仍为原 SHA `af8da9ed...`，没有同步引用新身份。因此不存在可按 M640 literal command 授权的完整 artifact set。

当前新 runner/wrapper 已改为 `#!/bin/bash -p`，属于正确方向，但它们只能进入下一代合同和 fresh review，不能被本 review 倒签为 M640 PASS。

## P2

1. wrapper 的 `/usr/bin/env` 未使用 `-i` 或显式 allowlist，`PYTHONPATH`、`PYTHONHOME`、`LD_PRELOAD`、`LD_LIBRARY_PATH` 等仍可传入精确 Python。解释器 SHA 与 distribution version 不能封住 sitecustomize、动态库或同版本 install-tree 漂移。下一代应清空/allowlist 并封 effective runtime environment，或明确降级 provenance。
2. `CUDA_VISIBLE_DEVICES`、`CUDA_DEVICE_ORDER` 等 logical-device mapping 未清空或写入 receipt。当前单卡身份降低了风险，但合同不应把物理 `nvidia-smi` 第一行自动等同于 PyTorch logical `cuda:0`。

## 已通过的静态部分

- M641 request 双 seal、M637 outer seal、M636 base contract、producer-r4 outer seal、Python、hostname、nvidia-smi、producer contract 与 `docs/359` 均复算通过。
- producer contract 的 21 个冻结输入逐文件重哈希全通过。
- 在审查时读取的 M640 目标中，正文内 `PATH` 已设置为 readonly；`/usr/bin/env`、`/usr/bin/grep`、`/usr/bin/sed` 绝对化；runner 在 attempt 前和 capture 后做 literal host/GPU gate及工具哈希；verifier要求 exact-nine identity、initial/final receipt exact key/value、双 seal 与 full payload semantics。
- receipt omission/extra/duplicate/value mutation、identity population drift 和 seal mutation均由 verifier 静态 fail closed。
- canonical output/attempt/quarantine 的 preflight 同时检查 `-e` 与 `-L`；EXIT trap 对 dangling canonical symlink 也进入 quarantine 分支。即使 quarantine 不是目录而触发 rc99，canonical 名称仍被移走且不会被 admission。

## 授权边界

`authorized_literal_command = NONE`。M641 不授权 capture、payload verify、decoder cycle fast-kill、RTL、Synopsys、energy、PPA、system speedup 或 DATE headline。下一步只能冻结新 runner/wrapper/verifier 的 exact SHA，修订合同/request 后重新 fresh static hammer。
