# M647｜M646/M511 allowlisted-environment one-shot 信任链 r6 fresh static hammer

最终结论：`GO__EXACT_M646_LITERAL_COMMAND_ONLY`，97/100，P0=0、P1=0、P2=2。

本轮只做静态审查和保证在 attempt 创建前退出的负控；没有运行 production capture、CUDA/checkpoint/model load、production payload verifier、GPU 查询、EDA、DSE 或远端任务。审查前后 canonical capture、fixed attempt、payload-verifier output 均不存在，`docs/359` SHA 仍为 `dedde7ce...`。

唯一授权命令为：

```bash
M632_EXPECTED_WRAPPER_SHA256=feaeb6247aaf10644bfe7088049f7ab9471dc2d54d928c3fe42210e74265269e hw_autoresearch_nts07/system_handoff/scripts/run_m632_m511_local_rtx3090_capture_exact_sha.sh
```

授权仅适用于当前 exact identity、canonical 三路径仍不存在且启动前主流程重新确认实际环境没有 `BASH_ENV`、`ENV`、`LD_PRELOAD`、`LD_AUDIT`、`LD_LIBRARY_PATH`、`PYTHONPATH`、`PYTHONHOME` 或 CUDA 可见性覆写的单次 capture。不得添加 `env` 前缀、变量、重定向或改变工作目录语义。

## 已关闭的 M643 P1

- runner 与 wrapper 首行逐字节均为 `#!/bin/bash -p`；privileged Bash 不导入真实 ambient `BASH_FUNC_which%%`，wrapper 不再把这个 inert 环境记录误判成攻击。
- wrapper 仍在任何 host/GPU query 和 attempt 创建前拒绝非空 `BASH_ENV`/`ENV`。
- wrapper 通过绝对 `/usr/bin/env -i` 只继承 `PATH=/usr/bin:/bin` 和四个身份变量进入 runner；`PYTHONPATH`、`PYTHONHOME`、`LD_PRELOAD`、`LD_LIBRARY_PATH`、CUDA visibility、startup hook 和导出函数均不会成为 runner 的继承输入。
- runner 保留 startup-hook/exported-function 拒绝、canonical path、自身 SHA、repo root、wrapper path/SHA、固定 Python、host/GPU/tool、21 输入、资源、idle、exact-nine 初末身份、双 receipt/seal 和 rollback 门。
- 当前真实环境只带一个 inert `BASH_FUNC_which%%`，没有上述 Python/loader/CUDA hook；以错误 wrapper SHA 的真实环境负控已经到达 wrapper SHA gate、rc=3，没有误触发函数，也没有产生 attempt。

## 逐项攻击结果

- `BASH_FUNC_m647evil%%`：`bash -p` 不导入函数；到达错误 wrapper SHA gate，rc=3，无 marker/attempt。
- `BASH_ENV` 与 `ENV`：wrapper 正文首门拒绝，rc=3。
- `PYTHONPATH`/`PYTHONHOME`/CUDA 覆写与 `LD_LIBRARY_PATH`：错误 SHA 负控到达 wrapper SHA gate；静态 `env -i` exec 证明这些键不传给 runner。
- fake caller `PATH`、fake command/function：wrapper 立即固定只读 `/usr/bin:/bin`，绝对 shebang `/bin/bash -p` 与绝对 `/usr/bin/env` 不受 caller PATH 控制；错误 SHA 门 rc=3。
- direct runner + public identity variables：错误 runner SHA 在首个身份门 rc=3；即使复述正确公开值也不提供 ancestry/authentication，后续仍须通过 exact host/GPU/tool/input/resource/receipt 语义。这一边界未被包装成不可伪造认证。
- runner/wrapper/verifier/tool/receipt/hash drift：exact SHA 常量、exact-nine identity、initial/final seal、payload verifier 的 exact file set、duplicate-key拒绝和全量语义 decode 均 fail closed。
- rollback：runner 对普通或 dangling-symlink canonical output 都用 `-e || -L` 检查；capture_started 后失败必须 quarantine，成功位只在 post-capture 双封完成后设置。

## 完整静态证据

- M647 request 双 seal、M643 review 双 seal与 outer seal、M511 producer-r4 双 seal全部复算通过。
- exact SHA 匹配 request/contract：runner `fddf6a0f...`、wrapper `feaeb624...`、verifier `d92997e5...`、M646 contract `2ea50673...`、Python `9f78cd42...`、hostname tool `c1f8c2c2...`、nvidia-smi tool `6b8be04c...`、docs/359 `dedde7ce...`。
- runner/wrapper mode 755，`bash -n` 通过；producer/verifier仅用 Python `compile()` 静态编译，分别 26,612/33,194 bytes。
- 冻结 producer contract 的 21 个输入、592,014,785 bytes 全量重新 SHA256，0 mismatch；没有导入模型或 CUDA。
- allowlist probe 的继承键精确为 PATH+四身份变量。Bash 启动后自行生成 `PWD`、`SHLVL` 和 `_`，未从 caller 继承，也不改变 runner 的拒绝/身份语义。
- 所有负控后 canonical capture、fixed attempt 和 verifier output 仍 absent。

## P2 边界（不阻断当前固定命令）

1. 合同 `runner_environment_exact_keys` 的严格字面解释需要收窄成“caller-inherited exact keys”：Bash 运行时必然自行生成 `PWD/SHLVL/_`。这不是外部环境泄漏，也不影响 capture。
2. `env -i` 是 wrapper-to-runner 的继承隔离，不是 hostile same-UID loader sandbox。`LD_PRELOAD/LD_AUDIT` 会在 wrapper 的动态解释器启动前由 loader 读取；错误路径负控可见 loader warning。当前实际授权环境这些变量为空，且项目没有声称 caller ancestry/secret authentication，因此不升为 P1。不得把本 review 描述为抵抗恶意同 UID caller 的安全证明。

## 授权边界

M647 只授权上面的 exact M646 one-shot capture 命令。capture 返回 PASS 后仍必须独立执行 exact `d92997e5...` payload verifier，并对实际 40-record/87.03 MB payload、raw DSEC sources、attempt 双 seal和 semantic decode 做 fresh result hammer，之后才能进入 decoder cycle fast-kill。M647 不授权 cycle、speedup、RTL、Synopsys、energy、PPA、system speedup 或 DATE headline。
