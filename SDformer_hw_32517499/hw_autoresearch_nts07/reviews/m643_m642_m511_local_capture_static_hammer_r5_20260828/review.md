# M643｜M642/M511 privileged-Bash one-shot 信任链 r5 fresh static hammer

最终结论：`NO_GO__AUTHORIZED_LITERAL_COMMAND_REJECTS_ACTUAL_AMBIENT_FUNCTION_ENV`，94/100，P0=0、P1=1、P2=2。`authorized_literal_command = NONE`，不得启动 one-shot capture。

本轮严格停在静态审计和 attempt 前负控：没有执行 production capture、CUDA/checkpoint/model load、production payload verifier、GPU 查询、EDA、DSE 或远端工作。审查前后 canonical capture、fixed attempt、payload-verifier output 均不存在，`docs/359` SHA 仍为 `dedde7ce...`。

## P1 阻断项

### M643-P1-01｜合同唯一字面命令在真实调用环境不可执行

M642 的 privileged shebang 修复本身有效：runner/wrapper 首行都精确为 `#!/bin/bash -p`；对照夹具证明普通 Bash 会执行自清洗 `BASH_ENV` hook并导入环境函数，而 privileged shebang 不执行 hook、也不导入函数。恶意 caller `PATH` 也无法把固定 `/bin/bash` 换成假 `bash`。

但是当前 Codex Bash 环境天然携带 `BASH_FUNC_which%%`。privileged Bash 正确地**没有导入**该函数，但保留了环境记录；runner/wrapper 正文第 6--10 行仍以 `/usr/bin/env | /usr/bin/grep '^BASH_FUNC_'` 拒绝任何记录。因此 M642 合同中唯一允许的原样命令：

```bash
M632_EXPECTED_WRAPPER_SHA256=60d12cfdd977af670cf461d647fe1a8f6d5922e6a854ac237c3a4b583619c720 hw_autoresearch_nts07/system_handoff/scripts/run_m632_m511_local_rtx3090_capture_exact_sha.sh
```

在目标调用环境实测输出 `M632 refuses startup hooks or exported shell functions`、返回 3，并在任何 host/GPU query、preflight 或 attempt 创建之前退出。三 canonical 名称保持 absent。安全没有失守，但本 review 不能授权一条确定无法消费 one-shot 的命令。

最小修复：在下一代合同中明确用 `/usr/bin/env -i` 建立 allowlist 环境后启动 wrapper/runner，或在已经用 `-p` 证明函数不导入的前提下删除对残留 `BASH_FUNC_` 环境记录的致命拒绝；随后更新 exact SHA 并 fresh hammer。不得把额外的 `env -u ...` 前缀临时加到 M642 已冻结字面命令上。

## P2 非阻断项

1. wrapper 到 runner 的 `/usr/bin/env` 仍非 `-i`，`PYTHONPATH`、`PYTHONHOME`、`LD_PRELOAD`、`LD_LIBRARY_PATH` 与 CUDA logical-device 变量未进入 allowlist/receipt。本项沿用 M641 的 provenance 降级；本轮没有扩展为新攻击面。
2. runner 所谓 caller proof 由公开路径和公开 SHA 环境变量组成，直接调用者可以复述；它不是进程 ancestry 或密钥认证。runner 的 host/GPU/tool/identity 检查与 wrapper 重复，因此这不构成本轮 admission 绕过，但论文/回执不得把它描述成不可伪造的 caller authentication。

## 已通过的 REQUEST 范围

- M643 request 双 seal、M641 outer seal、producer-r4 双 seal复算通过；runner `fddf6a0f...`、wrapper `60d12cfd...`、verifier `5f71b537...`、M642 contract `d07f4695...` 与 `docs/359` 精确匹配请求。
- runner/wrapper 均为 mode 755，第一行逐字节为 `#!/bin/bash -p`，`bash -n` 通过；producer/verifier仅做 Python `compile()`，分别 26612/33194 bytes。
- 21 个 producer contract 冻结输入共 592,014,785 bytes 全量重哈希通过；没有调用模型或 CUDA。
- 对照负控确认：无 `-p` 时 `BASH_ENV` hook先于正文执行；有 `-p` 时 hook标记完全不存在，正文仍看到非空变量，production wrapper因此 rc3。无 `-p` 时环境函数被导入并运行；有 `-p` 时 `type` 查不到该函数。
- clean-env fake `PATH` 负控没有命中假 `bash`，而是在正文的错误 wrapper SHA 门返回 3；clean-env exported-function 负控在正文返回 3。所有负控均在 attempt 创建前。
- runner静态存在 preflight 前、attempt 前和 capture 后的 identity/host/GPU检查；hostname、nvidia-smi、Python、wrapper、producer、producer contract、producer-r4 outer seal和 `docs/359` 被封为 exact-nine identity。
- verifier静态要求 attempt 精确文件集、initial/final 双 seal、exact-nine绝对路径和 SHA、host/GPU receipt、capture manifest/seal绑定。receipt增删、重复键、值变异、tool/runner/wrapper mutation都会 fail closed。
- EXIT rollback同时检查 `-e` 与 `-L`，capture_started 后的普通或悬空 canonical output 都进入 quarantine 分支；成功位仅在最终 receipt 双 seal后设置。

## 授权边界

`authorized_literal_command = NONE`。M643 不授权 capture、payload verification、decoder cycle fast-kill、RTL、Synopsys、energy、PPA、system speedup 或 DATE headline。下一步只能修复实际环境可运行性、更新 runner/wrapper/verifier/contract/request exact SHA，并再次 fresh static hammer。
