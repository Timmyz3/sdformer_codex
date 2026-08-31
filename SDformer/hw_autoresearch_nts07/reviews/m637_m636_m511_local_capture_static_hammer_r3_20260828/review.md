# M637｜M636/M511 本机 capture 信任链 r3 fresh static hammer

最终结论：`NO_GO__UNTRUSTED_PATH_CAN_FORGE_ADMISSION_OBSERVATIONS`，93/100，P0=0、P1=1、P2=2。未授权 one-shot capture，`authorized_literal_command = NONE`。

本轮严格限定为静态审计、全量重哈希、`bash -n`、Python `compile()` 和保证在 attempt 创建前退出的负控；没有执行 production capture、CUDA/checkpoint/model load、production payload verifier、GPU 查询、EDA、DSE 或远端工作。审查前后 canonical capture、fixed attempt、verifier output 均不存在。

## P1 阻断项

### M637-P1-01｜未清洗的 `PATH` 可伪造 host/GPU receipt 与资源/idle admission

M636 确实把 literal gate 移入了 exact runner，并在 attempt 创建前、capture 后各执行一次；`/usr/bin/hostname` 和 `/usr/bin/nvidia-smi` 本体也使用绝对路径。但 admission 链并未全部使用不可替换的命令：

- runner 第 118--125 行的四个 `/usr/bin/nvidia-smi` 结果仍交给未固定的 `sed -n '1p'`；
- runner 第 197--203、215--223 行的 GPU-free/cgroup/idle gate 使用未固定的 `awk`/`pgrep`；
- runner 的 runner/wrapper/Python/producer/contract/seal 身份与 attempt seals 使用未固定的 `readlink`、`sha256sum`、`awk`；
- wrapper 第 32--35 行同样以未固定 `sed` 处理绝对 `nvidia-smi` 输出；两脚本均未在任何外部命令前清洗/锁定 `PATH`，wrapper 还把调用者环境原样带入 runner。

因此，直接调用 exact runner 并注入四个公开环境变量时，调用者还可通过 `PATH` 提供同名 `sed`，按输入形态把错误 GPU 的 name/UUID/driver/memory 输出替换成 M636 literal 值；runner 随后会把这些伪造值写入 initial/final sealed receipts。payload verifier 第 197--214、285--308 行只检查 receipt 的 exact keys/literal values，不在 verifier 侧独立重测主机/GPU，所以最终 admission 无法区分真实观察与 PATH-spoof 观察。类似地，伪造 `awk`/`pgrep` 可绕过内存、GPU-free 和 workload-idle gate。

这正违反 M636 `direct_runner_with_injected_public_environment_cannot_bypass_host_gpu_semantics=true` 的信任声明。绝对化 `nvidia-smi` 但保留可替换的下游过滤器，并不构成绝对 observation root。

最小修复：在 wrapper 和 runner 的首个外部命令之前清空 exported shell functions、设置并锁定可信 `PATH`，同时把 admission/identity-critical 的 `readlink`、`dirname`、`sha256sum`、`awk`、`sed`、`df`、`pgrep`、`mktemp`、`mkdir`、`mv`、`date`、`sleep` 等改为绝对系统路径；更稳妥的是避免 `sed`，让一次绝对 `nvidia-smi` 返回四字段后做 shell literal 比较。修改 runner/wrapper 后更新 verifier literal SHA、M636 合同并 fresh hammer。不能只增加另一个公开环境标记。

## 已通过的检查

- M637 request/outer seal、M636 合同、runner `fcfd966a...`、wrapper `8ad12158...`、verifier `e459e5c...`、real non-symlink Python `9f78cd42...`、producer `e16a454d...`、producer contract `e556743d...`、M633 outer seal `0ebc3360...`、producer-r4 seal `1d2334c7...`、`docs/359` `dedde7ce...` 全部复算匹配。
- producer contract 的 21 个冻结输入逐文件重哈希全部通过；没有只信 manifest。
- runner/wrapper `bash -n` 通过；producer 26,612 B、verifier 32,201 B 均只做 `compile()`，未 import/执行。
- M633 的核心语义缺口在无 PATH 攻击时已修复：runner 的 literal host/GPU gate 位于第 116--135 行，第一次执行在 attempt 第 269 行之前，capture 后第 306 行再次执行；observed values 分别进入 initial 第 279--283 行与 final 第 324--328 行。
- verifier 对 initial receipt 要求 exact 16-key set 并逐项等于 `EXPECTED_HOST_GPU_IDENTITY`；final receipt要求 exact 13-key set、同一 literal identity、capture manifest/seal 和 cgroup 前后关系。字段 omission、extra key、duplicate key、value mutation 或 seal mutation均 fail closed。
- attempt identity 仍是 exact seven-file set，并由 verifier literal pin runner/wrapper/Python SHA；文件、路径或 symlink drift 会被拒绝。
- preflight 对 output/attempt/quarantine 使用 `! -e && ! -L`；EXIT rollback 对 canonical output 使用 `-e || -L`，修复了 M633 dangling-symlink rollback P2。
- 安全负控仅覆盖 attempt 前退出：wrapper 缺 caller SHA rc=3；runner 缺 repo root rc=3；runner 错 repo root rc=3。三次均未进入 host/GPU 查询、Python/package 检查或 attempt 创建。审查后 canonical 三路径仍 ABSENT。

## P2

1. package identity 仍是 interpreter SHA + distribution version，不是 wheel RECORD/site-package 内容 seal；同版本本地代码漂移未被 immutable runtime identity 覆盖。
2. `CUDA_VISIBLE_DEVICES` 等运行时 GPU 选择环境未清洗或写入 receipt。单卡主机上当前影响有限，但若设备拓扑变化，`nvidia-smi` 第一物理卡与 PyTorch logical `cuda:0` 可能不是同一设备。PATH P1 修复时应顺带清空/固定该环境。

## 授权边界

`authorized_literal_command = NONE`。P1 修复并重新封 runner/wrapper/verifier/contract、通过 fresh static hammer 前，不得创建 fixed attempt，不得启动 producer 或 production verifier。未来 capture/verify 即使通过，也只先授权 decoder exact payload/周期 fast-kill；不自动授权 speedup、RTL、Synopsys、energy、PPA、system 或 DATE headline。`docs/359` 未修改。
