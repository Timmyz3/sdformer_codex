# M633｜M632/M511 本机 exact capture 信任链 r2 fresh static hammer

最终结论：`NO_GO__WRAPPER_EXECUTION_NOT_TRANSITIVELY_ATTESTED`，94/100，P0=0、P1=1、P2=2。本轮严格只做静态源码审计、`bash -n`、Python `compile()`、文件 SHA/旧 seal 复核，以及保证在 canonical output/attempt 之前退出的负控；没有执行 production capture、checkpoint/model load、CUDA、payload verifier production run、VCS/DC/DSE。canonical capture、fixed attempt 和 verifier output 在审查前后均不存在。

## 阻断项

### P1-01｜exact runner 可直接绕过 wrapper，且最终 verifier 无法区分

M632 wrapper 第 21--37 行确实固定了 repo/hostname/GPU name/UUID/driver/memory，并在第 40--45 行向 runner 传入 exact runner/wrapper 环境变量。但 runner 第 64--72 行只验证 **wrapper 文件** 的 canonical path 和 caller 给出的 wrapper SHA；它没有验证 wrapper 是实际父启动链，也没有在 runner 内重复 hostname/GPU identity gate。

这不是单纯的“命令行约定”问题：直接执行未修改的 exact runner，并注入公开的 `M511_EXPECTED_RUNNER_SHA256`、`M511_EXPECTED_REPO_ROOT`、`M632_LAUNCH_WRAPPER_PATH`、`M632_EXPECTED_WRAPPER_SHA256`，可以满足 runner 的 wrapper 门。runner initial identity 第 235--239 行只记录 wrapper 文件哈希；verifier 第 232--273 行也只要求 exact 七文件及 literal runner/wrapper/Python SHA。两者都不能证明 wrapper 第 21--37 行的 host/GPU gate 实际执行过。attempt initial/final receipt 与 verifier 的 exact key set 同样没有 hostname、GPU name/UUID、driver 或 memory total。

因此，一个从 wrapper 入口看不合格的 host/GPU，只要能运行同一 exact runner 和文件树，仍可生成 verifier 接受的 exact-seven attempt。修改 runner 或 wrapper 的攻击已被 verifier literal SHA 阻断，但“绕过 wrapper 直接运行 exact runner”仍能最终 PASS；这正好违反 M632 合同的 `only admissible command` 与 `host_gpu_driver_identity_literal_in_wrapper` 闭合主张。

最小修复不是再增加一个可伪造环境变量，而是把 hostname/GPU name/UUID/driver/memory-total 的 literal gate 移入 exact runner、置于 attempt 前，并把观测值写入 sealed initial receipt；payload verifier 必须要求这些 exact key/value。完成后，直接执行 exact runner 至少不能绕过 wrapper 所承载的语义边界。若仍要声称“wrapper 实际执行过”，还需单独的不可伪造启动证明；否则应把 wrapper 定义为唯一授权入口、把 runner 内重复门作为 artifact admission 的信任根。修改 runner/verifier 后必须重新封 SHA 并做 fresh hammer。

本 P1 未关闭前，不授权 one-shot capture，也不提供字面 wrapper 命令。

## 已通过的核心检查

- request、M632 合同、runner `bc434af6...`、wrapper `a8245a95...`、real non-symlink Python `9f78cd42...`、verifier `1569aca7...`、producer `e16a454d...`、producer contract `e556743d...` 与 `docs/359` `dedde7ce...` 全部复算匹配；M631 outer seal、producer r4 与旧 verifier r3 的 member/outer seals 通过。
- producer contract 的 exact 21 inputs 全量逐文件 SHA 通过；checkpoint 为 591,167,876 B，SHA `4f33e086...`。
- runner 与 wrapper `bash -n` 通过；producer 26,612 B、verifier 31,534 B 均仅以 `compile()` 通过，未 import/执行。
- M631 Python P1 已实质关闭：runner 使用 `/opt/anaconda3/envs/pytorch310/bin/python3.10` real regular executable，启动前拒绝 symlink并固定 SHA；`m511_verify_identity` 在 capture 前两次、capture 后一次重哈希 Python；attempt identity 恰为七项；verifier literal pin Python path/SHA 和 exact-seven set。
- package version 集合在同一个 `m511_verify_identity` 内检查，因此随两次 pre-capture 与一次 post-capture identity gate 重验。版本集合为 torch 2.7.1+cu128、torchvision 0.22.1+cu128、numpy 2.1.2、spikingjelly 0.0.0.0.14、timm 0.6.13、einops 0.8.2、PyYAML 6.0.3、opencv-python-headless 4.11.0.86、h5py 3.16.0。
- output、attempt、随机 quarantine 在 preflight 同时拒绝 `-e` 与 `-L`；attempt 是全部身份、三轮资源、idle gate 之后的单次 atomic `mkdir`；普通 capture 后失败执行 canonical quarantine。
- 三次资源 snapshot 顺序为 1/2/3，前两次间隔 10 秒；每次恰有 commit headroom、MemAvailable、SwapFree、GPU free、cgroup failcnt/under_oom/oom_kill 七字段与阈值，且每次及 attempt 前均执行 idle gate。
- payload verifier 对数据语义没有降级：exact 10 samples x 4 modules = 40 records、696,240,000 bits、87,030,000 B；42 个 capture members、全文件 SHA/size/popcount、逐 timestep 解码、sample-major/module-minor 顺序、21 inputs 与原始 event/mask/flow start/end rehash 仍为 fail closed。
- 安全负控：wrapper 缺失 caller literal、wrapper 错 SHA、direct runner 动态 self-SHA + 错 repo root、direct exact runner + 缺 wrapper env 均 rc=3，且前后 output/attempt 不存在。禁止项使本轮未运行“exact 四环境变量的 direct runner”；该绕过结论由控制流和 verifier admission set 静态证明。

## P2

1. package identity 目前是 interpreter SHA + distribution version，不是 site-package 文件内容/RECORD seal。同版本的本地源码漂移不会被版本字符串发现。payload verifier 的全量 bitpack 检查限制了结果损坏，但不能独立重建 H67 数值语义；建议把关键 wheel RECORD 或安装树 manifest 纳入 runtime identity。
2. preflight 已拒绝 dangling symlink，但 EXIT trap 第 29 行仍只以 `-e` 判断 capture 后出现的 canonical；post-start dangling symlink可能不被 rollback。最终 verifier 会拒绝 symlink，故不会形成错误 admission，但可能留下 DoS/人工混淆；建议 trap 同时处理 `-L`。

## 授权边界

`authorized_literal_command = NONE`。修复 P1、重新绑定 runner/wrapper/verifier/M632 合同并通过 fresh static hammer 前，不得创建 fixed attempt，不得启动 producer 或 production payload verifier。未来 capture 与 verifier 即使通过，也只首先授权 decoder exact envelope repair/同资源 cycle fast-kill；不自动授权 speedup、RTL、Synopsys、energy、PPA、system headline 或 DATE headline。`docs/359` 未修改。
