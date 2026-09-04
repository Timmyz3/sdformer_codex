# M720｜M714-r2 one-shot runner fresh static hammer

## 裁决

**FAIL，86/100；不得运行当前 runner，不得创建 attempt/result。**

本轮是 receipt-blind 静态审阅：只读 capture、contract、runner 与冻结身份文件；没有 import 作者 capture/M366 模块，没有运行 runner，没有查询 GPU，没有调用 EDA，也没有读取任何 M714 result receipt。三份作者对象 SHA 与交接一致，`docs/359` 仍为冻结 SHA。

作者对 M716 的主体修复是有效的：immutable contract、canonical M366 身份、10/105/81/45/36/450 人口与四项零数值门、pattern 守恒、ideal-resource 下界措辞、`17N+12` 不重复配置、build/direct-load 分列、45-config resident 税、one-shot/staging/seal、review SHA 绑定和相对 pointer 都已静态闭合。真实 output miter、可执行周期、RTL/PPA/energy/system headline 继续明确为 false，因而本评审不把“未实现 miter”误判成 pattern capture 的阻塞。

但 PASS 合同要求 P0/P1/P2 全零；当前仍有 1 个 P0 和 1 个 P2。

## P0｜idle process gate 对本项目实际命名 fail-open

runner 第 144–145 行只匹配 token 后立即出现 `/_. -` 或字符串结束的 `profile`、`valid` 等名称，因此以下项目常见名字均不匹配：

- `profile100.py`
- `valid825.py`
- `validate.py`
- `run_date11_ft5_and_valid825.py`
- `run_h67_ep35_profile100_bit_trace.py`

这些命名族在当前工程中真实存在。`nvidia-smi --query-compute-apps` 只能挡住已经建立 CUDA context 的任务；排队、初始化、即将占卡的 profile/valid 进程可能在四次 sample 中仍无 compute-app 条目。此时 runner 会消费唯一 attempt 并启动 capture，违反 contract 中“存在 training/eval/valid/profile process 时禁止启动”的要求。

最小修复是让 matcher 覆盖带数字后缀和项目别名，例如 `profile[0-9]*`、`valid[0-9]*`、`validate`、`trainer/trainonly`，同时保持 fail-closed。runner SHA 变化后必须重新做 fresh static hammer。

## P2｜旧评审 P2-3 没有修

capture 第 129 行仍称 `Deterministic exhaustive-pattern algebra check`，但第 130–153 行仍是固定 seed 的 256 组随机向量。独立重建该 seed 后，2,560 个 scalar draw 碰巧覆盖全部 256 个 signed-INT8 code；然而源码没有显式 `range(-128,128)` exhaustion，也没有 coverage assertion，因此“exhaustive”不是由测试结构保证的性质。

最小修复二选一：

1. 把注释/分类改成 `deterministic randomized DA algebra smoke test`；或
2. 增加显式 256-code scalar exhaustion 与 coverage assertion。

## M716 逐项闭合矩阵

| 旧项 | 本轮结果 | 静态结论 |
|---|---|---|
| P0-1 immutable identity | PASS | contract 固定 capture/M366/M366-contract/M716/docs359，runner 另绑自身 SHA 与 review outer-seal SHA |
| P0-1 四次 idle 在 attempt 前 | **FAIL** | 控制流顺序正确，但 process matcher 漏 `profile100/valid825/validate` |
| P0-2 M366 人口/数值前置 | PASS | 10 samples、105/81/45/36 sites、450 calls、dead-called empty、range/nonfinite/bound/integer mismatch 全零 |
| P0-3 lifecycle | PASS | 唯一 attempt、同 FS staging、failure quarantine、成功 rename、member manifest/outer seal/终态回验 |
| P1-1 output miter | PASS as boundary | `real_output_miter=false`，只授权未来 pattern opportunity，不冒充 exact accelerator |
| P1-2 cycle classification | PASS | 全部候选周期明确为 ideal-resource lower bound，executable=false |
| P1-3 config accounting | PASS | Fixed=`17N+12`；build=`+64/call`；direct=`+23 beats/call`，没有再加五拍 |
| P1-4 resident 45 | PASS | 23/46/92/184 macros 与容量、面积按 P1/P2/P4/P8 单列 |
| P1-5 tp/area | PASS | provisional/incomplete fixed area 只作 diagnostic，PPA 与 admission 均 false |
| P2-1 pattern conservation | PASS | tile/bitplane、histogram、distinct/nonzero、port monotonic、per-site=aggregate 均有 assert |
| P2-2 chunk boundary | PASS | `column_base%16==0`；只有 final chunk 可 pad |
| P2-3 selftest naming | **FAIL** | randomized 实现仍称 exhaustive-pattern |
| pointer rename | PASS | writer 只写 `m714_path.name`，staging rename 后不失效 |

## 独立算术复算

- subset table：`2 × 32 × 10 × 11 = 7040 bit = 880 B`。
- M518 Fixed：N=1 为 29 cycles，N=4 为 80 cycles。
- direct table load：`ceil(7040/256)=28 beats`，相对 M518 已含 5 beats 只加 23。
- resident-45 macro：每个 128×128 macro 放两个 64×110 config，P1/P2/P4/P8 分别为 23/46/92/184 个 macro，即 46/92/184/368 KiB。

机器可读独立复算见 `recompute_m720_static.py` 与 `recompute_m720_static_stdout.json`。本目录双封后，runner 仍会因为 `review.json` 的 FAIL status 与授权位 false 而 fail-closed 拒绝启动。

## Claim boundary

本评审不授权当前 M714-r2 runner、A800 capture、attempt/result、任何 pattern 数字、真实输出等价、可执行周期、RTL/VCS、Synopsys PPA/energy、accuracy、系统倍率或论文 headline。
