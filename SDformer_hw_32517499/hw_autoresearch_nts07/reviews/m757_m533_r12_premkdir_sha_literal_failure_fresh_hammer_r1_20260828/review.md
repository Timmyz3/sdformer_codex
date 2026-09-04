# M757｜M746/M533 r12 pre-mkdir SHA literal failure fresh hammer

裁决：`PASS_FAILURE_AUDIT__M746_R12_BLOCKED_PRE_MKDIR__M743_MANIFEST_SHA_LITERAL_MISSING_ONE_NIBBLE__ADDITIVE_R13_ONLY`，100/100；P0/P1/P2 = 0/1/0。

## 1. 直接根因与 fail-closed 边界

r12 runner 第 314 行写入的 M743 `SHA256SUMS` 期望值是 63 位：

```text
626ba66587e86885020031ef5656c3cd971cdac803bc339b218d1171d796962
```

双封存 M743 manifest 的实际 SHA256 是 64 位：

```text
626ba66587e86885020031ef5656c3cd971cdacb803bc339b218d1171d796962
```

唯一差异是实际值第 40 位的 `b` 在 runner literal 中漏写。`require_regular_sha` 在第 149 行做严格字符串相等检查，因此已观测的唯一命令在 `phase=pre_mkdir_identity_gate` 返回 1 是确定性的。

失败点早于第 818 行 preflight `mktemp`、第 912 行原子 result `mkdir`、第 996 行 VCS 与第 1008 行 simv。审计时 r12 result、attempt sentinel、M746 run-root artifact、preflight 临时目录、compile/sim log 和 simv 均不存在；没有 M746 runner/VCS/simv 进程。系统上有一个 UID 1909 的外部长期 `simv`，它不是本用户、不是 M746，也没有改变上述结论。

所以 r12 没有消费 attempt identity，没有形成 VCS、功能、周期、PPA 或论文证据。这里不存在可引用的失败结果包，因为 runner 在 result identity 创建前即 fail closed。

## 2. M749/M753 为什么漏检

M749 与随后写到固定 M746 输出路径的 M753 final-release hammer，分别验证了两个端点：M743 包自身的双封存是好的，整个 runner 文件的 SHA 也确实与 contract/release 绑定一致。但它们没有验证这两个端点之间的交叉边：

1. 没有枚举 runner 内全部 `require_regular_sha` expected literal；
2. 没有强制每个 expected token 恰为 64 位小写十六进制；
3. 没有把每个 literal 与其引用文件的实时 SHA256 逐项对照。

`bash -n`、runner whole-file SHA 和 M743 self-seal 都无法发现“冻结 runner 内部冻结了错误 literal”。这是静态锤覆盖缺口，不是 TB r7、SVA、foundry 模式、claim boundary 或其他已完成检查的反证。因此不撤销 M749/M753 的其他检查，只撤销 r12 launch authorization，以及“该 runner 能通过 pre-mkdir identity gate”的推断。

## 3. 唯一合法 successor

不得原地修改或重跑 r12。唯一授权是建立 additive r13 身份：在新 runner 中仅把上述 63 位 literal 插入第 40 位 `b`，并作新 runner/contract/candidate/release/review/result/attempt/receipt 的必要身份改名和绑定。

RTL r2、TB r7、SVA r2、macro adapter/binding、foundry `.v`/`.db`、`+define+UNIT_DELAY`、资源与碰撞门、原子 ownership、terminal receipt、R7 PASS/COVERAGE、六种攻击、failure signature 和全部 watchdog 必须冻结。新静态锤必须新增“全部 `require_regular_sha` literal 恰为 64 hex 且逐一等于目标文件摘要”的机械规则。

M757 不授权 r13 立即运行；fresh source/candidate hammer、独立 release、fresh final hammer 全部闭合后，才可授权最多一次 VCS+simv。所有 DC/Formality/PT/PTPX/CPU/GPU/remote 运行继续为零。

`docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
