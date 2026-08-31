# M825｜M819 decoder true-release receipt-blind final hammer

## Decision

`NO_GO_M819_TRUE_RELEASE__P1_1__ADDITIVE_FAILURE_PREFIX_PRECONSUMPTION_GATE_REQUIRED`，92/100，P0/P1/P2 = 0/1/0。不得执行 M819 formal runner，不得消费 attempt，也不得运行 production。

M819 的 SHA、双封、Python 3.10/3.6 release preflight、parent-compatible token、M809 受控穿透、M811/M817 negative authority、M798 permanently-consumed evidence 和 40+120/T10/resource/headline 口径均通过。但 final-release request 要求任意已有 attempt/result/failure 在消费 attempt 前拒绝，release 也明确声明 `preexisting_attempt_result_or_failure_rejects_before_consumption=true`；exact runner 没有实现 failure-prefix 这一项。

## P1 blocker

Runner 第 23 行只生成本次随机 quarantine 名：

`m819_quarantine="${m819_result}.failed_or_incomplete.$$.${RANDOM}.${RANDOM}"`

第 121–130 行只检查 canonical result、attempt，以及本次随机生成的 stage/quarantine/log 精确路径。它没有枚举 canonical `m819_m785_h67_decoder_physical_residency_cycles_r1_20260829.failed_or_incomplete.*` prefix。第 152 行随后就创建 attempt stage。

因此，一个 suffix 不同的既存 failure-prefix 文件、目录或 symlink 不会触发 pre-attempt rejection；runner 会继续进入 attempt 创建/消费路径。这与 release 和 M824 request 的 one-way invariant 不一致。这里不需要也没有运行 formal runner：静态控制流已经精确证明缺失的 predicate。

最小修复是新建 additive runner/release identity，在 attempt-stage 创建之前 fail-closed 枚举 failure prefix，并把 regular file、directory、symlink 三类临时攻击纳入 source hammer。当前 M819 文件不能原地修改，也不能运行。

## Passed evidence

- request、release、candidate、compatibility contract、driver、runner、tests、M821 source PASS100、M823 handoff、M811/M817、M808/M798 的 SHA 与适用双封全部重算通过。
- Python 3.10 与 Python 3.6 均通过 compile 和 exact-release preflight；各自通过 10 个适用 source-only 临时测试。
- 两个 Python 都真正进入 exact-SHA M809 `run_production()`，通过 parent attempt token，在 `output.mkdir` 前受控停止；0 row、output absent、attempt identity drift=false，`finally` 恢复绑定。
- missing/wrong release SHA 均在 attempt 前拒绝；runner/release SHA、sidecar、canonical attempt/result、resource gate 的顺序均在 attempt-stage 创建前。
- runner 是 regular 0664、不可 direct exec；任何未来有效命令都必须是 root-only `env -i` + `/bin/bash -p` + exact runner/release SHA，但本次 NO-GO 不授权或发布该命令。
- 冻结口径仍为 M686 40 + M699 120、T10、A1/K1x8/K8、96 lanes、245760 B、Acc24、3 ns、192 B/cycle；D1 charged/nonheadline，唯一合法 headline 是 typed signed K8 versus equal-service K1x8。
- M819 canonical attempt/result/failure prefix 均 absent；没有 production、VCS、EDA、license、GPU 或 remote；docs/359 SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## Claim boundary

没有 production cycle、speedup、energy、PPA、decoder-complete、full-network、Table-A、system 或 paper claim。raw result 不存在。
