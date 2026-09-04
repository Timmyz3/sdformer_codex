# M767｜M765/r3 C1/M528 macro-integrated DC source/candidate fresh hammer

## 裁决

**PASS / 100，P0=P1=P2=0；但当前不放行任何 DC/EDA。**

本次仅审阅 M765/r3 source-only runner、合同和 `launch_now=false` candidate。
M746/r12 已正确降为失败血缘；未来功能硬门固定为 M758/r13 VCS 结果和固定路径
M766 独立结果评审。两项在本次审阅时均不存在，因此不得创建 M765 true release。

## 已核对

- request、source contract、candidate 与所有依赖均按精确 SHA 重算并通过双封存；
- M757 证明 r12 在建结果目录前失败，M761 为 r13 source/candidate PASS/100；
- M758 runner、已封存 release 和 M763 final-hammer request 的身份一致；
- runner 不内嵌未来 final-review SHA，future release 也被要求不得内嵌；final review
  只绑定已写出的 release，调用方再独立 pin review payload SHA，哈希依赖图无环；
- DC filelist 只含项目 RTL；slow macro `.db` 仅参与 link，行为宏 `.v` 禁止进入 DC；
- 结构门要求 elaboration pre/post 和 mapped netlist 均恰好 9 个宏；unresolved、
  blackbox 或 inferred parent 直接失败；setup/hold 必须 MET，面积必须为正；
- `bash -n` 与 request 自带的静态 NO_EDA selftest 均通过，且未创建任何结果、
  attempt、release 或 final-review 路径；docs/359 SHA 未变。

## 结论边界

本 PASS 只表示 source/candidate 静态证据可以作为未来 release 的输入。它不证明
M758 功能 VCS，不证明 macro-integrated DC、PPA、能量、周期或系统加速。宏只有 slow
DB，因此即使未来 DC 成功，也不构成 macro fast-corner hold signoff 或 paper-ready PPA。

下一步必须先得到 M758/r13 真实 PASS 和
`reviews/m766_m758_m533_r13_unit_delay_vcs_result_hammer_r1_20260828/review.json`
的独立 PASS/100、P0/P1/P2 全零；之后由另一作者写 true release，再做 fresh final-release
hammer，最后才允许一次 DC 尝试。
