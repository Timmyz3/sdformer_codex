# M803/C2 R16 fresh source hammer

结论：**FAIL_SOURCE_GATE，92/100；P0/P1/P2 = 1/2/0。** M803 的 RTL 修复本身通过独立静态审查：它仅把 M499 R5 的 request/response channel split 迁移到冻结 M490 的 same-cycle reuse 结构，K8 top 仅替换 adapter，K1/K1x8 仍是冻结 M519。但未来 VCS runner 的 canonical 发布可在目标碰撞时返回成功并嵌套 work tree，因此不得授权 candidate/release，更不得运行 VCS。

## RTL 与验证源审查

- request、author handoff、contract 和 runner 的内外层 seal 全部重算通过；contract 的 35 项 source SHA live replay 全过。`docs/359` 仍为 `dedde7ce...`。
- 冻结 M490/M499/M519 K1/M519 K1x8 分别仍为 `597e4d9e...` / `44f7df33...` / `6ea038ef...` / `11080d39...`，旧源未改。
- M803 adapter 对 M490 的功能差异只有：新增 M499 R5 式双通道门控；response state 先于 request state 更新；保留 `req_slot_open = !slot_valid || (core_rsp_accept && same_slot)`。`core_rsp_valid` 只依赖 response/寄存 slot 状态，不再依赖当拍 `illegal_request`，M800 的最小组合环在源级被切断。
- K8 top 相对冻结 M519 只改 module identity 和 adapter binding，仍绑定 `m519_fc2_registered_release_standalone_raw4_acc24`。matched shell 只在 ARCH_MODE=1 换 M803；ARCH_MODE=0/2 仍绑定冻结 K1/K1x8。
- attack TB/SVA 对 illegal request + legal cut-through/held response、illegal response 双通道关闭、pending、backpressure、same-slot reuse、sticky/reset 和 bundle/bank conservation 都有硬门与 cover。
- full-workload TB 相对冻结 M519 TB 未删减旧数值/元组/权重/攻击/stall/full8/out-of-order 门，并把五组 K8/K1x8 周期硬锁为 `51/53, 131/133, 486/499, 1231/1246, 14/14`。

## 源级复现

- `bash -n` PASS。
- 实际使用 Python 3.6.8 重跑 closure：函数闭包、undefined-function 负例、source SHA 闭包均 PASS。
- wrong-runner-SHA 在 trace 前返回 3；positive dry-run 在 live VCS/license boundary 前返回 86，五个 stub event 顺序精确，VCS/license/simv/result/attempt 副作用全为 0。
- 本评审后 prospective result/attempt/candidate/release/final-hammer 仍全部不存在。

## 阻断项

1. **M811-P0-01：canonical result 发布不是 no-replace。** Runner 第 308 行是 `mv work result`，只依赖很早之前的目标不存在检查。独立临时目录攻击证明：在 precheck 后创建 `result/`，GNU `mv` 仍返回 0，产生 `result/stage/RUN_COMPLETE.txt`。Runner 随后设 `complete=1` 并撤 trap，可以成功退出，但 canonical root 没有回执和 seal。这是可发布假 PASS 的 P0。
2. **M811-P1-01：attempt 消费与 failure trap 之间有窗口。** 第 247–253 行先直接创建最终 attempt 目录，然后创建 work、安装 trap，最后才写 `ATTEMPT.txt`；attempt 也从未双封。这个窗口内失败会留下空/可变 attempt，却无封存 failure receipt。
3. **M811-P1-02：JSON authority 允许重复 key。** `verify_source_contract` 与 `json_gate` 使用普通 `json.loads`，Python 会 last-key-wins；现有负例没有覆盖 duplicate `status` / `launch_now` / SHA identity。

## 裁决

M803 RTL/SVA/TB 可保留，无需改算术、端口、K8/K1x8 周期或冻结 K1/K1x8。仅允许一个新身份的 source-only runner/contract 修复：attempt 先 staging+双封+原子 no-replace，所有消费后失败都发布封存 failure receipt，canonical result 用 `renameat2(RENAME_NOREPLACE)` 类语义并复核 root 四件套，JSON 拒绝重复 key。新源仍需 fresh hammer。本评审**不授权 VCS candidate、release、VCS/simv/license 查询或 DC**。
