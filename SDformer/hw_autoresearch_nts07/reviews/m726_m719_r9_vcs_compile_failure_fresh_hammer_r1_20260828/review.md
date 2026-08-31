# M726｜M719 r9 VCS compile failure fresh hammer

## 裁决

**保留并准入 r9 failure evidence；functional VCS / RTL 功能无结论；r9 永久 consumed。**

本轮为 fresh receipt-blind、只读审阅。没有调用 runner、VCS、simv 或其他 EDA，没有修改 result、作者源文件或 `docs/359`。

评分与严重性分开：

| 项目 | 分数/结论 |
|---|---:|
| 独立审阅置信度 | 100/100 |
| failure evidence 完整性 | 100/100 |
| structured terminal receipt 完整性 | 60/100 |
| failure package 总体质量 | 90/100 |
| compile 根因置信度 | 100/100 |
| functional VCS 准入 | 0/100，NO CONCLUSION |

严重性为 **P0/P1/P2 = 0/2/0**。两个 P1 分别是 TB compile blocker 与 runner failure-receipt binding 缺陷；不存在假 PASS、未封结果或论文误准入，所以不是 P0。

## 1. 结果目录与双封

`results/m719_m533_m528_dead_write_only_1rw_vcs_r9_20260828` 是普通目录。独立回验结果：

- `SHA256SUMS` 的全部 13 个文件通过；
- outer seal 通过；
- `ARTIFACT_INVENTORY.json` 为 failure inventory，含 13 项：11 个 regular file、2 个 directory；
- inventory 与实时目录的路径、类型、bytes 和 SHA 全相等；
- 无 symlink、special object 或未封额外成员；
- `FAILED_DO_NOT_CITE`、`compile.log`、三次 collision、全部 resource evidence 与 partial `simv.daidir` 均在 seal 内。

因此它是**完整、不可引用、可审计的失败证据包**，但不是完整的 structured terminal receipt，后者见第三节。

## 2. 运行阶段与第一根因

三次 collision scan 均 `PASS` 且 `matches=[]`。prelaunch 恰有三样本，memory/swap/commit headroom 均过门；session/user failcnt、under_oom、oom_kill 全为 0。runtime 有 8 次 periodic 和 1 次 final synchronous sample，所有资源计数仍为 0，final ACK 为 sequence 8；资源和碰撞不是根因。

VCS 进入 compile 后恰好报两错：

1. `DTINPCIL`；
2. `IRFPCA-AUTOVAR`。

两者均定位到 TB r4 第 1285 行：

```systemverilog
force dut.slot0_data_q = legal_parent_data;
```

`legal_parent_data` 定义在 `task automatic test_held_final_stale_parent_then_legal` 内。VCS 不允许 task-automatic 变量作为 procedural force/continuous context 的 RHS，因此 compile 以 255 退出。marker 精确记录：

```text
FAILED_DO_NOT_CITE phase=vcs_compile runner_rc=1 child_rc=vcs_255_tee_0 monitor_status=final_sample_ack_pass
```

没有 `simv` 可执行文件，没有 `sim.log`，simv 没有运行。`simv.daidir` 只是 compile 生成的 partial working directory，不能作为 elaboration 或功能通过证据。因此 functional VCS、RTL correctness、coverage、cycles 全部无结论。

## 3. 第二故障：为什么没有 RUN_FAILED JSON

结果中没有 `RUN_FAILED_OR_INCOMPLETE.json`。runner 的精确控制流解释了这一点：

1. VCS 前 runner 已 `cd "${RESULT_DIR}"`；
2. `write_terminal_receipt` 把 `RUNNER_ENV` 设为原始 invocation-form `${BASH_SOURCE[0]}`；
3. 本次以相对 runner 路径启动，进入 result 目录后 Python `Path(relative_runner)` 指向 result 下的不存在路径，触发 `non-regular receipt binding`；
4. cleanup 已执行 `set +e`，Python 失败后函数继续执行 `FAILED_DO_NOT_CITE` 的 `printf`；该 printf 成功成为函数返回值，掩盖了 JSON writer 的失败；
5. 随后的 inventory 和 seal 正常完成，所以 marker 与整个失败目录双封，但 structured JSON/live-launch binding table 缺失。

这降低 terminal receipt 完整性，却没有制造假成功：failure marker 名称和内容明确不可引用，`terminal_kind=failure`，无 `RUN_COMPLETE`，无 simv，所有现存证据已双封。因此 failure evidence 可准入，structured receipt 不可声称完整。

## 4. r9 消费结论

release 的 `max_attempts=1`，消费点是 runner 的 atomic result `mkdir`。精确 r9 result 已存在；runner 第 625 行也拒绝已有 result。故：

```text
r9 = PERMANENTLY_CONSUMED
```

禁止删除、重命名后重跑、覆盖、追加、续跑、就地补 JSON 或复用 partial `simv.daidir`。当前包必须原样保留。

## 5. 唯一最小 r10 修复边界

只允许新开一个 r10 source identity，且只含以下两类修复：

1. **TB-only**：增加 module-scope static `logic [1151:0]` force staging register；在 task 中先填充该静态寄存器，再 `force dut.slot0_data_q = static_stage`。不得改功能 top RTL、SVA、macro adapter 或 binding plan。
2. **runner receipt path**：启动时、任何 `cd` 之前获取并冻结绝对 runner self path，或 canonicalize `${BASH_SOURCE[0]}`；所有 terminal binding 使用该绝对路径。receipt writer 失败不得再被后续 marker printf 掩盖。

r10 必须使用全新 result path，并重做 source-static → candidate → candidate hammer → release → final hammer；本评审只允许写一个最小 r10 source identity，**当前不授权 VCS/EDA**。

## Claim boundary

本 review 准入 failure package 完整性、compile 根因、receipt-binding 第二故障、r9 永久消费，以及一个最小 r10 source-authoring 条件许可。不准入 functional VCS、RTL correctness、trace recurrence、cycle、speedup、PPA、energy 或论文 headline。

`docs/359` 未修改，SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
