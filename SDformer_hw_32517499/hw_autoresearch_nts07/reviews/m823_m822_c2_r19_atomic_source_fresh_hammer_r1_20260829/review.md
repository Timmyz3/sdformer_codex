# M823 / M822 C2 R19 fresh source hammer

结论：**FAIL_SOURCE_GATE，96/100；P0/P1/P2 = 0/1/0。** M822 已正确修复 M819 的核心问题：四类实际 CLI 双封 receipt 严格得到 `false / false / true / true`，包括 pre-existing exact canonical 的 no-replace collision 保持 source stage 与 destination 均不变且不消费 attempt，以及 rename 后 canonical 被损坏时保守记为已消费。

但 receipt-blind 的 future-chain 负例发现新 P1：`validate_launch_chain()` 对 final hammer 的 `authorization` 只检查 `launch_now=true`，没有闭合 `run_vcs/run_simv/query_license/max_attempts` 或拒绝额外键。因此，一份外封 SHA 被 caller 精确 pin、status/score/P-count/target 全匹配，但内部明确写 `run_vcs=false`、`run_simv=false`、`query_license=false`、`max_attempts=0` 的 final hammer，仍返回 `PASS_M822_R19_EXACT_LAUNCH_CHAIN`。当前不能授权 true release 或 VCS。

## 通过项

- request、author handoff、M819、contract、candidate、runner 双封全部 live replay；contract 的 35 个 source SHA 全过。三份 filelist 无重复、缺失或 symlink。
- M803 RTL/SVA/TB/filelists 未改；五档 exact 周期仍为 K8 `51,131,486,1231,14`，K1x8 `53,133,499,1246,14`。numeric/tuple/weight/stall/full8/out-of-order 门均保留。
- Python 3.6.8 和 3.12.13 均通过内存 compile、12/12 unittest 与 source closure；主机无 Python 3.10 可执行文件。`bash -n`、函数闭包和 undefined-function 负例通过。
- strict parser 拒绝重复 top-level status、嵌套 `authorization.launch_now`、嵌套 identity SHA、NaN、Infinity 和负 Infinity。
- wrong-SHA 在 trace 前 rc=3；positive source dry-run 在 live VCS/license 边界 rc=86，formal attempt/result/quarantine 与工具副作用均为 0。
- attempt 是扁平三件套；attempt/result 均使用 Linux `renameat2(RENAME_NOREPLACE)`；failure primary collision 保留攻击者并原子发布 fallback。
- `docs/359` 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 阻断项 M823-P1-01

独立临时 future chain 使用一份合法 source-hammer、exact true release 和 caller 精确 pin 的 final-hammer outer seal。只把 final review 的 authorization 设为：

```json
{
  "launch_now": true,
  "run_vcs": false,
  "run_simv": false,
  "query_license": false,
  "max_attempts": 0,
  "unexpected_key": "accepted"
}
```

当前 guard 仍返回 `PASS_M822_R19_EXACT_LAUNCH_CHAIN`。status 字符串不能覆盖结构化 authority 的显式否定；否则 final hammer 不是 fail-closed 的第二道批准。

## 裁决与最小修复

本评审不授权 true release、final hammer、VCS/simv/license 查询或任何 EDA。允许创建一个 additive source-only successor：只把 final review authorization 改为严格等于闭合的一次 VCS/simv/license 权限键集、禁止其他 EDA并拒绝额外键，同时新增上述矛盾 authorization 负例。M803 RTL/SVA/TB/filelists、五档周期门以及本次已闭合的 attempt accounting 不得改变；修复后重新做 receipt-blind source hammer。
