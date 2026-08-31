# M514 exact-SHA VCS runner 独立静态打铁 r2

日期：2026-08-27  
结论：`LITERAL_GO__EXACT_SHA_ONE_SHOT_VCS_WHEN_RESOURCE_GATE_IS_CLEAR`  
评分：**97/100**  
P0：**0**  
P1：**2**  
VCS / DC / DSE / simulator 实际执行：**否**

## Literal verdict

本轮逐字复审的新 runner SHA 为：

```text
fd39ed72a13ceec74a95c5959b90bad24ccce7788e9bb37b16d46ef07f38558b
```

结论是 **literal GO**。r1 的 1 个 P0 与 4 个 P1 均已关闭；未发现修订引入的
`set -e`、grep helper、命令替换、attempt rename 或 complete ordering P0。只能在
M496 DC 已结束且三次 resource gate 实际通过后授权一次 VCS；本评审没有运行
runner，也没有为当前资源状态背书。

caller 必须把上面的 SHA 写成 `M514_EXPECTED_RUNNER_SHA256` 字面量，禁止在同一
命令中用 `sha256sum` 动态求值。VCS PASS 的边界仍然只有 standalone mapper 的
directed functional completeness。

## r1 P0/P1 逐项关闭

### P0｜assertion report fail-open：已关闭

仿真 rc=0 后，runner 先要求 `sim.log` 与 `assert.report` 都是 regular、
non-symlink 文件（215--217 行），再调用 `m514_require_no_match`。helper 显式保存
grep rc，只有 rc=1 才返回成功；rc=0 命中 failure、rc=2 I/O/语法错误都会在
全局 `set -e` 下失败。

本轮只用独立 Bash 片段复现同一 helper 语义，未执行 simulator：clean 输入 rc=0，
含 `Warning:`/failure pattern 的输入 rc=1，不存在文件的 grep-error 输入 rc=1。
因此缺失 report 不再能被 `! grep` 反转成成功。允许存在但为空的 passing
`assert.report`，与当前只有 assertions、无 cover property 的 TB 相容。

### P1-1｜`Warning:` 未捕获：已关闭

compile regex 同时覆盖 `Warning-[` 与 line-start `Warning:`，并继续覆盖
`Error-[`、line-start Error/Fatal 与 `Fatal:`；helper 又把 grep rc=2 作为失败。
合成 `Warning: synthetic` 静态测试被拒绝。

### P1-2｜attempt population/identity 不完整：已关闭

attempt 在 atomic rename 后要求顶层集合精确等于四个 regular、non-symlink
members：`ATTEMPT_CONSUMED.txt`、`identity.sha256`、`SHA256SUMS` 与 outer seal。
identity 现在提交完整八项：runner、VCS binary、RTL、TB、filelist、contract、
独立 static review seal 和 `docs/359`。member seal、outer seal 与 identity 在
publish 前重新验证。

### P1-3｜complete 早于 rename：已关闭

成功路径先要求 canonical 不存在，再执行同文件系统 `mv -T work canonical`，
rename 成功后才置 `m514_complete=1`。rename 失败时 EXIT cleanup 仍能给 work
写 failure marker 并送入 post-attempt quarantine。

### P1-4｜缺 `formality=false`：已关闭

receipt `claim_boundary` 已含 `formality: false`；十个 claim keys 与 contract
完全对应，没有产生 Formality、PPA 或性能正主张。

## 新 shell / one-shot 反向检查

- `bash -n` 通过；runner 自身实际 SHA 与本评审字面 SHA一致。
- `m514_require_no_match` 的 `set +e` 只包围 grep，读取 `$?` 后立即恢复 `set -e`；
  所有调用均处于 runner 的全局 errexit 模式，没有 command-substitution 吞错。
- exact PASS 的 command substitution 若零匹配会以 grep rc=1 退出；多匹配会被随后
  的 count 门拒绝；单行还要通过 Python `fullmatch`。
- attempt 在 VCS compile 前消耗；resource preflight 失败仍只产生 preflight
  quarantine，不消耗 one-shot。
- compile、simulation 都保留真实 rc；输入身份在 gate 前后、simulation 后和最终
  seal 后重复验证。
- work 的全部 regular files 先形成 member manifest，再形成 outer seal并双重
  `sha256sum -c`；只有 presealed work 才能 atomic rename 成 canonical。
- frozen VCS/RTL/TB/filelist/contract/review/docs identities 本轮独立复算均匹配；
  static review 自身 `SHA256SUMS` 也通过。

## 非阻断 P1

1. **attempt exact population 只在创建后检查一次。** publish 前会复验四个已知
   members、identity 和两层 seal，但不会再次拒绝并发新增的第五个未封存 entry。
   单 owner、非对抗运行下不影响可信字段；建议把 exact-set/regular-file helper
   在最终 publish 前再调用一次。
2. **资源排他仍是采样门，不是锁。** 三次 gate 能阻断当时的 DC/VCS/DSE，但第
   三次采样后另一个作业仍可能启动。当前调度纪律下不阻断；更强实现可用同一
   workspace 的原子 EDA lock，并在 compile 前做一次临门复验。

## Claim boundary

该 GO 只授权一次 exact-SHA Synopsys VCS directed run，用于证明固定
K3/S2/P1/OP1 mapper 的坐标、43 taps、phase 计数、stall、same-edge replacement、
ready/valid stability 与 fault-drain 行为。它不证明 full decoder trace/numeric
equivalence、cycle speedup、energy、area、timing、Formality、system speedup、
paper PPA 或 DATE headline。

`docs/359` 未修改；复算 SHA 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
