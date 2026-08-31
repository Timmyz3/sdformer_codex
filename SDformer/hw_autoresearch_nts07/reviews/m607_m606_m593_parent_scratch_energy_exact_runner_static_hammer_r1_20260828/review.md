# M607｜M606/M593 parent-scratch energy exact-runner fresh static hammer

日期：2026-08-28  
模式：`FRESH_INDEPENDENT_READ_ONLY_RUNNER_STATIC_HAMMER__NO_FORMAL_EXECUTION_EDA_GPU_REMOTE`  
裁决：`FAIL_RUNNER_STATIC__TRUE_LAUNCH_ADMISSION_FORBIDDEN__R4_REPAIR_REQUIRED`  
评分：**91/100**；`P0/P1/P2 = 0/1/1`

## 1. 裁决

M606 已关闭 M604 的两项 P0 和两项 P1：伪造的不完整 PASS tree、错误完成 token、缺失 identity/member
map、错误 schema、CSV/JSON 漂移均被 exact verifier 拒绝；普通目录形态下，post-publish、attempt-seal、
consume 与 post-consume 四个故障点均会把 result/attempt/consumed/staging 收入唯一、可验双封 quarantine；
authorization/static/result/consumed 的 terminal rehash 和 adapter/result/attempt 的真实
`RENAME_NOREPLACE` 也已补齐。

但 fresh hammer 找到一个新的 P1：failure quarantine 会先用 no-replace 移入任何 `lexists` canonical，随后
`seal_tree()` 禁止符号链接。若 result、attempt、consumed、runner staging 或 adapter internal staging 是
live/dangling symlink，canonical 虽已消失，隔离封存会在 `symlink path` 失败，留下未双封的
`.m606_energy.failed_quarantine.staging.*`，且下一次 coordinate preflight 不检查这类残留。请求明确要求所有故障
都进入一个可验证 quarantine，因此 score 不能为 100，M608 true-launch admission 不得起草。

本评审没有运行 formal analyzer，没有生成 canonical result/attempt/consumed，也没有创建授权。冻结诊断值
`38.2283079189%` / `1.2622562287 mJ` 仍不是 paper data。

## 2. P1 finding

### M607-P1-01｜symlink coordinate 使 failure quarantine 留下未封 staging

临时目录 fault injection 分别把 `RESULT`、`ATTEMPT`、`CONSUMED` 与 runner staging 设成 dangling symlink，
调用原始 `quarantine_failure()`：每个 case 都通过 `RENAME_NOREPLACE` 从 canonical 移走，但随后
`seal_tree(qstage)` 因 symlink member 拒绝；最终 `qfinal=0`、`qstage=1`，不存在可验证的 failure receipt seal。
同一结构也覆盖被扫描到的 adapter internal staging。

影响不是 false PASS——canonical 已被移除——而是失败审计链不完整，且 stale qstage 不是下一次 launch 的
阻断坐标。这违反 M607 请求第 2/3 项的全故障隔离合同。

修复要求：对每个待隔离 entry 先 `lstat`；regular directory/file 可按现路径迁移，symlink/特殊类型必须写入
一份 plain-file evidence（类型、原路径、link payload、`lstat` 元数据）后 unlink canonical，或者把 symlink
作为字节证据安全序列化；最终必须无条件生成唯一双封 qfinal，并显式断言没有遗留 qstage 与 canonical。

## 3. P2 finding

### M607-P2-01｜`plain_chain()` 在 `lstat` 前 resolve，live symlink identity path 可被接受

原函数以 `Path(path).resolve()` 开始，因此指向 exact-SHA 普通文件的 live symlink 会被接受；dangling symlink
会拒绝。exact SHA 和 schema 仍阻止内容替换，所以未升为 P1；但 authorization/static identity 应对调用者给定
路径逐段 `lstat`，不能先解析后遗失 symlink 身份。

## 4. M604 findings 逐项复核

- **P0-01 已关闭**：完整 synthetic tree 可过；错误 `RUN_COMPLETE`、缺/多 row/result 字段、错误 frozen source
  identity、CSV/JSON 漂移、空 terminal member map、缺 terminal identity 均被拒绝。
- **P0-02 已关闭（普通 tree）**：四个 post-publish fault point 均使所有 canonical 消失，并形成一个双封
  qfinal；本轮新 P1 仅是 symlink member 的封存边界。
- **P1-01 已关闭**：publish 后和 consume 后均重验 authorization、runner/adapter/upstream/contract、final
  result；consumed attempt exact member set 与双封也被重验，sealed member 漂移被拒绝。
- **P1-02 已关闭**：adapter internal staging publication、runner result publication、attempt consume 全部使用
  `renameat2(RENAME_NOREPLACE)`；临时碰撞测试中 source=`SOURCE`、target=`TARGET` 均保持不变。
- **P2-01 已关闭**：`verify_seal()` 枚举 actual member set 并与 manifest/expected set 精确相等。

## 5. 其他攻击结果

- adapter output/internal-staging 的 regular、live-symlink、dangling-symlink existing coordinate 全部拒绝，目标
  bytes 不变。
- runner result/attempt/consumed/staging 的 live/dangling/existing-file/existing-dir coordinate preflight 全拒绝。
- 无 authorization 的 `--execute` 以 70 在 attempt 前 fail-close；result/attempt/consumed/auth 全 absent。
- shell/source runner/adapter/candidate、M597 analyzer/contract、M602/M604 lineage、handoff/request 与 docs359
  frozen identity 均重新核对；handoff/request 双封通过。
- 合法 `--preflight-only` token 为
  `PASS_M606_M593_SOURCE_PREFLIGHT_ONLY__NO_RESULT_ATTEMPT_OR_LAUNCH`。

## 6. 允许的下一步

不得创建 M608 true-launch admission，不得运行 formal analyzer。只允许新建 immutable runner r4 identity：修复
symlink quarantine，逐段 `lstat` identity chain，清理/阻断 stale quarantine staging，然后由另一名 fresh reviewer
复核。只有新评审 score=100 且 P0=P1=0 才可授权一次 bounded formal run。

`docs/359_DATE终局冻结_20260813.md` SHA 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
