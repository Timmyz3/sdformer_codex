# M620｜M617 r5 energy runner fresh static/one-shot hammer

日期：2026-08-28  
模式：`FRESH_INDEPENDENT_STATIC_AND_SYNTHETIC_ONE_SHOT_HAMMER__NO_FORMAL_ANALYZER`  
裁决：`PASS_M620_M617_R5_RUNNER_STATIC_AND_ONE_SHOT_HAMMER`  
评分：**98/100**；`P0/P1/P2 = 0/0/0`

## 1. 裁决

M617 r5 已把 M616 的两个 P0 变成可执行谓词：M615 true-release、M616 FAIL evidence 与未来 M620 PASS
review 的完整 SHA/seal 均为 authorization 必选项；attempt 在任何 analyzer subprocess 之前永久消费，failure、signal、
rename 前后 crash 以及任意 prior blocker 都不能用同一 authorization 重试。

本次只运行 source preflight、synthetic self-test、临时 fake-analyzer/fault injection。formal analyzer、runner
`--execute`、GPU、EDA、remote 均为 0；正式 result/attempt/consumed/staging/quarantine 坐标仍全部 absent。

本 PASS 只允许后续作者另建 exact M621 true-launch admission/release。它本身不授权 formal execution；M621 还必须
绑定本 review 的 review/manifest/outer SHA，并要求 root 在唯一 invocation 紧前重新检查资源、cgroup 与碰撞。

## 2. M616-P0-01 closure｜release/review/evidence executable binding

- M615 true-release 临时副本 baseline 通过；删除、内容篡改分别拒绝。
- M616 FAIL review 双封副本 baseline 通过；删除、内容篡改分别拒绝。
- synthetic future M620 PASS review baseline 通过；review 缺失、seal drift、score=94、FAIL status 四项全部拒绝。
- verifier 冻结 M620 full ID/path、review SHA、manifest SHA、outer-seal-file SHA、score≥95、P0=P1=0，以及 reviewed
  r5 shell/Python/candidate/source-contract/release/M616 identities。

因此 future authorization 不能绕过 fresh M620，也不能删改 M615 release 或 M616 失败谱系。

## 3. M616-P0-02 closure｜permanent pre-analyzer consumption

20 组 blocker 注入全部在 analyzer 前拒绝：result、attempt、consumed、普通/点前缀 result staging、adapter internal
staging、runtime、qraw、qstage、qfinal，每类分别用 regular entry 与 dangling symlink。

fake analyzer 两条路径：

- 返回失败：fake call 时 consumed 已存在且 exact double-seal 通过；失败后 qfinal=1，consumed 不变；同 auth 第二次
  在 fake call 前拒绝。
- 抛出 signal：同样在 call 前 consumed 已双封；signal quarantine 后 consumed 不变、qfinal=1；第二次不进入 fake。

另注入两个 crash 窗口：

- attempt→consumed rename 前失败：sealed attempt 永久留下，第二次被挡，analyzer calls=0。
- rename 已完成、runner 尚未设置本地 consumed flag 时失败：permanent consumed 留下，第二次被挡，analyzer calls=0。

这满足“最多一次 analyzer attempt”，而不只是“最多一个成功 result”。

## 4. 身份、schema 与 claim

- r5 Python SHA：`cc7a721554d3da65a98f80c93c75f5e9c26de9914a68e9991876572d0a8d6844`
- r5 shell SHA：`7e367ffadb3cc8474969954e6fe55bb48b71cc47d6acd4d78c079214bd36c64b`
- source contract SHA：`404b789129b51469ebe81f8620e22c0e551bf60425d3bb56deb8b8191ac60509`
- candidate SHA：`d9f853726cc97a28d4ec7722091350eb9250c8cce547864206373291707f64a1`
- candidate `launch_now=false/release=false`；固定 M612/M597/M615/M616/docs359 identity 与双封通过。
- source/candidate 保持 component-only、per-frozen-sampled-inference、not-camera-frame、not-paper-data、
  no-full-network/system-energy/speedup/headline；M606 exact result schema/equation verifier 仍由 frozen core 提供。
- lineage preflight 与 synthetic self-test 精确 PASS token 均通过。

## 5. 保留边界

98 分而非 100 分只因为 shared-host 资源准入明确留给 root 紧邻执行复核，runner 不宣称控制外部共享任务；这不是
P1。后续 M621 必须 `max_attempts=1`，携带本 M620 exact seals，且任何正式 raw result 仍需 fresh independent result
hammer。

`docs/359_DATE终局冻结_20260813.md` SHA 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
