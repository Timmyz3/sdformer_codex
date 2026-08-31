# M616｜M614/M612/M593 energy true-launch fresh hammer

日期：2026-08-28  
模式：`FRESH_INDEPENDENT_READ_ONLY_TRUE_LAUNCH_HAMMER__VALIDATE_ONLY__NO_FORMAL_EXECUTION`  
裁决：`FAIL_TRUE_LAUNCH__FRESH_REVIEW_AND_ONE_SHOT_NOT_MECHANICALLY_BOUND__NO_EXECUTION`  
评分：**58/100**；`P0/P1/P2 = 2/0/0`

## 1. 裁决

M614 admission/release、M615 handoff、M612 runner/adapter、M613 PASS100、M597 source、PAFT M614 namespace 与
component-only claim 的静态身份全部正确；当前正式 result/attempt/staging/quarantine 也全部 absent，三次资源样本
过门。

但是 production runner **不读取 companion true release，也不绑定 M615/M616**。独立 validate-only 在本 M616
review 尚不存在时已经返回 admission SHA PASS。进一步在临时 results 命名空间放置一个先前失败留下的
`m612_energy.failed_or_incomplete.*` qfinal，`verify_coordinates()` 仍返回 PASS。因此：

1. release 中的“fresh M616 required”不是执行门，当前 admission 在 M616 前已经可被 runner 接受；
2. release 中的 `max_attempts=1` 不是 one-shot 门；失败后 qfinal 存在、canonical/raw/qstage 消失时，同一 admission
   可再次通过 preflight。

这是 launch authority 的两个 P0。本 review **不授权 root 正式执行**，也不得将 frozen diagnostic 升为 result。

## 2. P0 findings

### M616-P0-01｜fresh M616 / M615 true-release 未进入 runner 信任链

证据：

- `verify_authorization()` 只绑定 M613 `source_static_hammer`；runner/shell 中没有 M614 true-release SHA、M615
  handoff 或 M616 review 的任何引用。
- 在 `reviews/.../m616.../review.json` 不存在时执行只读 `verify_static + verify_authorization`，仍返回
  `0e194055d4a6ac396b091d6c3d0dba61b94d28d0936ecf89352c96e95a23f630`。
- admission 自身 `launch_now=true/release=true`，所以 companion 文档中的 fresh-review 条件无法 fail-close。

影响：本评审前或本评审 FAIL 后，runner 的 executable authorization predicate 仍为真。

### M616-P0-02｜`max_attempts=1` 未执行；失败 qfinal 不阻断第二次调用

证据：

- `max_attempts=1` 只存在于 runner 不读取的 true-release JSON。
- `stale_quarantine_entries()` 仅匹配 `.m612_energy.failed_raw.*` 与
  `.m612_energy.failed_quarantine.staging.*`，不匹配 sealed `m612_energy.failed_or_incomplete.*` qfinal。
- 临时注入 prior-failure qfinal，保持 result/attempt/consumed/staging/raw/qstage absent，原始
  `verify_coordinates()` 返回 PASS。

影响：第一次 authorized invocation 若失败并被正确隔离，永久 consumed coordinate 不会形成；同一个 admission 可
再次执行，违反 release 明示的一次 attempt，而不仅是一次成功 result。

## 3. 通过项

- admission SHA `0e194055...f630`，10-key exact schema、`launch_now/release=true`、runner/M613/M597/canonical/
  component-only 全匹配；sidecar 与 outer seal 通过。
- true release SHA `9f465b9a...fa2`，`max_attempts=1`、`still_not_executed=true`、namespace acknowledgement 与
  component-only 边界文本正确；问题是没有被执行路径消费。
- M615 manifest `0832285e...8d91`、outer-seal-file `b2e574e1...2871`；M613 review/manifest/outer 均匹配。
- energy M614 和 PAFT M614 仅共享数字前缀，完整 ID/path 不冲突；PAFT manifest
  `b77df8fc...e439`、outer-seal-file `3b6676cc...5215` 未变。
- canonical result/attempt/consumed、runner staging、qfinal/qstage/raw 当前均 absent；lexists 口径。
- 三次 UID-local 资源样本最小值：commit headroom `81,749,256 KiB`、MemAvailable `417,404,476 KiB`、
  SwapFree `57,216,508 KiB`；session/user failcnt/under_oom/oom_kill 全 0；UID-local M612/EDA collision=0。
- docs359 SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 4. 必需修复

不得执行当前 M614 admission。下一版必须把 review/release/one-shot 变成 executable predicate，而非 companion
声明：

1. 新 immutable launcher/runner 必须 exact-SHA 绑定 fresh launch review 与 release；FAIL review 必须拒绝。
2. 在调用 analyzer 前以 `RENAME_NOREPLACE` 原子消费一个永久 attempt token；成功、失败、signal 后 token 均保留，
   第二次 invocation 必须在 analyzer 前拒绝。
3. 已有 qfinal 必须阻断启动；同时仍阻断 result/attempt/consumed/staging/raw/qstage 的任何 lexists entry。
4. fresh root live resource/cgroup/collision recheck 必须在唯一 token 消费前紧邻执行；若仍由人工执行，receipt 必须
   绑定样本 SHA 与时间。
5. 新 identity 需要 fresh static/launch hammer 后才可正式运行；当前 admission/release 保留作失败谱系，不得复用。

## 5. Claim boundary

本次 formal analyzer runs=0、runner execute invocations=0、result/attempt/auth mutations=0。冻结
`38.228307918921945%` 与 `1.2622562286593053 mJ/frozen sampled inference` 仍只是预期；不得写成 paper data、
camera-frame/full-network/system/silicon energy，也不得与性能倍率相乘。
