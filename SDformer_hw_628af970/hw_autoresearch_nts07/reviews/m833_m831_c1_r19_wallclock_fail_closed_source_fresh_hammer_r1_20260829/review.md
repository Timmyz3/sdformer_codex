# M833：M831 C1 R19 wall-clock fail-closed source fresh hammer

## 裁决

**FAIL_SOURCE_GATE，98/100，P0/P1/P2 = 0/0/1。**

M831/R19 的功能冻结与 wall-clock fail-closed 实现均通过独立复核：RTL r2、SVA r2、TB r8、宏适配器、binding plan、foundry `UNIT_DELAY` 模型未改；13 个 normal cover、P2、held-final、六攻击、资源门、终态门和双封门未放松。生产 simv 命令仅有一次，严格为：

`/usr/bin/timeout --signal=TERM --kill-after=30s 300s ./simv -no_save`

300 秒只作为基础设施 wall-clock 上界，不是 cycle、性能或 RTL timeout。Python 3.6 的 fast / TERM / TERM→KILL / tee / failure-receipt 测试均通过，TERM/KILL 后无 fake-simv orphan；pre-mkdir dry-run 以 rc86 在 live VCS/license 边界前停止，五类副作用全部为 0。

但本轮不能给 PASS100。冻结请求与 author handoff 都声称 runner 有 94 条 `require_regular_sha` lower-hex edge；独立枚举得到 **95 条唯一 edge**，无重复，全部是 64 位 lower-hex。差异来自最后的 `docs/359` 校验采用跨行 shell continuation：此前计数器只数到了 94 条单行调用，漏计了这一条跨行调用。

这不是少校验，执行路径反而多一条有效冻结边；但它使合同中的“精确 94 条”断言失真，违反本次必须 100/100 且 P0/P1/P2 全零的准入规则。因此记一个 P2，禁止进入 candidate hammer、禁止写 release、禁止 launch。

## 独立机械结果

- TB r8 source-static：PASS；六行 witness parent map 为 `[null, 0, null, 0, 2, 1]`。
- Function closure：34 definitions、266 calls、21 external commands；0 undefined、0 duplicate、0 SHA error。
- delete-definition / rename-definition / inject-stale：三种负变异全部被拒。
- fake simv：fast `(0,0)`、TERM `(124,0)`、KILL `(137,0)`、tee failure `(0,7)`；receipt member seal 与 outer seal 均通过。
- pre-mkdir：rc86；事件为 collision-initial → cgroup → resource → collision-final → live-boundary；VCS identity/license/compile/simv/result mkdir 全为 0。
- Strict JSON：top-level duplicate、nested duplicate、NaN、±Infinity 全部被拒。
- R18、M827、M829：成员与外层 seal 全通过；R18 继续永久 `FAILED_DO_NOT_CITE`，不能归因 RTL。
- Prospective result/source-review/candidate-hammer/release/final-hammer 在本审阅创建前均不存在。
- `docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 唯一允许的下一步

只能新建一个 additive source-only successor：让审计器正确解析 shell continuation，并将冻结口径改为“95 条唯一 edge”，或显式冻结为“94 条单行 + 1 条跨行”。功能源、timeout 命令及所有 fail-closed 门必须保持不变，然后重新请求 fresh independent source hammer。

本轮未运行 VCS、simv、许可证查询或任何 EDA，未创建 result/attempt/release；没有周期、加速、PPA、能量或论文可引用结论。
