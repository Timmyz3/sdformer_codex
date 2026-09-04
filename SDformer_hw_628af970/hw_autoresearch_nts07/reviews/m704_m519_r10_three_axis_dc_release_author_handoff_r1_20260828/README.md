# M704：M519 R10/r3 DC 发布链作者交付

## 结论

这是 M701 pre-EDA shell failure 之后的 additive R10/r3 identity。R9、R8、M580 与 `docs/359` 均保持不可变；作者只进行了语法、身份、封存和 no-EDA 故障注入测试，没有启动 DC、VCS、PT 或 Formality。

R10 runner 修复了 `set -u` 下同一 `local` 声明中先声明 `payload`、同时用它展开 `sidecar` 的缺陷。静态审计又发现并修复了同类的 `id/point` 声明依赖。runner 现在在 admission、资源 preflight、attempt 消耗和工具路径之前先执行 `bash -n`，并提供可注入的 no-EDA self-test。

## Pre-attempt fail-closed

在 attempt 尚未消耗时，任何 shell failure 都会写入 fresh、noncanonical receipt，并生成 `SHA256SUMS` 与 `SHA256SUMS.seal.sha256`。注入退出码 86 的测试已验证该回执，且 R10 canonical 和 attempt sentinel 在测试前后均不存在。

## 冻结身份

- runner：`7dc7d79c27b85820c621ac142e104cc155afd949ffa8b2ec46dd7279a314d27f`
- recovery contract：`2ba563ed4c3ddb2c89d0a13855bb4b11be7522aef505cfe1ef374a33b5501a4e`
- launch admission：`f4bccc501dea216396d2755ef6b1f627209efe18346701cd5d448367cf4a3424`
- 作者 no-EDA test：`1f0f8426002ef3412341fedf59cd1f400fb38ff1cdabc1ccab58b9b7cf3e6817`

contract 与 admission 均逐项绑定 M694 和 M701 的 review、manifest、outer-seal exact SHA 与 exact status，并在 runner 中交叉相等。M694 的历史 P2（共享主机同 UID foreign EDA）不被消除或降级；同 UID collision、64 GiB commit headroom、128 GiB MemAvailable、32 GiB SwapFree 和 runtime exact-birth-tuple gates 全部保留。

## 下一步

必须由另一个 fresh reviewer 重跑 `static_no_eda_test.sh`，检查 P0/P1、实时同 UID EDA collision 与资源状态。只有该 reviewer 给出精确状态 `GO_ONE_M519_R10_DC_ONLY_ATTEMPT__FINAL_LIVE_RECHECK_REQUIRED` 后，才可发布唯一的 `env -i` 一次性 DC 命令。作者包本身不授权执行，也未消耗 attempt。

## Claim boundary

当前没有 R10 DC 结果；area、timing、power、energy、throughput/area、完整 FC2 与系统加速均为 false。本包不能作为论文 PPA 或性能证据。
