# M522/M514 r6 独立静态打铁请求（严禁运行 DC/VCS）

请由非 r6 作者独立审阅以下冻结身份：

- runner `dc_handoff/scripts/run_dc_m522_m514_c2d_logic_only_exact_sha.sh`
  - expected SHA256 `1329b1656dff4580a227ab3f5143f4ccc843632536a25e15e2942680dd2d8d5d`
- contract `contracts/m522_m514_c2d_logic_only_dc_contract_r6_20260827.json`
  - expected SHA256 `2b450b9fc32436da9c67c820debe6247169a725ef62bfcfcda1ca0b6a18a7215`
- sealed root-cause review `reviews/m522_m514_dc_tool_invocation_failure_hammer_r1_20260827/`
  - JSON/manifest/outer-seal-file SHA256 分别为 `a4dd356e29681181cd5eb78b795394ed075f12e213e61c862a8585525f27f746`、`9cb2a122ee9d758c92bf9508bdc01ef042f82e7cba44aa7e59a2791be589c480`、`bdead599b2b692c6a67dd0dd096badeb181f9c4feeb41f0edaa5a1025343f3f0`。

## P0 必查

1. 正向 launch 的 `argv[0]` 必须逐字为 `/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell`，调用形态只有 `dc_shell -f <frozen Tcl>`；禁止直接调用 resolved `snps_shell`，禁止 `snps_shell -shell dc_shell`，禁止任何 `-shell` 修补。
2. runner 必须继续要求 `dc_shell` 为 raw link text `snps_shell` 的 symlink，解析为冻结 target；resolved target SHA 必须为 `23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2`。请从安装 wrapper 源码静态证明 `dc_shell` basename 分支构造 `common_shell_exec -shell dc_shell`，且 runner 本身没有绕过 launcher。
3. 新 canonical/staging/quarantine/receipt 必须全部为 r4 身份，旧 r3 quarantine 不得修改或覆盖。新 persistent attempt 必须是独立 r4 路径。
4. attempt 必须由 `mkdir` 原子消费，只能在所有 exact identity、sealed authorization、resource 和 wrong-self-SHA negative preflight 通过后、正向 launch 前创建；任何成功/失败均保留，trap 不得移动或删除它。同一 runner 的第二次正向调用必须 fail closed。
5. r5 原 16/16 frozen input SHA 必须不变；新增 root-cause review 三件套后 contract 应为 19/19。新 review root 必须纳入 zero-symlink sealed-root verifier；成功 receipt 的 sealed input root 数必须为 5。
6. 新 r6 static review 必须被 runner 以 schema `m522_m514_dc_static_hammer_r6`、status `STATIC_GO__EXACT_SHA_ONE_SHOT_DC_AUTHORIZED`、P0=0 和 exact authorized runner SHA 自举校验。不得预建伪 review 绕过。
7. no-follow failure quarantine、wrong-runner-SHA rc10、资源门、进程冲突门、双库、`SYNTHESIS`、3 ns/ZeroWireload/ideal clock、TIM-209/OPT-150、五类 constraints、finite receipt、exact topology、staging seal/atomic move/canonical reverify 全部不得退化。
8. r4 receipt 必须分别记录 launcher `invoked_path`、raw link text、resolved path、resolved-target SHA、attempt path；禁止只记相同 digest 而混淆 launcher 与 target。

## 静态测试要求

- `bash -n`、strict contract JSON、全部 embedded Python `compile()`。
- 19 项 frozen input 独立 SHA 重算；旧 16 项与 r5 contract 逐项键/路径/SHA 对比。
- 不运行 runner 正向路径，不运行 DC/VCS/PT/Formality，不使用开源 EDA。
- 可做隔离的纯 shell/Python 状态机测试，但不得创建真实 canonical/staging/attempt/quarantine。
- 检查 r4 canonical、staging、attempt、quarantine 当前均不存在。
- 检查 `docs/359_DATE终局冻结_20260813.md` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

只有独立结论 `P0=0` 且 review 以 inner manifest + outer seal 封存，并逐字授权 runner SHA `1329b1656dff4580a227ab3f5143f4ccc843632536a25e15e2942680dd2d8d5d`，才允许一次正向 DC。任何失败消费授权；成功后仍需 receipt-blind hammer，且只能引用 standalone pre-macro additive logic area/timing。
