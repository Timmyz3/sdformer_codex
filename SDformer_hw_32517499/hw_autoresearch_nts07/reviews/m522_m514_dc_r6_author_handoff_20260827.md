# M522/M514 logic-only DC r6 作者交接

状态：**AUTHOR_COMPLETE__INDEPENDENT_STATIC_REVIEW_REQUIRED__DO_NOT_RUN_DC**。

本次只修复已封存 failure review 认定的唯一 P0，并创建全新的 one-shot 身份；作者没有运行 DC、VCS、PT 或 Formality，也没有自审。

## 冻结身份

- runner：`dc_handoff/scripts/run_dc_m522_m514_c2d_logic_only_exact_sha.sh`
  - SHA256：`1329b1656dff4580a227ab3f5143f4ccc843632536a25e15e2942680dd2d8d5d`
- r6 contract：`contracts/m522_m514_c2d_logic_only_dc_contract_r6_20260827.json`
  - SHA256：`2b450b9fc32436da9c67c820debe6247169a725ef62bfcfcda1ca0b6a18a7215`
- tool-invocation failure review：`reviews/m522_m514_dc_tool_invocation_failure_hammer_r1_20260827/`
  - JSON SHA256：`a4dd356e29681181cd5eb78b795394ed075f12e213e61c862a8585525f27f746`
  - inner manifest SHA256：`9cb2a122ee9d758c92bf9508bdc01ef042f82e7cba44aa7e59a2791be589c480`
  - outer seal file SHA256：`bdead599b2b692c6a67dd0dd096badeb181f9c4feeb41f0edaa5a1025343f3f0`
- resolved `snps_shell` target SHA256：`23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2`
- `docs/359_DATE终局冻结_20260813.md` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

## 唯一机制修复

正向调用现在是：

```text
/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell -f <frozen Tcl>
```

runner 仍要求该路径是 raw link text `snps_shell` 的符号链接，解析目标必须是 `/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell`，并单独冻结目标 SHA。解析目标只用于身份/哈希，不再作为正向 `argv[0]`。runner 不含 `-shell` 调用形式。

## 新结果和单次执行身份

- canonical：`dc_handoff/runs/m522_m514_c2d_logic_only_dc_3p000ns_r4_20260827`
- staging：`dc_handoff/runs/.m522_m514_c2d_dc_r4.staging.XXXXXX`
- quarantine：`dc_handoff/runs/m522_m514_c2d_logic_only_dc_r4.failed_or_incomplete.<pid>.quarantine`
- receipt：`m522_m514_c2d_logic_only_dc_receipt_r4.json`，schema `m522_m514_c2d_logic_only_dc_receipt_v4`
- persistent attempt：`dc_handoff/runs/.m522_m514_c2d_logic_only_dc_r4.one_shot_attempt`

attempt 目录必须最初不存在，在所有输入/授权/资源/负向预检通过后、正向 `dc_shell` 调用前由 `mkdir` 原子创建；成功或失败都不删除、不隔离。并发或重复调用只能有一个创建成功。

## 继承与加强

r5 合同原有 16/16 冻结输入 SHA 原样继承；新增已双封存的工具调用失败 review 三件套，共 19/19 经作者静态重算一致。原 historical VCS exact-two-symlink、review zero-symlink、no-follow quarantine、资源门、进程冲突门、负向 self-SHA、TIM-209/OPT-150、五类 constraint、finite JSON、exact topology、staging 原子发布和 claim boundary 均保留。

结果输入清单还显式加入了原 16 项、r6 contract/runner、r6 static review 三件套、`dc_shell` launcher path、resolved target 和双库；成功 receipt 会分别记录 `invoked_path`、`resolved_path`、resolved-target SHA 和 persistent attempt path。

## 作者静态检查

- `bash -n`：PASS。
- 6 个 embedded Python heredoc 全部 `compile()`：PASS。
- r6 contract strict JSON：PASS。
- 19/19 frozen input SHA：PASS。
- runner 中正向调用 exact `dc_shell -f`：1 处；直接 `snps_shell` 正向调用：0 处；`-shell` 调用：0 处。
- 未创建 r4 canonical/staging/attempt/quarantine；未运行 EDA。

必须由另一位审阅者完成 r6 静态打铁并返回 P0=0，才允许一次正向执行。成功后仍需独立 receipt-blind DC 打铁才可引用 additive decoder-support area/timing；cycle/system/energy/SRAM/Formality/paper-ready PPA/DATE headline 仍全部 false。
