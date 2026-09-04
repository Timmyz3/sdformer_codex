# M522/M514 r4 独立静态打铁请求（禁止运行 DC）

日期：2026-08-27

## 审阅对象与精确身份

- runner：`dc_handoff/scripts/run_dc_m522_m514_c2d_logic_only_exact_sha.sh`
  - SHA256：`c50ce62dcda22c61a1263c2d194f41003036d604b4c7bcc1c271d9be82086005`
- r4 contract：`contracts/m522_m514_c2d_logic_only_dc_contract_r4_20260827.json`
  - SHA256：`e25d21c81a9032c30871d80175c201ef7ea13c768fd42e6cda46485eeb383863`
- DC Tcl（未改）：`dc_handoff/scripts/run_dc_m522_m514_c2d_logic_only.tcl`
  - SHA256：`bb749419b25ba91a17cd445a76ee7bc703eabf289fa4769e97c40c71ca8687e8`
- RTL（未改）：`rtl_m514/m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv`
  - SHA256：`90c44fc9bde839c3cf325ccc8f45c153bf5d30e18de7f39b26d7a4456b017a9a`
- filelist（未改）：`dc_handoff/filelists/date_m522_m514_c2d_logic_only_dc.f`
  - SHA256：`fc0d31ec1869120528abfbf61736df7ac6828095f6c58f0a4c31edcd892660c7`
- SDC（未改）：`dc_handoff/constraints/date_m522_m514_c2d_3ns.sdc`
  - SHA256：`9516a8f775ac7e688b9d7813ad613362fd6c03e1548323a2efdce30fdddf3bec`
- docs/359（未改）：SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

## 第二次 pre-tool 失败边界

请独立核对并明确写入评审：此前两次尝试均在资源门、staging 创建和 `snps_shell` 调用之前退出；当前不存在任何 M522 canonical、staging、quarantine、`dc.log`、`dc.rc` 或 DC receipt，因而没有 DC 结果，也没有消耗正向 DC 执行授权。

r4 绑定的独立失败审计为：

- `reviews/m522_m514_dc_pretool_failure_hammer_r1_20260827/m522_m514_dc_pretool_failure_hammer_r1.json`
  - SHA256：`f18b7a5467793db4dc3ff67475e2f2fd048426ea8a323281cf74143bd0625918`
- 其 `SHA256SUMS`：`b836e84c279ddaf0b7d670ccb31441b6c79c5d216303fc9e46383edf382b1ca9`
- 其 outer seal：`c3947bce6d55257e4585b11fbb1ea724d03cf750abec7b182a6ec72a6174af85`

该旧审计正确证明了“pre-tool、未运行 DC、重复后必须创建带 root/inventory 的 r4”，但其 race 推断不是最终根因。最终根因是历史 M514 VCS canonical 中两个未进入 manifest 的 VCS 自动生成链接；旧 verifier 的 `is_file()` 跟随链接把它们计入 actual extra，随后全局零链接断言再次失败。

## r4 必查 P0

1. 历史 VCS root 只允许下面两条精确链接，必须逐项核对 `link_path`、原始 `link_text`、root 内解析目标、目标是非链接普通文件、目标是 manifest 成员且目标 SHA 相同：
   - `csrc/_1351757_archive_1.so` -> `.//../simv.daidir//_1351757_archive_1.so` -> `simv.daidir/_1351757_archive_1.so`，目标 SHA `be4c425a88be6d5cc24581c8bc8746e855d66ffbc9f5e7f842ef5adee9bd522d`。
   - `simv.vdb/snps/coverage/db/testdata/test/assert.verilog.shape.xml` -> `../../common/assert.verilog.shape.xml` -> `simv.vdb/snps/coverage/db/common/assert.verilog.shape.xml`，目标 SHA `7f9d032a25fef79765e43e9ec60afd7ec8255af2a658702bbb9a57bdef3f8781`。
2. `actual_regular_files` 必须排除 symlink；第三条链接、link text 漂移、出 root、悬空、目录目标、未封目标、目标 SHA 漂移均 fail closed。
3. M514 receipt-blind review、第二失败审计、r4 static review、新 DC staging/canonical/quarantine 全部仍是严格零 symlink。
4. staging 和 quarantine trap 必须在四个 input-root verifier 之前建立；每个 verifier 必须先打印 root/profile 和完整 inventory，并把同一 inventory JSON 留在 staging，最终由 output manifest/outer seal 封存。
5. 错 runner SHA 必须在 staging 和 DC 之前以 10 退出；正确路径必须验证 r4 static review 的 schema/status/P0/授权 runner SHA。
6. DC 输入身份、resolved `snps_shell` 身份、双库、`SYNTHESIS`、3 ns、ZeroWireload、显式 ideal clock、precompile TIM-209/OPT-150 三源门、五类 constraint、有限 JSON、精确 topology、staging verify -> atomic move -> canonical verify -> complete 均不得退化。
7. 新 canonical 必须且只能是 `dc_handoff/runs/m522_m514_c2d_logic_only_dc_3p000ns_r2_20260827`；失败时 staging 或已发布 canonical 必须移入不可引用 quarantine。
8. receipt 只能准入 decoder-support additive logic-only area/timing；cycle/system speedup、energy、full decoder trace、physical SRAM、Formality、paper-ready PPA、DATE headline 全部 false。

请至少静态或隔离复放以下 mutation：VCS root 用 `historical_vcs_exact2` 为 PASS；同 root 用 `zero_symlink` 为 FAIL；零链接 review root 用 `historical_vcs_exact2` 为 FAIL；任一白名单 link tuple 的 path/text/target/SHA 变化为 FAIL。禁止执行正向 runner、DC、VCS、PT 或 Formality。

## 要求的独立产物

目录：`reviews/m522_m514_dc_static_hammer_r4_20260827/`

至少包含：

- `m522_m514_dc_static_hammer_r4.json`
- 人类可读 `.md`
- 静态审计 JSON
- `SHA256SUMS`
- `SHA256SUMS.seal.sha256`

若且仅若 P0=0，可使用：

- schema：`m522_m514_dc_static_hammer_r4`
- status：`STATIC_GO__EXACT_SHA_ONE_SHOT_DC_AUTHORIZED`
- `decision.execution_authorized=true`
- `decision.authorized_runner_sha256=c50ce62dcda22c61a1263c2d194f41003036d604b4c7bcc1c271d9be82086005`
- authorization cardinality：只允许一次新的正向、独立 M514 logic-only 3 ns DC/STA；运行后仍需独立 receipt-blind hammer 才能引用 additive area/timing。

任何 P0、身份漂移或无法证明第二次尝试没有到达 DC 时，保持 LOCKED，不得授权。
