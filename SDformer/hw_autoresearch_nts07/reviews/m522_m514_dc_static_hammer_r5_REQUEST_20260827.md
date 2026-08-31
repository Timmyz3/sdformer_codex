# M522/M514 r5 独立静态打铁请求（禁止运行 DC）

目标：只审查 r5 对 r4 唯一 P0（失败 quarantine 未执行零 symlink 收口）的最小修复，以及三项窄 P1 加固。评审前后都禁止运行正向 runner、DC、VCS、PT 或 Formality；不得自授权。

## 冻结身份

- runner：`dc_handoff/scripts/run_dc_m522_m514_c2d_logic_only_exact_sha.sh`
  - SHA256：`375e7602106e46d13520e3e1301254c61e489002030004bac751d7f5fb921a88`
- r5 contract：`contracts/m522_m514_c2d_logic_only_dc_contract_r5_20260827.json`
  - SHA256：`203b3f6b6f3820e2d6266366af3b2b473bb5ba5a8573b1eb7ac82001340ede56`
- r4 独立 NO-GO：`reviews/m522_m514_dc_static_hammer_r4_20260827/m522_m514_dc_static_hammer_r4.json`
  - SHA256：`2566b06d47cc3a37fd74b8eaaa40c3408e940a93c2a63637c7c3a420b846933d`
  - 预期状态：`STATIC_NO_GO__QUARANTINE_ZERO_SYMLINK_CONTRACT_GAP`，P0=1
- Tcl / RTL / filelist / SDC 未改：
  - `dc_handoff/scripts/run_dc_m522_m514_c2d_logic_only.tcl`：`bb749419b25ba91a17cd445a76ee7bc703eabf289fa4769e97c40c71ca8687e8`
  - `rtl_m514/m514_c2_convtranspose_k3s2_polyphase_address_mapper.sv`：`90c44fc9bde839c3cf325ccc8f45c153bf5d30e18de7f39b26d7a4456b017a9a`
  - `dc_handoff/filelists/date_m522_m514_c2d_logic_only_dc.f`：`fc0d31ec1869120528abfbf61736df7ac6828095f6c58f0a4c31edcd892660c7`
  - `dc_handoff/constraints/date_m522_m514_c2d_3ns.sdc`：`9516a8f775ac7e688b9d7813ad613362fd6c03e1548323a2efdce30fdddf3bec`
- `docs/359_DATE终局冻结_20260813.md`：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`
- 新 canonical（必须不存在）：`dc_handoff/runs/m522_m514_c2d_logic_only_dc_3p000ns_r3_20260827`

## 必查 P0：失败 quarantine 的 no-follow 零链接收口

请逐条确认 runner 的 EXIT trap：

1. trap 在四个 input-root verifier 前安装；失败对象只能选择本次 r3 staging 或 post-move r3 canonical。
2. 移动前使用 `os.scandir`、`lstat/stat(follow_symlinks=false)` 递归检查，不调用 `resolve()`，不跟随目录链接。
3. 对每个 symlink 记录 root-relative path 和原始 `os.readlink()` text；unlink 前再次核对 type 和 raw link text。
4. `FAILED_SYMLINK_INVENTORY.json` 与 failure marker 都以 `O_CREAT|O_EXCL|O_NOFOLLOW`（平台支持时）创建为 regular file；JSON 必须 `allow_nan=false`。
5. unlink 后、move 前严格零 symlink；quarantine 目标有 collision gate；move 后再次 `lstat/os.walk(followlinks=false)` 严格零 symlink并重读 inventory schema/status/count/exit code。
6. 任何 sanitizer/inventory/post-move 断言失败都不得发布 canonical PASS；quarantine 仍然不可引用。

请在独立临时目录对上述**原样嵌入代码**做 mutation replay，至少覆盖：零链接失败树、相对 file link、相对 directory link、悬空 link、absolute/out-of-root link、嵌套 link、预占 inventory regular file、预占 inventory symlink、raw link text 在扫描与 unlink 间变化（预期 fail closed）、quarantine collision（预期 fail closed）、move 后注入 link（预期 post-move fail）。不得把临时 mutation 结果写入 production result/canonical。

## 三项 P1 加固

1. collision gate 必须同时包含 `dc_shell`、`dc_shell-t`、直接调用的 `snps_shell`、`fm_shell`、`pt_shell`。
2. sealed-root inventory 和 output manifest 只排除 root-relative `SHA256SUMS` / `SHA256SUMS.seal.sha256`；嵌套同 basename 文件必须作为普通 member 被精确封存或导致拓扑失败。
3. receipt 构造必须重新读取 `precompile_loop_gate.rpt` 和 `constraint_violators.rpt`，唯一解析 TIM-209/OPT-150，要求 0/0，并重新计数五类 clean constraint；不得只写常数。

## 不得退化的原门

- historical M514 VCS root 仍只允许原两条 exact symlink tuple；所有 review roots、r3 staging、r3 canonical、r3 quarantine 都是 zero-symlink。
- runner self-SHA、工具/双库/RTL/filelist/SDC/Tcl/contract、VCS receipt、三组 double seal、docs/359 仍 exact pin。
- resolved `snps_shell`、`SYNTHESIS`、3 ns、ZeroWireload、显式 ideal clock、TIM-209/OPT-150 三源 precompile gate、五类 constraints、mapped outputs、finite JSON、exact topology、staging verify → atomic move → canonical reverify → complete 不得退化。
- 新 receipt 只能是 `m522_m514_c2d_logic_only_dc_receipt_v3`；新 topology 只能是 `m522_exact_output_topology_v2`；成功结果仍是 macro_count=0、paper_ppa_ready=false、cycle/system speedup=false。

## 静态与隔离测试

- `bash -n` 应通过；6 个 embedded Python block 应逐个 `compile()` 通过；contract 应 strict JSON 通过。
- 错误 self-SHA negative preflight 应在 staging/DC 前 exit 10。
- historical VCS symlink policy 的 r4 16-case replay必须保持原结论。
- 确认当前不存在 r3 canonical/staging/quarantine，且没有因为本修复运行任何 EDA 工具。

## 期望独立输出

目录：`reviews/m522_m514_dc_static_hammer_r5_20260827/`

- `m522_m514_dc_static_hammer_r5.md`
- `m522_m514_dc_static_hammer_r5.json`
- `m522_m514_dc_static_audit_r5.json`
- `SHA256SUMS`
- `SHA256SUMS.seal.sha256`

若且仅若 P0=0，可写：

- schema：`m522_m514_dc_static_hammer_r5`
- status：`STATIC_GO__EXACT_SHA_ONE_SHOT_DC_AUTHORIZED`
- `decision.execution_authorized=true`
- `decision.authorized_runner_sha256=375e7602106e46d13520e3e1301254c61e489002030004bac751d7f5fb921a88`

授权范围只能是一次 M514 standalone logic-only 3 ns DC/STA，用于 additive decoder-support area/timing cost。不得准入 full decoder trace、cycle/system speedup、energy、physical SRAM、Formality、paper-ready PPA 或 DATE headline；成功 DC 后仍必须再做独立 receipt-blind hammer。
