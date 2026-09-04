# M528 r3 recovery source-only static hammer

## Verdict

**PASS，99/100，P0=0、P1=0、P2=0。** 本评审只放行 root 新签一次 preflight-only、三用例 schema-smoke admission；**不放行 production CPU、EDA、GPU、RTL、论文引用或性能 claim**。

本评审没有运行 analyzer 的任何模式，也没有运行 smoke/production runner。只进行了 sealed-source/hash/key-path/control-flow 审计、`bash -n`、Python AST parse 和 strict-JSON parse。

## 冻结身份

| 对象 | SHA256 |
|---|---|
| r3 analyzer | `a52b4e21bbbe2ab2123763ba0dba7353217fec85f4e8be1c1c24396f2211c0ae` |
| smoke runner | `cf9aaca2178b1e5290490ff720011649f1775493ea06993f27607671e362c126` |
| production runner | `68fed5f590b2c716b000ff94cd79dc7a4646209d0b95786f37752dacf5566685` |
| r3 execution contract | `680a351618fb0cd6e653bc6b2c770d14effa717048bdce67bf9ab98846b8ae65` |
| frozen legacy analyzer | `c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a` |
| author handoff outer-seal file | `b5866f2849f0fd5015a1d7c6b9b23f05ddd00145a8f39b3b0f4a26236db090fd` |
| review request outer-seal file | `1e32c318e7f5fc76b21fea5c902cfdebb88cb078268bf5b0a5091c07073e06e8` |
| docs/359 | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

Author handoff 的 `SOURCE_SHA256SUMS`、内部 manifest、外层 seal，以及 request 的内部 manifest、外层 seal 均通过。`docs/359` 未修改。

## 逐项锤审

### 1. r2 永久 NO-GO 边界

- r2 execution contract/runner/failure review SHA 分别仍为 `fc0c3a...a95e2`、`361525...d386`、`b6508f...c3a2`。
- `results/.m528_h67_single_port_same_ledger_recompute_r2_20260827.attempt_consumed` 仍存在且内外双封通过。
- `results/m528_h67_single_port_same_ledger_recompute_r2_20260827.failed_or_incomplete.3515398.quarantine` 仍存在且内外双封通过；r2 canonical 仍不存在。
- r1/r2 admission 的 schema/status/runner/analyzer/execution identity 均不能满足 r3 smoke runner 或 production runner 的谓词。它们不可重跑、不可改名复用、不可引用。

### 2. 唯一 mapping 修复

- live mapping SHA 为 `68017fb5...be4d`，schema 为 `tsmc28_sram_macro_mapping_audit_v1`。
- 唯一接受的指针是 `generated_view_inventory.slow.area_um2`；slow corner 必须为 `ssg0p9v125c`，cell 必须为 `TS1N28HPCPHVTB128X128M4S`，shape 必须为 `128x128b 1RW SP`。
- 单宏面积严格为 `8758.3606 um2`，宏数严格为 9，静态复算 `8758.3606 * 9 = 78825.2454 um2`，并与 governing contract 相等。
- wrapper 没有 `.get()`、default、fast-corner、顶层 area 或 corner-agnostic fallback。兼容层只在 path-resolved mapping 文件上深拷贝并补入 legacy 所需的顶层 compatibility scalar。

### 3. 所有 live JSON 路径

- governing contract 的全部 frozen input 和 r3 execution contract 的全部 additional input 的实际 SHA 均匹配。
- M468 的所有 point 均具备 mode、240 KiB gate、B、bandwidth、cycle、traffic、DMA、commit 和 capacity 子树；实际存在两个 `strong_zero/B8/128 B/cyc/fits` 候选，最小 cycle 是 `760350133`。
- M473 的 selected point、output-files 和 capacity 子树完整，所有 output 均由双封 manifest 覆盖。
- M505 的 row-ledger、M504/M505 analyzer identity、cycle anchor 与 recurrence CSV 路径完整，M505 双封通过。
- smoke 对 51.84M-row ledger 只做冻结身份的顺序 SHA 读取；不解析/重放 row 语义，不建立 numpy phase array，也不建立 process pool。这符合本轮澄清后的 smoke 边界。

### 4. smoke 控制流与负控

- r3 wrapper 只导入轻量标准库；`validate_schema()` 完成后，`--schema-smoke-only` 在 `load_legacy()` 之前打印唯一 token 并返回。因此 smoke 不会导入 legacy 的 `ProcessPoolExecutor`/numpy/M505 worker，不会创建 phase arrays、pool 或 production output。
- smoke runner 正测固定 exact pointer/corner，并要求 stdout 中 token 恰好一次、stderr 为空、forbidden production output 不存在。
- wrong-pointer 和 wrong-corner 分别传入 `generated_view_inventory.fast.area_um2` 与 `ffg1p05vm40c`；两者在 wrapper 的显式 `require` 上非零失败，且 runner 要求无 PASS token并匹配独立错误文本。
- smoke 的 canonical、attempt sentinel 和 failure quarantine 都是独立 r3 身份；第一次三用例 smoke 会一次性消费 smoke attempt，但不会消费 production attempt。

### 5. production 语义与失败边界

- production wrapper SHA-pin frozen legacy analyzer 后调用其 `main()`；row worker、pipeline、cycle、traffic、capacity、sample-major/operator-isolated aggregation、anchors 和 decision gates 仍由 byte-frozen `c611f8c...afb8a` 执行。
- 仅当 legacy 读取 exact mapping path 时，strict-JSON adapter 注入已证明的 slow-area scalar；其他 JSON 文档不变。
- row64、B8、128 B/cycle、CAM64、3 workers/chunksize 2、48 GiB commit、128 GiB available、32 GiB swap、三快照、clean OOM、UID-local Synopsys/VCS/simv 冲突门均冻结。
- production runner 在 caller SHA/admission、双封链、r2 boundary、EDA collision 与三次资源门通过后，现场重复 exact positive smoke；只有该 smoke 通过后才创建 production attempt sentinel。
- 现场 smoke 或资源失败进入 pre-attempt quarantine，不消费 production attempt；attempt 后失败进入 post-attempt quarantine。成功路径会把全部根日志移入 result、重建双封、检查 work-root 只余 result、原子提交 canonical、删除空 work-root，再解除 trap；旧 r1 terminal-cleanup bug未复现于控制流。

## 结论与 claim 边界

本次只能证明 r3 source 与 smoke 入口可安全执行。不存在 r3 cycle、speedup、RTL、VCS、Synopsys PPA、energy、full-network/system speedup 或 DATE headline。Production 必须等待 smoke receipt 的独立锤审后由 root 另签 production admission。

## Root 的 smoke-only admission 字段模板

```json
{
  "schema": "m528_r3_schema_smoke_static_admission_v1",
  "date": "2026-08-27",
  "status": "AUTHORIZED_ONE_M528_R3_PREFLIGHT_ONLY_SCHEMA_SMOKE",
  "authorization": {
    "schema_smoke_runs": 1,
    "cpu_production_runs": 0,
    "eda_runs": 0,
    "gpu_runs": 0,
    "rtl": false
  },
  "identity": {
    "smoke_runner_path": "system_simulator/scripts/run_m528_r3_schema_smoke_r1_exact_sha.sh",
    "smoke_runner_sha256": "cf9aaca2178b1e5290490ff720011649f1775493ea06993f27607671e362c126",
    "analyzer_path": "system_simulator/scripts/analyze_m528_h67_single_port_same_ledger_recompute_r3.py",
    "analyzer_sha256": "a52b4e21bbbe2ab2123763ba0dba7353217fec85f4e8be1c1c24396f2211c0ae",
    "execution_contract_path": "contracts/m528_h67_single_port_same_ledger_execution_contract_r3_20260827.json",
    "execution_contract_sha256": "680a351618fb0cd6e653bc6b2c770d14effa717048bdce67bf9ab98846b8ae65",
    "author_handoff_path": "reviews/m528_single_port_same_ledger_recompute_author_handoff_r3_20260827",
    "author_handoff_outer_seal_file_sha256": "b5866f2849f0fd5015a1d7c6b9b23f05ddd00145a8f39b3b0f4a26236db090fd",
    "static_review_path": "reviews/m528_r3_recovery_static_hammer_r1_20260827",
    "static_review_outer_seal_file_sha256": "<fill with SHA256 of this review directory's SHA256SUMS.seal.sha256 file after sealing>"
  },
  "expected": {
    "area_json_pointer": "generated_view_inventory.slow.area_um2",
    "corner": "ssg0p9v125c",
    "pass_token": "PASS_M528_R3_SCHEMA_SMOKE_ONLY",
    "cases": 3,
    "production_output_created": false
  },
  "docs359_sha256": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
  "claim_boundary": {
    "smoke_only": true,
    "production_authorized": false,
    "paper_admitted": false,
    "system_speedup": false,
    "date_headline": false
  }
}
```

调用者还必须通过环境变量分别 pin 上述 smoke runner SHA、admission 的仓库相对路径和 admission SHA。Admission 自身必须双封；执行后仍需独立 smoke-receipt hammer。
