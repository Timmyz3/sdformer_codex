# M473 preflight contract / analyzer 独立 hammer r3

日期：2026-08-26  
裁定：**GO_TO_EXECUTION_CONTRACT_FREEZE**  
评分：**95/100**  

该 GO 只表示 M473 preflight contract 与 analyzer 已足以冻结 execution contract 并启动 CPU DSE；不表示 CPU nomination、RTL、PPA、energy、system speedup 或 DATE headline 已通过。

## 1. 本轮冻结身份

| 对象 | SHA256 / identity | 结果 |
|---|---|---|
| preflight contract | `3c9a66edc2e9bf5dcde4be3a335b0993d66e35accc999021466044b010a2053b` | 与委托一致 |
| analyzer | `e3dab3fbf528e9e3df5365b268af5676804385df9028e81ace3e4cadaf183557` | 与委托一致，Python compile PASS |
| docs359 | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` | 未改 |
| official Prosperity | commit `6ee1c6f1cb419fcf942f2eda63db84ca28248f4b`，clean | PASS |

合同九个文件型 frozen inputs 均逐文件 SHA 匹配。producer、contract、analyzer 与 docs359 均未由本轮复审修改。

## 2. r2 P0：zero-mask parent exclusion 已关闭

冻结 analyzer 现在在 candidate popcount 全零时直接 `continue`，与 official Prosperity 的 `max_subset_size < 1` 规则一致。

独立执行结果：

```text
M473 synthetic self-test PASS cases=5 official_parity=5
```

新增的 `[0,3]` corner 明确要求 residual `[0,3]`、parent `[-1,-1]`；五个 synthetic cases 均直接调用 frozen official API 比较 residual 和 parent，不再只做 XOR reconstruction。r2 的功能、scratch read 和 unfused-cycle 阻塞已实质关闭。

## 3. official parity 与覆盖率 hammer

独立调用 `deterministic_mapping_checks(..., required=128)`：

| 检查项 | 实测 |
|---|---:|
| checks | 128 |
| residual + parent mismatches | 0 |
| samples covered | 0–9（10/10） |
| operators covered | 0–3（4/4） |
| row tiles covered | 32、64、96、128、192、256（6/6） |
| short-final-tile cases | 4 |

分层 mandatory cases 已保证每个 sample/operator pair 至少出现一次，tile 轴轮转覆盖全部六档；其余 case 使用固定 seed 473。128-check full-run 路径满足合同 `required_mapping_mismatches=0`。

## 4. r2 P1 收口

| r2 项目 | r3 证据 | 裁定 |
|---|---|---|
| M468/M41 只核 SHA | 新增 `validate_frozen_semantics()` 并 fail-closed 检查 status、hammer score、19-bit accumulator、M468 signed12 bound 和 128 B/cyc strong-zero anchors | **CLOSED** |
| M468 128 B/cyc anchors | 4-bank `752,580,192` cycles @ row192；8-bank `760,350,133` @ row64，独立调用 validation PASS | **CLOSED** |
| official checks 缺 sample/operator/tile 分层 | 128 checks 实测覆盖 10/10 samples、4/4 operators、6/6 tiles，0 mismatch | **CLOSED** |
| matcher / descriptor scan traffic 缺失 | 新增 `candidate_store_search_read_bytes` 与 `descriptor_order_scan_read_bytes`，并声明 logical on-chip traffic scope | **CLOSED** |
| `bit_frontend_cycles` 名称误导 | point/comparison/output 已改为 `bit_cycles`，另保留 `bit_cycles_without_commit` | **CLOSED** |
| 合同遗留占位/重复 PPA 项 | TO_BE_FILLED 文本已替换为 execution-contract freeze；重复 reduction obligation 已删除 | **CLOSED** |

冻结 semantic validation 的独立返回为 `PASS_FROZEN_SEMANTIC_ANCHORS`，M468 result/hammer/M41 三者均与合同值相符。

## 5. capacity / cycle / traffic 复核

- 12 个 `row_tile × block_bank` 坐标的 logical capacity 由独立公式重算，全部与 analyzer 相等；scratch entries 始终等于 full `row_tile`，peak-live 不参与 gate。
- macro-rounded capacity 仍按 64-depth、144-bit slice 收费，logical 与 macro 两道 240 KiB gate 均保留。
- 4-bank 为两个完整 pass，8-bank 为一个 pass；source/descriptor traffic 在 4-bank 乘二，scratch traffic直接按全部 8 output blocks 计数，没有重复或漏乘。
- product frontend 仍为 capture + `search_rows*ceil(rows/L)` + 17-pass descriptor scan + 2；L 是并行 comparator 数而非 `L*row_tile`。
- fused 与 unfused upper 在同坐标成对；unfused 每 output block增加 parent-edge read 与 active-row completion，成对 nomination 四个门限未放松。
- 独立 pipeline microcase得到 22 cycles，与冻结 `preprocess[0] + Σ(max(work[i], preprocess[i+1])+tail) + work[last]+tail` 实现一致。
- traffic scope 已明确为 logical on-chip access bytes + off-chip weight payload，不冒充 physical SRAM energy 或 DRAM system energy。

## 6. 最多五个 remaining items

当前没有遗留的 preflight P0。以下均不阻止冻结 execution contract：

1. **执行前必需动作，不是 analyzer 缺陷**：生成独立 execution contract，精确冻结上述 preflight/analyzer SHA；当前默认 execution-contract 文件尚未出现，因此不能直接启动 sealed run。
2. **P1 / RTL-open**：fused-forwarded zero-bubble、same-address RAW forwarding、96-lane signed12 row accumulator、dual-update signed19 psum 和 144 B 1R1W scratch 仍必须由新 RTL/VCS 证明。
3. **P1 / PPA-open**：CAM comparator/reduction、17-pass scanner、descriptor store 与 scratch macro 尚未 physicalize；CPU nomination 即使通过也仍是 `performance_admitted=false`。
4. **P2 / receipt hardening**：analyzer 的 checks 列表足以重建覆盖率，但正式 JSON 可额外直接写 sample/operator/tile coverage summary 并 fail-closed assert，方便外部 reviewer 无需重算。
5. **P2 / energy boundary**：新增 traffic 是 logical valid-byte ledger，未计 macro padding/toggle；当前 scope 已诚实，后续 PTPX/宏能量不可直接用这些 bytes 替代。

## 7. 评分与最终裁定

| 维度 | 得分 |
|---|---:|
| official exact semantics | 20/20 |
| coverage / provenance | 18/20 |
| schedule / cycle equations | 19/20 |
| capacity honesty | 15/15 |
| traffic / dual-latency accounting | 14/15 |
| claim boundary / fail-closed discipline | 9/10 |
| **总分** | **95/100** |

**最终裁定：GO_TO_EXECUTION_CONTRACT_FREEZE。** r2 的关键 official-semantics P0 和列出的 P1 均已关闭。下一步应只生成 exact-SHA execution contract 并运行 sealed CPU DSE；跑数后仍需独立 result hammer，任何 1.75x/1.50x 等数字在此之前都不得预写。
