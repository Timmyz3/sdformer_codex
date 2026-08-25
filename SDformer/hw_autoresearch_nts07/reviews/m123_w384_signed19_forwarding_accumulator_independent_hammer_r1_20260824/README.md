# M123 W384 signed19 forwarding accumulator independent hammer

## Verdict

**86/100，reset-free 功能里程碑通过；P0=0、P1=1、P2=3。**

M123 确实修掉了 M120 暴露的连续同址 accepted-but-dropped 问题，不是只靠 production SVA 自说自话：

- exact-SHA production sealed VCS 已独立重编重跑，PASS line 和六项 cover 与 sealed run 一致；1072 个正向 accept 对应 1072 次 write，16-deep 同址链有 15 对 forwarding、15 次宏读抑制。
- 独立 standalone hammer 覆盖 M120 原始 A-A 两事件、A-A-A、A-B-A、new/unvalid row、已有 row、不同 bank 同 row、混合正负 delta、signed19 合法边界、两次 forwarding-path overflow、非法 row、end/commit/backpressure。
- 独立 scoreboard 得到 16/16 正向 accept/write、1536 个 write-lane pending-sum 精确检查、6144 个 commit vector 和 589824 个 commit-lane 检查；6/6 同址 pair 均抑制宏读。
- 不修改 production M120 wrapper，只用 review-only module-name shim 将其 M118 instance 替换成 M123，并对原 M120 independent TB 只修改失败预期。event 级反例现在是 2 service accepts / 2 mapped updates / 2 writes，96 lane 均等于两次事件的精确和，无 protocol fault。
- Production assertion covers 非 vacuous：sealed forward-chain cover 14，独立 standalone forward-chain cover 1，II=1、read/write overlap、commit stall、full commit 和 fault covers 均命中。

因此可以说：**在 reset-free、固定一周期同步 1R1W lane-memory 模型下，M123 forwarding 使用上一 pending update 的已计算 signed19 sum 作为下一 base，并闭合了被测试的 M120 同址两事件丢失反例。**

不能说 arbitrary macro、reset exact-once、retry deduplication、DC/PPA 或任何 speedup。

## Scorecard

| Dimension | Score | Evidence |
|---|---:|---|
| 同址 forwarding 算术与 conservation | 30/30 | Standalone 16/16 accept/write；1536 write-lane pending-sum；A-A/A-A-A/A-B-A；正负与边界。 |
| M120 原反例闭合 | 18/20 | Review-only M119/M120 substitution 得到 2/2/2 和 96 lane exact；尚未成为 production M120-r2。 |
| VCS/SVA 与负例质量 | 19/20 | Sealed rerun、独立 scoreboard、non-vacuous covers、overflow/invalid-row、full commit；SVA 本身仍缺 exact write-data/address。 |
| Reset/协议健壮性 | 7/15 | Reset-free 路径通过，但 reset 高电平仍可见 accept，reset edge 仍可写宏。 |
| Macro/claim boundary | 12/15 | RDW-mode independence 成立；固定一周期假设被负例刻画，但无 foundry macro。 |
| **Total** | **86/100** | **功能创新成立；修 reset 并正式集成后再升 production milestone。** |

## P0

**0 个。** 在合同明确承认的 reset-free directed scope 内，没有再找到 accepted positive update 丢失、错误 forwarding base、写地址错误、数值错误、overflow 误写或 commit 错误。

M120 原始同址 event 反例也已在 review-only integrated replay 中由 2/1/1 修复为 2/2/2。

## P1

### P1-1 — reset 不隔离握手和 lane-memory side effects

独立 VCS 做了两个生产 TB 未做的 reset 攻击：

1. 一个合法 update 已 accept、尚在 pending pipe 时，在下一 write edge 前拉高同步 reset。
2. Reset 保持高电平时拉高 `window_start_valid`。

结果：

```text
reset_edge_write_enable=1
reset_physical_writes=1
reset_edge_accept=1
reset_quiescence=false
```

RTL 的 ready/accept、`commit_valid`、`mem_rd_en`、`mem_wr_en` 都没有由 `!rst_core` 显式门控。第一种情况下，状态在 reset edge 被清空，但外部 lane macro 同一 edge 仍看到 write enable 并发生物理写；第二种情况下，上游看到 `window_start_accept=1`，而 reset 分支不会保留这次握手。

这不推翻合同中已经限定的 `reset_recovery=false` 和 reset-free 正向结果，但它会破坏一个可集成的 reset protocol。

要求修复：

- reset 高电平时强制所有 ready/accept、`commit_valid`、macro read/write enable 为 0；
- 增加 reset-quiescence SVA；
- 明确定义 reset 前刚 accept 的 pending update 是 abort，还是必须 drain 后才能确认 reset；
- 重跑 pending-update、commit-stall 和 valid-held-high 三类 reset 边界。

## P2

### P2-1 — M120 closure 目前是 review-only substitution，不是 production M120-r2

冻结的 `rtl_m120/m120_pwp_tail_mapper_signed19_accumulator_island.sv` 仍实例化 M118。本 review 没改它，而是用一个同端口、同 module-name shim 把实例解析到 M123，再复用原 M120 independent TB；diff 只把旧的 2/1/1 failure expectation 改为 2/2/2，并增加 96 lane doubled-sum 检查。

该证据足以说明换入 M123 后原反例可闭合，但还不能叫 production-integrated M120 closure。下一步应生成正式 M120-r2 wrapper，实例化 M123，并用未改 counterexample TB 加 exact-SHA contract 封存。

### P2-2 — 宏接口严格依赖 one-cycle synchronous read

独立正向模型在每个无 read 的 update cycle 主动把 read bus poison 为 X；forwarding 仍通过，说明连续同址计算确实没有偷吃宏 read-data 或 held value。

同一个 simv 改成 two-cycle read model 后，在 A-B-A existing-row 场景立即被 scoreboard 抓住：

```text
M123 hammer forwarded/pending sum mismatch addr=404 lane=0 got=14746 expected=107
```

这证明 M123 对同址 RDW mode 不敏感，但并不 latency-elastic。当前接口没有 response-valid/tag，必须把“一周期同步读、独立 1R1W port”写成硬合同，并绑定具体 3072x19 macro/wrapper；否则不要写 macro-portable。

### P2-3 — Production SVA 非空，但弱于 conservation 文字结论

`ap_every_accepted_update_writes_next_cycle` 的 consequent 是 `lane_mem_wr_en || protocol_error`。任意 protocol fault 都能让它通过；它也不核对 write address/data、不禁止无 prior accept 的 write、不直接证明 forward base 等于上一个 computed sum。全部 20 个 property/cover occurrence 又都 `disable iff (rst_core)`，因此完全看不到 P1。

Production TB 的 1072/1072 conservation 与 full-window numeric miter，以及本 review 的独立 pending-sum scoreboard，足以支撑被测试轨迹；但若要把“每个合法 accept exact-once write”升级成一般性质，应增加 transaction scoreboard SVA/formal：

- accept 后下一拍 exact address/data write；
- 非 overflow/fault 时不得丢，write 不得无 prior accept；
- read/write onehot、forward-data equality；
- reset quiescence 和 reset boundary semantics。

## Coverage matrix

| Scenario | Result |
|---|---|
| M120 original consecutive same-address A-A | PASS，standalone；integrated review 2/2/2 |
| A-A-A | PASS，逐写 pending-sum exact |
| A-B-A on existing row | PASS，A 的第二次从宏读取得已写 base，再进入 forwarding |
| New/logically invalid row with poisoned physical contents | PASS，首更新以 0 为 base |
| Same row in different banks | PASS，不误触发 forwarding |
| Mixed positive/negative lanes | PASS |
| +262143 then -1；-262144 then +1 | PASS，无假 overflow |
| +262143 then +1；-262144 then -1，连续同址 | PASS fail-closed；第二写被抑制，fault sticky |
| Row 384 | PASS fail-closed；0 accept/read/write |
| Full end/commit with stalls | PASS，6144 vectors / 589824 lanes |
| No-read data poisoned | PASS，forward base 不依赖宏输出 |
| Two-cycle macro | Expected fail，固定一周期边界已检出 |
| Reset after accept | Finding：reset edge 仍写宏 |
| Valid held during reset | Finding：可见 phantom accept |

## Safe claim

> Exact-SHA commercial VCS plus an independent pending-sum scoreboard validate reset-free M123 same-address forwarding under a one-cycle synchronous 96×3072×19 lane-memory model. Across the frozen run and targeted hammer, positive accepted updates are conserved to exact writes and full numeric commits, including A-A-A and A-B-A histories. A review-only M119/M120 substitution replays the original same-address event counterexample as 2 accepted services, 2 mapped updates and 2 exact writes. Reset quiescence, production M120-r2 integration, retry deduplication, foundry macros, PPA and speedup remain unadmitted.

## Artifacts

- `m123_w384_signed19_forwarding_accumulator_independent_audit.json`：machine-readable hashes、sealed/independent/integrated counters、covers、P0–P2 和 claim boundary。
- `audit_m123_independent.py`：fail-closed machine audit。
- `independent_vcs/`：standalone directed hammer 与 two-cycle macro expected-fail。
- `m120_integrated_vcs/`：review-only M119/M120+M123 integrated replay。
- `sealed_vcs_rerun/`：production exact-source independent rebuild/rerun。
- `manifest.sha256`：review 封存清单。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
