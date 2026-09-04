# M528 r4 最小恢复 author handoff

## 收口结果

R4 source-only 包已完成；没有运行 spawn self-test、schema smoke、production、EDA、GPU 或 RTL。R2 保持永久 consumed/NO-GO，R3 的撤回 review 和 wrapper red-team 均作为双封错误证据原样保留，R3 永久 NO-LAUNCH。

R4 修复两个实质缺口：

1. 旧 analyzer 现在通过脚本目录上的真实模块名 `analyze_m528_h67_single_port_same_ledger_recompute` 正常导入，明确检查 `__name__`、`__file__` origin 和 `c611f8c...afb8a`。禁止虚拟 `spec_from_file_location` 名。这样 production 中传给 `ProcessPoolExecutor(spawn)` 的 `worker_init/worker_phase` 具有子进程可导入的稳定模块身份。
2. 两个 runner 不再把 outer seal 当 PASS。它们先用精确 SHA 的 strict-JSON 工具拒绝 duplicate key，再验证 admission 自身 member/outer 双封，并直接 `jq -e` sealed review/receipt/hammer 的 schema、status、verdict、P0/P1、source tuple、authorization 与 transitive identity。

## 非 production spawn self-test

一次未来 preflight suite 会以单 worker、`spawn` context 和 exact `legacy.worker_init` 初始化子进程；initializer 只打开冻结 row-ledger 文件，不读取 phase。随后 submit 同一稳定 legacy 模块的 `sha256_file(docs/359)` 并取回冻结 SHA。主进程同时 pickle round-trip 检查 `worker_init` 与 `worker_phase`，但绝不调用 `worker_phase`。

因此 receipt 明确区分：`worker_init_called=true`、`spawn_process_pool_created=true`；而 `worker_phase_called=false`、`row_ledger_semantic_read=false`、`row_replay=false`、`production_process_pool=false`、`production_result_created=false`、`production_attempt_consumed=false`。该 preflight 有自己的 attempt/canonical，不消耗 production attempt。

## 证据绑定与旧失败边界

- Preflight runner 必须读取一个全新 r4 static review，要求 exact PASS schema/status/verdict、P0=P1=0、`withdrawn=false`、五份 live source SHA、author outer seal、只允许 root 签一次 preflight admission且 production authorization=false。
- Production runner 除上述 static review 外，还直接解析并验证 preflight admission 自身封条、receipt 三 case、所有 forbidden fields、receipt hammer 的 PASS/P0/P1/receipt outer identity，以及 root 只可签一次 production admission的边界。
- 两个 runner 都直接解析 R3 withdrawal、R3 wrapper red-team 和 R2 consumed-failure review，并现场复验 R2 exact attempt/quarantine 内外封，要求 R2 canonical 继续不存在。

## 保持不变

唯一 SRAM pointer 仍是 `generated_view_inventory.slow.area_um2`，corner 为 `ssg0p9v125c`，面积为 `9 × 8758.3606 = 78825.2454 µm²`。旧 worker/cycle/traffic/capacity/aggregation/decision 主体仍来自 byte-frozen legacy analyzer。row64、B8、128 B/cycle、CAM64、3 workers/chunksize 2、48 GiB commit headroom 等均未改变。

## 下一步独立静态锤审

当前只能进行 source-only static hammer，不得运行 analyzer 的任何模式。评审必须验证 stable import/pickle 路径、两个 runner 的每一个直接语义谓词和 admission 自身双封、R2/R3 现场边界、正常/失败 cleanup 以及所有 SHA。只有 exact expected JSON shape、`withdrawn=false`、P0=0、P1=0 才可让 root 新签一次 non-production preflight admission；static review 本身不授权执行。

## Claim boundary

当前没有 r4 preflight receipt、cycle、speedup、RTL、PPA、energy、full-network/system speedup 或 DATE headline。
