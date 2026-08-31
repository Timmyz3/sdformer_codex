# M528 r3 最小恢复 author handoff

## 结果

M528 r3 source-only 恢复包已备妥；本阶段没有运行 schema smoke、production CPU、EDA、GPU 或 RTL，也没有修改 `docs/359`。R2 的 attempt、quarantine、contract、runner 和 admission 均原样保留并永久 NO-GO。

唯一数据修复是把生成 SRAM 面积绑定到 `generated_view_inventory.slow.area_um2`，同时冻结 slow corner `ssg0p9v125c`、9 个 macro、单宏 `8758.3606 um2` 和总面积 `78825.2454 um2`。实现没有 `dict.get`、缺省、顶层 area 或 corner fallback。R3 wrapper 在进入冻结计算前证明上述关系，再仅把已证明的标量交给 SHA 为 `c611f8c...afb8a` 的旧计算；worker、pipeline、cycle、traffic、capacity、aggregation、baseline 和 decision 代码仍由该 byte-frozen 模块执行。

## 新增的 preflight-only 路径

Analyzer 的 `--schema-smoke-only` 路径会检查 execution/governing contract、全部冻结 SHA、M468/M473/M505 的实际消费路径、M473/M505 双封、row-ledger 身份、SRAM schema/cell/shape/corner/pointer/geometry，以及 row64/B8/128 B/cycle/CAM64。该分支不会加载 row worker、不会建立 process pool、不会 replay 51.84M rows、不会创建 production output，只输出一次 `PASS_M528_R3_SCHEMA_SMOKE_ONLY`。

独立 smoke runner 包含三例：精确 pointer/corner 正测，以及 wrong-pointer、wrong-corner 两个负测。它必须等 source-only static hammer 通过且 root 新签 smoke-only admission 后才可运行。当前没有运行回执。

Production runner 也不具备当前授权。未来它必须验证 author/static/smoke/smoke-hammer 四层封条，重新执行三次 48 GiB/128 GiB/32 GiB/OOM 动态门，在创建 r3 attempt 前现场重复同一个正向 schema smoke。该 smoke 失败只会生成 pre-attempt quarantine，不得消耗 production attempt；row replay 启动前才创建 attempt。

## 独立静态锤审请求

请严格 source-only 审阅，禁止运行 analyzer smoke/production runner，并逐项验证：

1. R2 失败证据未删未改，r2 的 contract/admission/attempt 不能满足任何 r3 身份。
2. 真实 mapping 的唯一访问是 `generated_view_inventory.slow.area_um2`，corner/cell/shape/9×面积均严格相等，没有 fallback。
3. Smoke 分支在 `load_legacy()` 和任何 process-pool/row-worker 调用前返回，正测唯一 token，wrong-pointer 和 wrong-corner 必须非零且无 token。
4. R3 production 调用 SHA 冻结旧 analyzer 的 `main()`；除显式 slow-area 兼容标量外，cycle worker、pipeline、traffic、capacity、sample-major/operator-isolated、anchors 与 gates 的规范化语义不变。
5. Production runner 只在资源/碰撞门和现场 smoke 通过后创建 attempt；pre/post-attempt quarantine 分离，成功路径仍可达且 canonical 不被 trap 破坏。
6. 48 GiB commit 门、128 GiB MemAvailable、32 GiB SwapFree、三快照、clean OOM、3 workers/chunksize 2 及 UID-local Synopsys/VCS/simv 冲突门不变。

只有 P0=0/P1=0 的 static hammer 才能授权 root 新建 smoke-only admission。Static hammer 本身不得直接授权 production。

## Claim boundary

当前只有 source。不存在 r3 cycle/speedup、RTL、PPA、energy、full-network/system speedup 或 DATE headline。即使未来 production 成功，仍只是单 sequence、四个 bottleneck Conv 的 raw CPU 模型，必须经独立 result hammer 后才可引用。
