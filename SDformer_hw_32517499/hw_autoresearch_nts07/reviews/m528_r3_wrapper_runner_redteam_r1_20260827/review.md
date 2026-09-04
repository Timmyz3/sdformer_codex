# M528 r3 wrapper/runner 独立 source-only 红队

## 裁决

**61/100，P0=3、P1=2、P2=2。当前 r3 不允许签 smoke-only admission，也不允许 production。** 本审阅没有运行 analyzer、smoke、production、process pool、EDA/GPU，也没有修改被审源码或 `docs/359`。

最严重的问题不是 M505 算术，而是恢复链本身仍有两个执行空洞：production wrapper 的动态模块名不能被 `multiprocessing spawn` 可靠导入；两个 runner 又只验证 review/receipt 的封条，不验证封条里面到底是 PASS 还是 NO-GO。

## 三个 P0

1. wrapper 第 355–360 行把 legacy 源码加载成不存在的合成模块 `m528_frozen_r1r2_for_r3`，且没有注册进 `sys.modules`。legacy 第 445–452 行却把该模块的 `worker_init/worker_phase` 交给 spawn pool。函数保留这个不可导入的 `__module__`，production 会在 pool 建立或首次提交时因 pickle/import 失败。schema smoke 在 `load_legacy` 之前返回，所以它无法发现这个 production-only 缺陷。
2. smoke runner 只校验 author/static review 的 outer seal 和目录内封条，没有解析 static review 的 PASS/status/verdict、P0/P1、授权边界或 reviewed source tuple。双封 NO-GO review 也能被新 admission 绑定后启动 smoke。
3. production runner 同样不解析 static review、smoke receipt、smoke hammer 的语义。它不要求三 cases 成功、不要求 forbidden activity 为 false/0，也不要求 hammer 的 P0=P1=0。现场只重跑 positive smoke，不能替代两个 negative case 和独立锤审。

因此当前链上即使 root 尚未签 admission、尚未运行，也必须先换新身份修复，不能消耗现有 r3 的任何一次性 identity。

## 两个 P1

- 两个 runner 都不验证 admission 自身的 member/outer seal，也用允许 duplicate key 的 `jq` 直接消费 admission；这没有落实合同声称的“double-sealed admission”。
- production 只检查 r2 attempt 目录存在和 r2 canonical 不存在，没有现场复验 r2 attempt 的双封，也不检查精确 failure quarantine。当前两份 r2 证据封条仍通过，但 runner 没有保护其后续完整性。

## 保留下来的正确部分

schema positive path 的 slow pointer/corner/cell/shape/9 宏面积检查是严的；smoke 在加载 legacy、建 pool、重放 row、创建 analyzer output 之前返回；path-exact adapter 只给 mapping 的 deep copy 增加兼容字段；legacy cycle/traffic/baseline/decision 主体仍是冻结 SHA；row64/B8/128 B/cycle/CAM64 已在 delegate 前检查；production 的资源门和现场 positive smoke 的确位于 attempt sentinel 之前；r2 canonical 当前不存在，r2 attempt/quarantine 的双封当前均通过。

wrong-pointer/wrong-corner 两个负例实际上在命令行期望值 guard 处失败，并没有构造“live mapping 缺字段”的 malformed-schema 情形。它们可以叫 fail-closed argument controls；若论文/合同叫 malformed-schema controls，需另加隔离 fixture。

## 最小 r4

不要改 legacy 算法。用真实文件名对应的可导入 module name 正常 import legacy，先钉 `legacy.__file__` origin 和 `c611f8c...afb8a` SHA，再保留现有 slow-area adapter。修复后增加一条独立、非 production 的 spawn-import self-test：单 worker、单独 admission/attempt/receipt，不建 production result、不重放 production rows、不消耗 production attempt；用 exact `legacy.worker_init` 启动 spawn worker，再执行同一 legacy module 的无 row 函数并返回固定 token，同时静态 serialize-check `worker_init/worker_phase`。通过独立 receipt hammer 后才可签 production admission。

runner 还必须 strict-parse 每个 review/receipt/hammer：检查 exact schema/PASS/status/verdict、P0=P1=0、reviewed source SHA、receipt 三 cases、forbidden activity、review-to-receipt binding；同时验证 admission 自身双封和 r2 attempt/quarantine 双封。

## Claim boundary

本红队只授权 author 起草最小 r4 源码修复及新的 source-only review。当前不授权 smoke、CPU production、RTL、VCS、Synopsys PPA、energy、full-network/system speedup 或 DATE headline。
