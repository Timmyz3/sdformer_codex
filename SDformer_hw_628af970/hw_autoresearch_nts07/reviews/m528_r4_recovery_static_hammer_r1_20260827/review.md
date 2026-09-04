# M528 r4 recovery source-only static hammer

Date: 2026-08-27  
Reviewer role: independent source-only hammer; not the author or launch authority  
Score: **98/100**; P0/P1/P2 = **0/0/2**

## Verdict

**PASS source-only。Root 最多可再创建一份独立双封的 non-production
preflight-only admission；本评审不直接授权运行 preflight，更不授权 CPU
production、EDA、GPU 或 RTL。** 必须先对 admission 做独立只读锤审，之后才可
消费唯一一次 spawn + 三 schema case preflight suite。

本评审没有执行 analyzer、schema smoke、spawn self-test、preflight runner 或
production runner；没有启动 CPU production、EDA、GPU 或 RTL，也没有修改被审
source、author handoff 或 `docs/359`。

## 身份与封印

Author handoff 与 review request 的 member/outer 两层封印均通过。五份 live r4
source SHA 分别为：

- analyzer `c94b2ca...3d4f`
- preflight runner `893a89c...a0944`
- production runner `a29827cd...07ac`
- execution contract `c02faf52...390`
- strict-JSON tool `b2e95ec...cf6`

冻结 legacy analyzer 仍为 `c611f8c...afb8a`。两个 runner 都钉死
`/opt/anaconda3/envs/pytorch310/bin/python`，实际 SHA 为
`9f78cd42...2115`、版本 3.10.18。系统 `/usr/bin/python3` 是 3.6.8，SHA
`9c9502e2...7f`，没有被任何 r4 runner 使用。

`docs/359` 仍为 `dedde7ce...dfc4`，未修改。

## Spawn/import 红队

r4 通过脚本目录上的真实模块名
`analyze_m528_h67_single_port_same_ledger_recompute` 做 normal import；Python 的
normal import 把该名字注册进 `sys.modules`。wrapper 在使用前又检查
`LEGACY.__name__`、解析后的 `LEGACY.__file__` 和 legacy SHA。因此 r3 的虚拟
`spec_from_file_location` 名以及 spawn 无法重新导入的问题已经关闭。

未来正例 preflight 会同时做 `worker_init` 与 `worker_phase` 的 pickle identity
round-trip，然后用 `spawn` context 创建 **1 个 worker**。initializer 是 exact
`legacy.worker_init(rows_path)`；调用链最终仅在 M504 中执行
`os.open(rows_path, O_RDONLY)`，不执行 `os.pread`。提交函数是同一真实模块的
`legacy.sha256_file(docs/359)`。只有该 future 返回冻结 SHA 后父进程才打印一次
PASS token。

因此静态控制流与未来 receipt 的语义一致：`worker_init_called=true`，但
`worker_phase_called=false`、`row_ledger_semantic_read=false`、`row_replay=false`。
两个 argument negative controls 在 `validate_schema` 的 pointer/corner guard
处、建 pool 之前失败，且 token 数必须为零。它们被准确命名为 argument
controls，不冒充 malformed-live-schema 测试。

## 两级 runner 证据闭合

Preflight runner 在建 attempt 前执行以下 fail-closed 检查：caller 钉 runner 与
admission SHA；admission 用严格 JSON 工具拒绝 duplicate key，并检查 member/outer
sidecar；author/static review 双封；static review 的 exact
schema/status/verdict、P0=P1=0、`withdrawn=false`、五份 source identity、author
outer seal 和仅 preflight 授权；r3 red-team/withdrawal 的 NO-LAUNCH；r2 failure
review 的 NO-GO、attempt/quarantine 两层封和 canonical absence。r4 preflight 与
production canonical/attempt 当前全部不存在。

Production runner 不仅现场重复正例 spawn/schema，还在 production attempt 前
直接验证：

1. 新 production admission 自身 strict JSON 与双封；
2. static review exact PASS、身份与 `root_may_create_one_production_admission=false`；
3. preflight admission 的 SHA、双封和 preflight-only 语义；
4. receipt 的 SHA/outer seal、正例 1 token、两个负例非零退出且 0 token、
   `worker_init=true`、`worker_phase=false` 与所有 forbidden false/zero；
5. 独立 receipt hammer 的 exact PASS、P0=P1=0、reviewed receipt/outer identity，
   以及仅允许 root 再签一份 production admission、直接 production 授权仍为 0；
6. r3 NO-LAUNCH 与 r2 consumed failure 边界；
7. 三次 48 GiB commit / 128 GiB available / 32 GiB swap / clean cgroup resource
   snapshot 和 UID-local Synopsys/VCS collision gate。

所以 sealed NO-GO、伪造的三例字段或不匹配 hammer 不能仅凭 outer seal 洗成
production authority。

## 冻结计算边界

唯一 schema 修复仍是从 exact mapping 的
`generated_view_inventory.slow.area_um2` 读取 `8758.3606 um²`，经已核验的
slow corner/cell/shape 后，在 mapping deep copy 上填充 legacy 兼容字段
`generated_view_inventory.area_um2`。九宏面积为 `78825.2454 um²`。

row64、B8、128 B/cycle、CAM64、3 workers、chunksize 2 均未变。worker、cycle、
traffic、capacity、aggregation 与 decision 仍全部委托给 byte-frozen
`c611f8c...afb8a`。没有发现算法、阈值、baseline 或 denominator 更换。

## 两个非阻断 P2

1. 通用 admission-sidecar helper 会对两个预期 sidecar 文件执行
   `sha256sum -c`，但没有再显式解析 sidecar 内记录的 target basename。调用方
   另行钉死 admission JSON SHA，因此不构成当前绕过；未来 root admission 与其
   独立锤审必须要求 canonical self-naming sidecar。
2. Production runner 对 byte-pinned r3 red-team/r2 failure review 的直接字段谓词
   比 preflight runner 略短。由于 exact review JSON SHA 和两层封都被固定，且
   已检查 NO-LAUNCH/NO-RERUN 的关键状态与授权，这不改变判决；后续可把两边
   合并成同一 exhaustive helper。

## 精确授权边界

未来 admission 必须满足：

- schema `m528_r4_preflight_static_admission_v1`
- status `AUTHORIZED_ONE_M528_R4_NONPRODUCTION_SPAWN_AND_SCHEMA_SUITE`
- `preflight_suites=1`
- `cpu_production_runs=0`，EDA/GPU=0，RTL=false
- pin 五份 live source SHA、author outer seal、本 review JSON SHA 与 outer seal
- admission 使用 canonical self-naming member/outer sidecar，并先过独立审阅

除此之外一律 NO-GO。当前没有 r4 preflight receipt、cycle、speedup、RTL、PPA、
energy、full-network/system speedup 或 DATE headline。
