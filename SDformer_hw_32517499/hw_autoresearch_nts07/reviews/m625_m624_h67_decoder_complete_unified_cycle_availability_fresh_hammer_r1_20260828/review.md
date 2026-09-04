# M625 fresh independent hammer：M624 decoder-complete availability

## Verdict

**NEEDS REVISION，93/100，P0=0 / P1=1 / P2=1。**

M624 的当前截点结论是可靠的：输入与可执行语义不完整，因此必须 fail-close；五个配置行的 cycles、traffic、stall、fixed numerator 与 speedup 全部保持 `null`，且没有执行 production CPU simulator、GPU、EDA 或 remote 工作。该 null 裁决可保留。

但 R1--R6 目前不是严格充分的恢复清单。M624 自己报告 `m527_configuration_registry_ready=false`，而六项中没有一项明确要求生成、SHA 绑定并封存 M527 common-resource manifest 与 B0/B1/B2/B3/Ours executable configuration manifests。只完成当前六项文字动作，仍可能在同一 gate 上 fail-close。

## 独立重算

- 身份：M624 contract、analyzer、result、markdown、receipt、结果 manifest/outer seal、REQUEST manifest/outer seal 和 20 个绑定输入均逐一重哈希通过；analyzer `py_compile` 通过；`docs/359` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- ordered trace：10 samples、1,840 rows；每样本固定 184 rows = 79 operator + 93 ATLIF + 12 attention。全量为 790 operator、930 ATLIF、120 attention；operator 为 160 Conv2d + 630 Linear、79 个唯一 module，每个 module 恰出现 10 次；ConvTranspose2d 恰为 0。dual-line trace 为 3,580 rows，每样本 358。
- M51：manifest 310 records，31 modules，每样本 31、每 module 10；物理存在 160 个 bitpack、缺 150 个。对全部 160 个现存 payload 独立重算 size 与 SHA，mismatch=0。present/missing bytes 为 748,800,000 / 564,480,000；缺失 operator 为 Linear 140、Conv2d 10；缺失 module 15。
- M511：4 个 module 每调用元素数为 4,608,000 / 9,240,000 / 18,528,000 / 37,248,000；乘 10 samples 得 40 records、696,240,000 bits、87,030,000 packed bytes，与 contract 完全一致。
- M578：四个 `COUT_CIN_KY_KX` INT8 tensor 独立乘积为 5,308,416 / 1,330,560 / 333,504 / 167,616 bytes，合计 7,140,096 bytes。
- M590：M596 明确为 56 分、P0=3/P1=2，`formal_cpu_execution_allowed=false`；当前 M590 r6 结果目录不存在。
- 边界：M510 仅 analytic projection；M522 仅 decoder mapper logic support；M523 仅 bundler functional support。M216/M518/M519/M528 均是不同 scope 的 component evidence，不可相加或相乘。
- 截点后观察：M511 input、M511 verifier、M578 weight、M590 result 四个 optional runtime 目录在本次 fresh review 时仍全部不存在；无 post-M624 artifact 可记录。

## Findings

### M625-P1-01：R1--R6 缺少独立 configuration-registry 恢复动作

证据：M624 contract 把 “M527 complete configuration registry and common resource manifest” 列为 required ready gate；result 也报告 `m527_configuration_registry_ready=false`。R5 只要求 operator scope 与 fixed numerator，R6 只要求修复/替换 unified CPU source 并放入一个共同资源 schema，没有要求产生 M527 schema 的 common-resource manifest、五档 executable configuration manifests、各自 SHA 与 charge/fallback policy。

影响：当前 null 裁决不受影响，但六项清单不是“完成即可运行”的充分条件；若照文字执行，下一次 availability audit 仍可能在 registry gate 上停止。

最小修复：不新增 R7。扩写 R6，明确要求在 fresh static hammer 前生成并双封 M527 common-resource manifest 与 B0/B1/B2/B3/Ours 五档 executable configuration manifests，逐字段绑定相同资源 tuple、charge policy、fallback policy 和 simulator SHA；然后再把 `m527_configuration_registry_ready` 置为经验证的 true。

### M625-P2-01：decoder verification gate 目前只检查目录存在

证据：analyzer 的 `decoder_ready` 仅对 M511 inputs、M511 verification、M578 weights 三个路径做 `Path.is_dir()`；没有解析 manifest/schema、重哈希 seal、检查 40 records/87,030,000 bytes 或四权重/7,140,096 bytes。

影响：当前三个目录都不存在，因此 false/null 裁决正确；若未来目录出现但内容残缺，该字段名 `decoder_inputs_weights_verified` 会过度陈述。由于其余多个 gate 当前仍为 false，本次结果没有被错误晋级。

最小修复：下一个 superseding availability analyzer 必须验证三包的 schema、member manifest、outer seal、精确 population、checkpoint/config identity；不得把 `is_dir()` 当验证。

## 五档配置攻击结论

| 配置 | 当前证据 | 独立裁决 |
|---|---|---|
| B0 Dense96 Fixed-T10 | partial ordered trace + M22/M23 inventory + M518 directed VCS | blocked；无 decoder、完整 numerator、共同 completion schedule |
| B1 PTB-like K1x8 | M527 定义 | blocked；无 executable manifest、full-pop scan/fallback ledger |
| B2 exact K1 | M216/M519 component + M51 manifest | blocked；150 payload 与 decoder/共同 schedule 缺失 |
| B3 exact K1x8 | M519 directed component + M51 manifest | blocked；150 payload 与 replicated-resource/common schedule 缺失 |
| Ours C1+C2+C3 | M528 + M216/M522/M523 + M518 disjoint evidence | blocked；不可拼接，M590 禁止，decoder 与非重叠 memory schedule 缺失 |

## R1--R6 充分性裁决

- R1、R2、R3、R4、R5 均必要，人口与 byte 算术正确。
- R6 的 safe-source/fresh-hammer 要求必要，但需按 P1-01 扩写后才足够。
- 修复后仍只授权接收/生成缺失包、建立新 source contract 与 fresh static hammer；不得直接运行 simulator，不得产生 headline，也不得将 M510 projection 或 component 结果拼成 unified row。

## Admission

- `M624_FAIL_CLOSED_NULL_RESULT_VALID=true`
- `M624_MINIMUM_REQUEST_SUFFICIENT=false`
- `M624_PASS_GATE=false`
- 允许下一步：修订/取代 M624 的 R6 与 decoder verification predicates，再发起 fresh static hammer。
- 禁止：production simulator、headline、component addition/multiplication，以及任何对 `docs/359` 的修改。

