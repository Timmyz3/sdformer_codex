# M728｜M714-r2 repair2 one-shot runner fresh static hammer

## 裁决

**FAIL，90/100；P0/P1/P2 = 1/0/0。** M724 的两个 P0 和一个 P2 已经修复，但新发现 terminal exact-SHA TOCTOU；不得运行当前 runner，不得创建 attempt/result，不授权远程 GPU capture。

本轮没有运行 runner，没有 import/执行作者 capture，没有查询 GPU/`nvidia-smi`，没有调用 EDA 或远程，也没有修改作者文件。

## M724 修复回归

- **contract/capture 兼容 PASS**：两边 milestone 都为 `M714-r2`，identity 精确是同一五 key：`m714_script/m366_script/m366_contract/m716_prerun_review/protected_docs359`。
- **compute-app 失败门 PASS**：`GPU_APPS` 查询已删除 `|| true`；在 `set -Eeuo pipefail` 下查询失败会退出，不会伪装成空 app 列表。
- **claim 措辞 PASS**：docstring、contract、payload status/fields 和 terminal receipt 均只写 `ideal-resource lower bound`；executable cycle、miter、RTL/PPA/energy/system/headline 仍为 false。

## 新 P0｜长 capture 后没有重绑冻结 SHA

runner 在启动前会把 runner/contract/capture 与冻结 SHA 对比，这一段正确。但 capture 返回后，直到 `RUN_COMPLETE.json` 和 seal 之前，没有再比较：

- `M714_R2_EXPECTED_RUNNER_SHA256`；
- `EXPECTED_CONTRACT_SHA256`；
- `EXPECTED_CAPTURE_SHA256`。

terminal validator 只比较 payload 中的 SHA 和**终态当下磁盘文件** `sha(contract_path)/sha(capture_path)`。它虽然读入 `c=strict(contract_path)`，但没有将 payload capture SHA 与 `c.identity.m714_script.sha256` 对比。capture 本身也是在 M366 长任务返回后重新 hash 当下磁盘文件，没有 require 它们与启动时 observed/contract SHA 一致。

所以若长时间 GPU capture 期间发生意外或并发编辑，最终可能封出一个“当前文件 SHA 内部自洽”的 result，但这些 SHA 不是 M728 授权的冻结对象，也不一定对应内存中真正执行的代码/合同。这与 one-shot exact-SHA 封存目标直接冲突。

修复必须在 capture 完成后、写 terminal receipt/seal 之前，再次把三个当前 SHA 与同一冻结 expected SHA 对比；同时把 payload identity 与 contract 内 pin 的 capture SHA 和启动时 contract SHA 对比。任一 drift 都必须 quarantine staging。

## 其余 M716/M720 门

| 项 | 结果 |
|---|---|
| 进程正例 `profile100/valid825/validate/trainer/trainonly/evaluation/training` 及两个项目别名 | PASS |
| 10 个无关名字负例 | PASS |
| 四次 idle 在 attempt 前，三次 5 s 间隔 | PASS |
| M366 10/105/81/45/36/450 人口、dead-called empty、四项零数值门 | PASS |
| M366 合同 14 个嵌套输入文件 SHA | PASS |
| pattern 守恒、chunk tile boundary、relative pointer | PASS |
| deterministic randomized smoke，`exhaustive=false` | PASS |
| Fixed `17N+12`；build `+64/call`；direct `+23 beats/call` | PASS |
| resident-45 P1/P2/P4/P8 = 23/46/92/184 macros | PASS |
| staging/failure quarantine/atomic publish/member manifest/outer seal | PASS，但 terminal identity rebind 失败 |

## 独立算术

- signed INT8 256 个码独立重构：0 mismatch。
- subset range `[-640,635]`，signed 11-bit 足够；含 Q24 bias 保守绝对界 `8,715,008 < 2^24`，signed25 足够。
- table：`7040 bit = 880 B`；Fixed N1/N4=`29/80 cycles`；direct=`28 beats`，相对五 beat 多 23。
- resident-45：P1/P2/P4/P8 = `23/46/92/184 macros`，`46/92/184/368 KiB`。

## Claim boundary

本 FAIL 不授权 runner、remote/GPU capture、attempt/result、M714 pattern 数字、executable cycle、real-output miter、RTL/VCS、Synopsys PPA/energy、accuracy、system speedup 或 paper headline。`docs/359` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
