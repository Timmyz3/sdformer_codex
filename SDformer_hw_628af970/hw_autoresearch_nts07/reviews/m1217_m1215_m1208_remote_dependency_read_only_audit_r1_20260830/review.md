# M1217 M1215→M1208 远程依赖只读审计

## 裁决

`NO_GO_RETRY_M1215__AUTHOR_NEW_DISJOINT_SUCCESSOR`。

M1215 本地一次性 marker 已于 2026-08-30 16:11:50 CST 消费，不得重试。远程 M1208 attempt/result/log 仍全部缺席，证明 GPU capture 尚未开始，不是 checkpoint 或 GPU 故障。

## 唯一缺失集合

对 143 个去重后权威文件逐一比较 remote size/SHA/type：134 个 EXACT，0 个 drift，9 个 missing。九个 missing 全部来自同一个递归双封存目录：

`hw_autoresearch_nts07/reviews/m1215_m1210_m1208_first_launch_failure_forensic_r1_20260830/`

缺失成员为 `READ_ONLY...txt`、`RUN_COMPLETE.txt`、`independent_forensic.py`、`mechanical_checks.json`、`remote_read_only_observation.json`、`review.json`、`review.md`、`SHA256SUMS`、`SHA256SUMS.seal.sha256`。

## 已排除

- M1216、M1211、M1209 递归双封存均 EXACT。
- successor launcher、launch contract、capture source/contract/test、docs/359 均 EXACT。
- M1182 旧 inventory 95 行全部 EXACT。
- M1180 marker token EXACT，M1180 result/log ABSENT。
- M1208 attempt/result/log ABSENT，远程 GPU compute process ABSENT。
- docs/359 SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 根因与修复边界

M1215 secure wrapper 的 archive 成员是“1 个 corrected launcher + M1216 hammer”，但 remote successor preflight 在启动 M1208 前还必须递归验证 M1215 forensic。该 forensic 没有被纳入 M1215 transfer inventory，因此在未创建 M1208 marker 前 fail closed。

合法修复只能是新的 disjoint successor：精确传输 forensic 九个递归 seal 成员，在 publication 前重验全部 143 个依赖，保持 M1208 attempt/result/log 身份不变，由新的不同作者 hammer 授权且仍只允许一次远程 launch。

本审计不授权远程写入、M1215 重试、GPU capture 或 EDA。
