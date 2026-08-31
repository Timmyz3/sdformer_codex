# M501 Python runtime 修复独立打铁审查（r1，2026-08-27）

## 裁决

**GO_FOR_EXACT_OPPORTUNITY_AUDIT_ONLY（99/100）**。

本次修复只把 runner 从主机默认 `python3`（实际解析为 Python 3.6.8）固定到 `/opt/anaconda3/envs/pytorch310/bin/python`（Python 3.10.18），并在启动前检查解释器可执行。它修复了 `math.prod` 的真实运行时阻塞，没有改变 M501 analyzer、contract、冻结输入、输出路径、机会门或 claim boundary。仍不准据此启动 RTL 或宣称 cycle/system speedup。

## 身份与逐字差异

- 新 runner SHA256：`51b1011abd31fb31ba9049d06695ff46f1bd3a6c3369c5ba721f574b8368f02a`
- analyzer SHA256（不变）：`5bdfa6f6fa81510d11751d6867748515763d3d4b31927b8cfe03e03ee597b7e7`
- contract SHA256（不变）：`bbb7bce5015ab3a3a5772b86d594853da353380df8dcd85a295e480d422eb2d6`
- docs/359 SHA256（不变）：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

将当前 runner 恢复为 `python3 "${m501_analyzer}"`，并删除新增的 `m501_python=...` 与 `[[ -x ... ]]` 两行后，重建 SHA 精确等于旧审查记录的 `473f1c076891148bdde15d3c743d2b1156f8c8399fd05bd212cb15569c99cc3b`。因此可证明修改范围只有这三处 runtime 选择变更。

## 实测

- `bash -n`：PASS。
- 固定解释器：Python 3.10.18；`math.prod([2,3,4])`：PASS。
- 原 `python3`：`/usr/bin/python3 -> /usr/libexec/platform-python3.6`，Python 3.6.8；同一 `math.prod`：`AttributeError`，复现原阻塞。
- analyzer 在固定解释器下 `py_compile`、`--help` 和无副作用 `runpy` import smoke：PASS。
- runner 内五个 pinned input 的现场 SHA 全部匹配；默认结果目录不存在。
- no-overwrite 攻击：令 `M501_OUTPUT_DIR` 指向预先存在的空目录，runner 返回 1；目录 inode/size/mtime 不变、条目数仍为 0，分析器未启动。
- 未执行全量 M501，未启动 VCS/DC/PT/GPU，未修改生产文件或 docs/359。

## 非阻塞备注

runner 本身没有被写入其将来结果目录的 `SHA256SUMS`；本审查封存了 runner SHA，因此本轮执行准入不受影响。若后续统一证据规范要求自包含复跑包，可把 runner 本身加入外层 manifest，但不要在本次已经批准的修复上继续扰动。
