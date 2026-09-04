# M603 request｜M601/M579 r4 source + launch-now-false candidate fresh static hammer

请对 M601 的 r4 mechanical overlay 与 `launch_now=false` candidate 做一次**全新、独立、只读** static hammer。
不得运行正式 80-record CPU，不得创建 true execution contract/release/result/attempt，不得运行 GPU/EDA/远程，
不得修改 r4 source tuple、candidate、M598/M594 证据或 `docs/359`。

## 冻结评审对象

- `system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r4.py`
  - SHA256 `ba8fc0326b4d17f45d6db156d89b29da0894560d70d82f65ea5ae5f40b115195`
- `system_simulator/scripts/run_m601_m579_paft_control_single_port_product_capture_r4_exact_sha.sh`
  - SHA256 `8c0fcbea21eb99d2ff740d2c710e552ee7db3c5f773221cc5579899e58ad53fe`
- `contracts/m601_m579_paft_control_single_port_product_capture_source_contract_r4_20260828.json`
  - SHA256 `27e995145c91de62fe687cff7a5a34889047ca1c29fa29f517e27305101d0276`
- `contracts/m601_m579_paft_control_single_port_product_capture_execution_candidate_r4_20260828.json`
  - SHA256 `ff6aae0b782e08c48354c0f62739e553ed74991217c0cdbda1ad4929981d28c4`
- `reviews/m601_m579_paft_control_single_port_product_capture_r4_source_author_handoff_20260828/`
  - manifest SHA256 `ba5ce81748ac7b99f97bc5d6eb18cbed88384c5b1b9026efe668feb61da2cb87`
  - outer-seal file SHA256 `bf46f039a746c909a0bf88a652b088ae690439a1e62601ba0a3750251230b554`
- M598 PASS review：manifest SHA256 `187157eb64210203f5c3c050e90c035e9549246b932b54e9fa9309ba2a7bd8d8`。

## 必攻项

1. 实际运行 immutable runner `--preflight-only`，确认 frozen Python/NumPy、spawn、M43/M504/M505、八行
   recurrence、chunk-major anchor、15 keys、零正式 record、零 result/attempt。
2. 证明 r4 只委托 exact-SHA M594 r3，且 r3/r2/r1/M43/M504/M505 身份均锁；重点检查 frozen r2 与 r1
   的 `__file__`/analyzer identity 在 validator 和 production boundary 都正确指向 r4，不能因 schema bridge 假拒绝
   或绕过 top-level analyzer SHA。
3. 用临时 true v4 contract 只跑 `--validate-contract-only`，重验精确 15 inputs 和 80 payload，正式 record=0；
   攻击少/多 key、path/SHA、schema/auth、contract bytes、r4 analyzer/runner identity。
4. 攻击 M598-P2-01 全坐标：contract、result、attempt、consumed、PID staging、quarantine staging/final、terminal
   result/CSV。dangling symlink、live symlink、regular file 冒充 directory、directory 冒充 file 都必须在 attempt 前
   或发布前 fail-closed；bash 必须使用 `-e OR -L`，Python 必须使用 `os.path.lexists` 并显式拒绝 symlink。
5. 确认 failure cleanup 对 attempt/staging 使用同一 unique quarantine；failure receipt 分别记录 lexists、
   is_symlink、is_directory 与 start/current identities；封存前 tree 无 symlink；member/outer seal 与 final
   `RENAME_NOREPLACE` 保持。
6. 确认 success 仍是 terminal all-input/80-payload rehash、result member/outer seal、pre-publish identity、result
   `RENAME_NOREPLACE`、attempt completion seal 与 consume `RENAME_NOREPLACE`；不得因 overlay 引入覆盖式 `mv`。
7. 攻击 `launch_now=false` candidate：它必须是 production analyzer 不接受的 candidate schema，且
   `launch_now/run_cpu/max_attempts/release=false/false/0/false`；即使调用 runner `--execute` 也必须在 attempt 前
   拒绝。candidate 与 source exact 15-input mapping 必须逐字节一致。
8. 复核 chunk-major、M504/M505、DMA/tail/commit/8 blocks、M255 三个 accuracy scope、64 帧 PAFT 退化
   1.0189020311889285%、九行容量 213,376 B 均未改变；macro/PPA/energy open。
9. 确认 arithmetic-work/local-cycle/PAFT-control activity increment 不相乘，system/RTL/PPA/energy/headline
   全 false；r4 不是新性能结果。
10. 确认 true execution contract、release、formal result/attempt 全 absent，`docs/359` SHA 不变。

## 裁决门

- 给出 100 分制、P0/P1/P2 计数和 PASS/FAIL。
- PASS 必须 `score>=95、P0=0、P1=0`。
- 即使 PASS，也只说明 source + `launch_now=false` candidate 静态合格；root 仍须另建 exact-SHA true v4
  execution contract，经过独立 true-launch admission/release 后才可运行一次 80-record CPU。
- 输出 `review.md`、structured receipt、mechanical checks 和 member/outer 双封。

