# M595 request｜M579 r3 fresh source-static hammer

请对 M594 的 M579 r3 source tuple 做一次**全新、只读、独立** static hammer。不得运行正式 80-record CPU replay，不得创建 execution contract/result/attempt，不得运行 GPU/EDA/远程，不得修改 r3 source tuple、M592/r2 双封或 docs/359。

## 冻结评审对象

- `system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r3.py`
  - SHA256 `c684ac4ddc4cbea46e1eca7b088c303d8b0cf3acf6284e2a98d66d6e83136fd2`
- `system_simulator/scripts/run_m594_m579_paft_control_single_port_product_capture_r3_exact_sha.sh`
  - SHA256 `268b47295447d2a16bc0e438eec0f35639f51fa2050119ec80ed37a474687011`
- `contracts/m594_m579_paft_control_single_port_product_capture_source_contract_r3_20260828.json`
  - SHA256 `aca41b746ed9982a66f365e9160ced7b112f01b9eac11dc57f9be1e82f61f50d`
- `reviews/m594_m579_paft_control_single_port_product_capture_r3_source_author_handoff_20260828/`

## 必攻项

1. 实际执行 exact runner `--preflight-only`，确认 Python/NumPy/spawn/M43/M504/M505/八行 recurrence、required-input count=15、零正式 record、零 result/attempt。
2. 独立证明 r3 只是 exact-SHA r2 identity wrapper：M43/M504/M505、r1 worker、r2 analyzer SHA 全锁；spawn entrypoint 只委托 r2；chunk-major、80 payload、DMA/tail/commit/8 blocks 未改变。
3. 攻击 exact input set：任意少 key、多 key、改历史 path/SHA、改 r3 analyzer/runner path/SHA 均应在 attempt 前失败；不能只满足“all declared inputs”。
4. 攻击 contract bytes 同一性：start SHA 必须在 validator entry/exit、production result binding、terminal entry/exit、pre-rename 全部比较；result/terminal receipt 必须携带同一 start SHA。重点检查中途替换为另一份也能通过语义的合同是否仍 fail-closed。
5. 攻击 runner self identity：start SHA 必须由 live runner 得到、与 contract top-level/input 一致，terminal 直接 rehash，pre-rename 再比较；不能依赖可选 input key。
6. 攻击 attempt 无窗口：EXIT/INT/TERM/HUP trap 必须在任何 attempt mkdir/marker 之前安装。检查 marker 写失败、production 失败、terminal 失败、seal 失败、rename race 和 signal 分支。
7. 攻击 failure quarantine：attempt 与 staging 必须进入同一个 unique container；failure receipt 包含阶段/signal/exit/start-current identities/result-exists；全部成员与 outer manifest 双封；quarantine final rename 必须 NOREPLACE。不得在 formal 坐标留下未隔离 partial。
8. 攻击 success 路径：same-filesystem staging、terminal all-input/80-payload rehash、result member/outer seal、final RENAME_NOREPLACE、attempt completion seal 与 consume NOREPLACE；不得覆盖 result/attempt/consumed。
9. 复核 M255 三种 accuracy scope、64 帧 PAFT 退化 1.0189020311889285%、single seed、无共同 evaluator SHA、`accuracy_performance_pareto=false`；复核容量九行 213,376 B 与 macro/PPA/energy open。
10. 确认 arithmetic-work/local-cycle/PAFT-control activity increment 不相乘，system/RTL/PPA/energy/headline 全 false；docs/359 SHA 不变。

## 裁决门

- 给出 100 分制、P0/P1/P2 计数和 PASS/FAIL。
- PASS 必须 score>=95、P0=0、P1=0。
- 即使 PASS，也只能允许 root **另建** launch_now=false execution candidate，不能由 source hammer 直接授权 release 或 CPU production。
- 输出 review.md、structured receipt、mechanical checks 与 member/outer 双封。
