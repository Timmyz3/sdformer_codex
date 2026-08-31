# M594｜M579 r3 source author handoff

日期：2026-08-28  
状态：**AUTHOR_SOURCE_ONLY；请求 fresh source-static hammer；没有 execution candidate/release。**

## 交付物

- r3 analyzer：`system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r3.py`
  - SHA256：`c684ac4ddc4cbea46e1eca7b088c303d8b0cf3acf6284e2a98d66d6e83136fd2`
- immutable future runner：`system_simulator/scripts/run_m594_m579_paft_control_single_port_product_capture_r3_exact_sha.sh`
  - SHA256：`268b47295447d2a16bc0e438eec0f35639f51fa2050119ec80ed37a474687011`
- r3 source contract：`contracts/m594_m579_paft_control_single_port_product_capture_source_contract_r3_20260828.json`
  - SHA256：`aca41b746ed9982a66f365e9160ced7b112f01b9eac11dc57f9be1e82f61f50d`
  - member sidecar 与 outer sidecar 已双封。
- frozen r2 dependency：`system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r2.py`
  - SHA256：`70eb07465bb008569967f69ae0ea0d51057d64dd0d51669b604a8f1cd4d4b471`

## M592 的 2 P1 + 1 P2 修复

1. **同一 execution-contract bytes**：runner 在正式验证前保存 `CONTRACT_SHA_START`，并把它显式传给 analyzer。validator entry/exit、production result binding、terminal entry/exit 和 pre-rename 都要求当前 SHA 等于 start；result/terminal receipt 同时保存 start identity。合同中途换成另一份“也合法”的 bytes 会 fail-closed，不能再让旧 result 配新 terminal contract。
2. **trap-before-mutation + sealed quarantine**：EXIT/INT/TERM/HUP handlers 在 `ATTEMPT_DIR` mkdir/marker 之前安装。失败时 canonical attempt 与 staging 通过 `RENAME_NOREPLACE` 进入同一 quarantine container，写入 failure stage、signal、start/current contract/runner/analyzer identity，再生成 member manifest 与 outer seal，最终 no-replace 发布 quarantine。
3. **exact required input set**：analyzer 内冻结 15 个 key；13 个历史 evidence/dependency 逐项冻结 path+SHA，r3 analyzer/runner 逐项要求固定 path、contract SHA 与 live SHA 一致。`set(inputs)==REQUIRED_INPUT_KEYS`，不能以“all declared inputs”掩盖漏声明。terminal 直接重哈 runner，不依赖 future author 自觉。

## 冻结计算没有改变

- r3 只通过 exact SHA 导入 M586 r2；M43 unpack、M504 cleanroom subset、M505 `simulate_liveness_task(tile, False)`、七个 cost arrays 的 `(432,47).T.reshape(-1)`、DMA=160、tail=2、commit=96,000/sample、8 output blocks 均未改。
- task order 仍是 `[sample,operator,row-chunk,partition]`，anchor `[0,47,94,141]`，20,304 tasks/operator，末 chunk 56 rows。
- future execution 必须同时重验 PAFT/control 各 40 packed payload、10x4 cohort、三个 288,000-byte plane、M255 三种 accuracy scope 和 M528 九行容量账。
- valid825 单 seed +0.5730215% 与完整 64 帧 PAFT 退化 1.0189020% 必须同列；`accuracy_performance_pareto=false`。
- 213,376 B 只表示 macro-rounded capacity fits 240 KiB，macro integration/PPA/energy 仍未准入。

## 已执行的唯一测试

- Python strict compile：PASS。
- runner `bash -n`：PASS。
- final exact runner `--preflight-only`：PASS。
  - Python 3.10.16 / NumPy 2.0.1 精确路径和 SHA；
  - spawn child 成功 import M43/M504/M505；
  - 八 synthetic rows：6 ideal issue、8 liveness cycles；
  - required input keys=15；
  - formal trace records=0，result/attempt created=false。
- 作者早期 wrapper 有一次 spawn pickling 自测失败，已通过 r3 top-level spawn entrypoint 修复；失败发生在最终 SHA 冻结前，未处理正式 record，也未创建 result/attempt。

未运行 80-record 正式 CPU、GPU、EDA 或远程。future execution contract、formal result、formal attempt 均不存在。docs/359 SHA 仍为 `dedde7ce...bdfc4`。

## 评审门

fresh hammer 必须独立检查 source/runner/contract，并重点攻击 contract bytes 中途替换、runner bytes 终态、trap ordering、signal/marker failure、failure quarantine 双封、exact 15-key set 和 schema bridge。不得运行正式 80-record。只有 score>=95、P0=0、P1=0，root 才可另建 launch-now=false execution candidate；source review 不能直接授权 production。
