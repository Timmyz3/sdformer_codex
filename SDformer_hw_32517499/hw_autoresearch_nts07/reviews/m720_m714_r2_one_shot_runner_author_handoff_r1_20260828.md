# M720｜M714-r2 PCTDA one-shot runner 作者交接

本轮只修复 M716 的 pre-run P0/P1，不运行 GPU，不创建 attempt/result，不准入周期或 RTL。

审阅对象：

- `system_simulator/scripts/trace_m714_h67_ep35_pctda_pattern_s10.py`
- `contracts/m714_h67_ep35_pctda_pattern_s10_contract_r2_20260828.json`
- `system_simulator/scripts/run_m714_h67_ep35_pctda_pattern_s10_r2_one_shot.sh`

当前 SHA256：

- capture: `f35fa0ec051f8e45c89a0ab0c0280695ae71b429fe3e8b1ac4806dcb8a276200`
- contract: `184458d69b542386c45b99af0ba8744dc8c6ad1a91e6c06b88812c906a4dd723`
- runner: `f5f4202d5e934f7dc44838052d493d43d671d1590daf51cad800d0747b30e857`

已做修复：

1. contract 固定 M714/M366/canonical M366 contract/M716/docs359 SHA；capture 严格拒绝 duplicate/non-standard JSON。
2. M714 PASS 前强制 `samples=10`、installed/live/T10/T2=`105/81/45/36`、T10 calls=450、dead-called empty、range/nonfinite/bound/integer mismatch 全零。
3. pattern counter 强制 tile/bitplane、histogram、distinct/nonzero、per-site=aggregate、chunk tile-boundary 守恒。
4. 周期统一改名 `ideal-resource lower bound`；M518 `17N+12` 不重复加五拍；build-from-weights 只加 64 build/call，direct-table-load 按 28 beat、相对 M518 额外 23 beat；45 配置 resident 宏容量/面积单列。
5. runner 在 attempt 前要求 independent runner SHA/outer-seal 环境绑定，验证 review JSON，再做四次连续 GPU/process idle check；不空闲不消费 attempt。
6. 唯一 attempt、同文件系统 staging、失败 quarantine、成功 atomic rename、manifest/outer seal/terminal reverify 完整。
7. 输出指针改为相对文件名，避免 staging rename 后绝对路径失效。

静态审阅若 PASS，必须创建恰好以下路径和机器身份，供 runner fail-closed 验证：

- directory: `reviews/m720_m714_r2_one_shot_runner_fresh_static_hammer_r1_20260828`
- `review.json` schema: `m720_m714_r2_one_shot_runner_fresh_static_hammer_v1`
- status: `PASS_M720_M714_R2_ONE_SHOT_RUNNER_STATIC_HAMMER`
- verdict/score/P0/P1/P2: `PASS/100/0/0/0`
- identity 必须含 exact `runner_sha256`、`contract_sha256`、`capture_sha256`
- decision 必须含 `exactly_one_remote_gpu_capture_authorized=true`、`four_fresh_idle_checks_required_at_launch=true`
- directory 必须有 `SHA256SUMS` 与 `SHA256SUMS.seal.sha256`

即使静态 PASS，也只有 runner 自己在四次新鲜 idle gate 后才能启动一次 A800 capture。任何 pattern 数字仍只准作为 opportunity/lower-bound；真实 DA output miter、可执行周期、RTL、PPA、能量、系统倍率、headline 全为 false。

`docs/359` 不得修改，冻结 SHA 为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
