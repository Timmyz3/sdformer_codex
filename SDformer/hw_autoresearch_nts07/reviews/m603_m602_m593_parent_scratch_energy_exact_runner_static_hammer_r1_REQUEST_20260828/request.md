# M603｜M602/M593 energy exact-runner fresh static hammer request

请由**未参与 M602 作者工作**的 fresh agent 只读审查：

- `system_simulator/scripts/run_m602_m593_parent_scratch_generated_macro_energy_r2_exact_sha.sh`，SHA `6a54d938...`；
- `contracts/m602_m593_parent_scratch_generated_macro_energy_exact_runner_launch_candidate_r1_20260828.json` 及双层 sidecar；
- `reviews/m602_m593_parent_scratch_energy_exact_runner_source_author_handoff_r1_20260828/`，manifest SHA `735d2491...`、outer-file SHA `b03dd544...`；
- M597 analyzer/contract/handoff、M599 PASS review 和所有 frozen business inputs。

禁止正式 `--execute`、正式 analyzer、result/attempt/launch、EDA/GPU/remote，禁止修改被审文件。允许 `bash -n`、`--preflight-only`、只在临时目录运行的 extracted/fault tests 与 strict JSON/seal 复核。

## P0/P1 必查

1. runner 必须在 attempt 前冻结并重验 M597 analyzer/contract/handoff、M599 review、M504/M528/macro-map/M595/docs359 的 exact path/SHA/manifest/outer，且终端与 pre/post publish 再验。
2. result/attempt/consumed/staging/quarantine 必须同 parent，所有坐标用 `lexists/lstat` 拒绝 symlink/dangling/既有 entry；trap 必须在 attempt mutation 前；所有 publish/consume/quarantine 使用真实 `renameat2(RENAME_NOREPLACE)`。
3. failure 必须把 attempt、runner staging、analyzer internal staging 收进同一 unique quarantine，并递归 member seal + outer seal；不得遗留可冒充 canonical PASS 的 staging。
4. success 必须验证 analyzer 原始 exact member set/manifest/outer、strict result schema/status/two rows/conservation/claim，生成 terminal receipt 后重封 exact member set；pre-publish 和 post-publish 均直接重哈 canonical。
5. runner `--execute` 只能接受固定 M604 path 的双封 true admission；该 admission 必须绑定 live runner SHA 和 fresh M603 review，且 M603 `P0=P1=0`/明确授权。当前 M604 不存在，source candidate 必须保持 `launch_now=false, release=false`。
6. 对 signal、marker/production/terminal/seal/publish/post-publish/attempt-seal/consume 失败逐段做静态或临时 fault audit，检查 canonical result 与 quarantine 的状态不可能形成 false PASS。

只有 M603 `P0=P1=0` 才允许 root **另立** M604 true launch admission；本 M603 不得自行创建 admission 或运行正式 analyzer。正式 M597 result 产生后仍需 fresh result hammer。`38.2283079189%`/`1.2622562287 mJ` 在此之前仍仅是诊断。

`docs/359` 必须保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
