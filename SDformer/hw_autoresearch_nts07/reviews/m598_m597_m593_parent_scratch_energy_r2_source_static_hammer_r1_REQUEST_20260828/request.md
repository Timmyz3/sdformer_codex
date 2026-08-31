# M598｜M597/M593 parent-scratch energy r2 source static hammer request

请由**未参与 r2 作者工作的 fresh agent**严格只读审查：

- `contracts/m597_m593_m528_parent_scratch_generated_macro_energy_source_contract_r2_20260828.json`
- `system_simulator/scripts/analyze_m597_m593_m528_parent_scratch_generated_macro_energy_r2.py`
- `reviews/m597_m593_parent_scratch_energy_r2_source_author_handoff_20260828/`
- M595 failed review 及 r2 所绑定的 M504/M528/macro-map 证据。

禁止运行正式 analyzer，禁止创建 canonical result/attempt/runner/launch，禁止 EDA、GPU、远程或修改被审文件。允许 Python 3.6 native compile、只访问临时目录的 built-in `--self-test`、clean-room fault injection 与独立公式重算。

## 必查 P0

1. all-write 必须直接来自 sealed M504 result/hammer：`456,016,645 cycles`、`16,490,761 macro reads`、`1,714,628 RAW forwards`、`27,305,568 macro writes`。forward 不得收 macro-read 能量；不得使用 M473 `18,205,389` parent edges 作为宏读。
2. dead-only 必须来自 sealed M528：`435,293,339 cycles`、`16,490,761 reads`、`1,714,628 forwards`、`9,947,701 writes`、`17,357,867 elisions`。
3. 独立检查 read+forward、write+elision、8-bank×144-B、M504/M528 cycle 交叉守恒，且 combined-PVRF row 不可能混入。
4. 独立复算 datasheet 单位与修正诊断。`38.2283079189%` / `1.2622562287 mJ per frozen sampled inference` 只是 review diagnostic，当前不得准入。

## 必查 P1/P2

- CLI 不得接受任何业务输入 path/expected SHA；contract/path/SHA/key-set 与所有 frozen input SHA/manifest/outer 必须在 analyzer 内冻结并先验验证。
- 验证 analyzer→contract SHA、handoff→analyzer SHA 的反向绑定没有循环替换口。
- future rows 必须保留 traffic/cycle source、read/forward/write、bank/word multiplier 与 conservation。
- 单位必须是 `per-frozen-sampled-inference`，不得写成 per-frame；claim 必须局限九宏 parent-scratch datasheet component model。
- 检查 staging→atomic rename、重复路径、symlink/path traversal、strict JSON 和 failure-close。

输出新的 `reviews/m598_m597_m593_parent_scratch_energy_r2_source_static_hammer_r1_20260828/`，包含 JSON/Markdown/mechanical receipt 和双层 seal。只有 `P0=P1=0` 才能允许 root **另立** exact runner 链；本 M598 仍不得授权正式运行。

`docs/359` 必须保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
