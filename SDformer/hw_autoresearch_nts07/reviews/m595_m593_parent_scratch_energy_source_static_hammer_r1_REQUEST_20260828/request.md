# M595｜M593 parent-scratch generated-macro energy source static hammer request

独立只读评审以下 source-only 包：

- `contracts/m593_m528_parent_scratch_generated_macro_energy_source_contract_r1_20260828.json`
- `system_simulator/scripts/analyze_m593_m528_parent_scratch_generated_macro_energy_r1.py`

禁止运行正式 analyzer，禁止创建 canonical result/attempt/launch admission，禁止 EDA、GPU、远程和修改被审文件。允许在临时目录执行小型 clean-room self-test、独立复算公式与 fault injection。

必须重点检查：

1. Python 3.6 兼容、strict JSON duplicate/nonfinite、输入 SHA fail-closed；
2. M528 result hammer 的 exact CPU admission 与 `logical bytes != energy` 边界；
3. 生成宏 `128x128b 1RW SP` 的 13/13 view、slow 0.9 V datasheet 数字、九宏面积与 M528 容量账一致；
4. all-write 的 traffic 与 M504 1RW cycle、dead-write traffic 与 M505 cycle 配对唯一，144 B 对齐；
5. `uA/MHz * V = pJ/access`、dynamic/leakage/frame 单位换算与 40.56342165646709% 独立重算；
6. result schema 必须把数字限制为 parent-scratch generated-macro datasheet component model，明确不是 integrated PPA、C1/全网能量、silicon、system speedup 或 DATE headline；
7. 检查是否存在会把临时 self-test、旧结果或 combined-PVRF traffic 误收为正式 dead-write-only 结果的路径。

输出新的 M595 review 目录、JSON+Markdown 评分、P0/P1/P2 与双层 seal。只有 P0=P1=0 时才允许后续另立 exact runner/launch admission；本评审本身不得运行正式结果。`docs/359` SHA 必须保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
