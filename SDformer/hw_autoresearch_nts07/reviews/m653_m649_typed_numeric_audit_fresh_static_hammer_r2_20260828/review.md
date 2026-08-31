# M653｜M649 ConvTranspose typed numeric audit fresh static hammer r2

## 裁决

`GO_EXACT_ONE_SHOT_DIAGNOSTIC_GPU_COMMAND`，98/100；P0=0、P1=0、P2=3。

本裁决只放行 launcher `b5a4f71a...`、contract `651dddb3...`、tests `5b3c8bc3...` 在 repo root 执行合同中的唯一候选命令。M652 对旧 launcher `32e347ff...` 的 NO-GO 仍有效，旧身份不得运行。

## P1 修复复验

- launcher 现在用 `checked_path` / `checked_path_match` / `checked_contract_path`，在任何 `resolve()` 或 canonical equality 之前拒绝原始路径链中的 symlink、dangling symlink 和 `..` traversal。
- 重放 M652 的 dangling output leaf 与 runtime/contract input alias，均 fail closed。
- 新增重放 output parent alias、failed M511 staging alias、consumed attempt alias、forbidden canonical dangling alias，均 fail closed；正常缺失的 canonical leaf 仍通过。
- main 对 runtime launcher/contract/output、M511 contract、config、checkpoint 都走 raw-chain check；23 个 M649 输入与 failed M511 三条状态路径走 repository-relative checked path。
- 当前 23/23 M649 inputs 与 21/21 frozen M511 inputs 均是 regular file，SHA 全匹配。failed M511 staging 仍是原两个成员，M511/M649 canonical 均 absent。

## 数值与事务复验

- Python 3.10 CPU 定向测试 14/14；`/usr/bin/python3.6 -m py_compile` launcher/tests 2/2。
- 独立尾块向量确认 `(0,1,3,4)` 对 `T_B_C_H_W` 精确保留 C；first2/source-order 与 last2/diagnostic 的准入角色未混淆。
- suffix nonbinary、flow nonfinite、wrong dtype/population 均 NO-GO；strict JSON 拒绝 non-standard number 与 duplicate key。
- M650、M649 author handoff、M511 consumed attempt、旧 M652 review 的双 seal 均通过；`docs/359` 仍为 `dedde7ce...`。
- 没有运行 GPU/model forward、M511、EDA 或远端任务；没有生成 M649 canonical/staging/quarantine。

## 唯一授权命令

工作目录必须是 `/home/zhumd/work/sdformer_codex/SDformer`，且执行前必须再次确认 launcher/contract/tests 三个 SHA、author handoff outer-file SHA `4f4e2573...`、Python SHA `9f78cd42...`、checkpoint/docs359 SHA、23 inputs regular、M511/M649 canonical absent、GPU 空闲。唯一授权命令为：

```bash
/opt/anaconda3/envs/pytorch310/bin/python3.10 neuron_experiments/H9_bipolar_self_attention/entrypoints/audit_m649_h67_convtranspose_typed_numeric_inputs.py --contract hw_autoresearch_nts07/contracts/m649_h67_ep35_convtranspose_typed_numeric_audit_contract_r1_20260828.json --m511-contract hw_autoresearch_nts07/contracts/m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json --config neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml --checkpoint hw_autoresearch_nts07/system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth --output-dir hw_autoresearch_nts07/results/m649_h67_ep35_convtranspose_typed_numeric_audit_s10_r1_20260828 --samples 10 --num-workers 0 --channel-block 64
```

该授权是一次独立 numeric diagnostic，不是 M511 one-shot 的重置或复用。命令输出无论 typed split GO/NO-GO，都必须先做 fresh post-result hammer；不得直接据此生成 payload、cycle、speedup、RTL、EDA、PPA、system 或 DATE headline。

## P2 边界

1. frozen M511 的旧 `verify_inputs()` 对其 21 个 predecessor path 仍采用 resolve-first；本轮外部重验确认 21/21 当前均是 regular file，所以不阻塞这个 exact cut，但任何未来身份/路径变化必须重审。
2. 结果尚不记录 Python/package/CUDA/GPU 完整 runtime identity；post-result receipt 应从执行环境补齐。
3. `typed_split_decision()` helper 不独立验证 10x4 lattice 唯一有序；main hook state machine 已保证生成顺序，post-result verifier仍须重验 `(sample_id,module_index)` 全格。

