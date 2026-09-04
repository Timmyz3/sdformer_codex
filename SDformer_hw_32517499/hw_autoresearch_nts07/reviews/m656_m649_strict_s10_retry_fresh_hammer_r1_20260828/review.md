# M656｜M649 strict-S10 retry fresh independent hammer

## 裁决

`GO_EXACT_SINGLE_RETRY_NUMERIC_DIAGNOSTIC_GPU_COMMAND`，98/100；P0=0、P1=0、P2=3。

本裁决只放行 launcher `d9e362ee...`、contract `580ddee0...`、tests
`6808105e...` 在 repo root 执行合同内唯一命令一次。M653 r2 对旧身份的首次授权已经由
`staging.un14w70w` 消费，不得沿用；本 M656 授权无论成功或失败也立即消费。

## 严格 S10 修复复验

- `take_exact(iterable, 10)` 只在 `range(10)` 内显式调用底层 `next()`。第十项 loop body
  返回后，外层 for 为判断生成器结束只恢复生成器，不会再进入循环体、不会调用第十一次
  `next()`。
- 15/15 CPU 单测通过；带“第 11 次 `next()` 立即爆炸”的 iterator 调用数严格为 10。
- 另用真实 `torch.utils.data.DataLoader`、`num_workers=0` 和第 11 个 `__getitem__`
  立即爆炸的数据集独立重放：底层访问严格为索引 `[0,1,...,9]`，第 11 项未访问。
- 短于 10 项的 iterator 在第 10 次请求处显式 fail closed，不会静默发布不足样本结果。

## 首次失败与身份链复验

- 首次失败目录
  `results/m649_h67_ep35_convtranspose_typed_numeric_audit_s10_r1_20260828.staging.un14w70w`
  只有 `FAILED.json`，544 bytes，SHA256 `289063528565...`。
- strict JSON 全字段相等门确认：`completed_records=40`、原 M511 staging 保留、失败原因固定为
  `zurich_city_09_a_0101.npy` 缺失、状态为 `FAIL_CLOSED_NO_RESULT`；不是可引用结果。
- M649 canonical 与 M511 canonical 均不存在。首次失败目录在新运行开始和发布前都重新核验；
  任意 population、字节数、SHA、字段或 canonical 漂移都会拒绝执行/发布。
- 24/24 M649 输入与 21/21 frozen M511 输入当前均为 regular file、无 symlink，SHA
  全匹配。M511 consumed attempt 双 seal、M511 失败 staging 两成员、checkpoint/config/model/
  source code、`docs/359` 均重放一致。
- `docs/359` 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 数值、路径与事务边界

- d0 只准 exact `{0,1}`；d1--d3 只准 source-ordered first-two finite analog flow 与
  `[2,C)` exact-binary suffix。last-two 仅诊断，不参与准入。
- wrong dtype、nonfinite flow、nonbinary suffix、错误 population 均进入 typed-split NO-GO；
  不做 threshold/coercion，不生成 activation payload。
- runtime contract/config/checkpoint/output、24 个 M649 输入、失败 M511/M649 状态路径均在
  canonical equality 之前拒绝 symlink、dangling symlink 与 `..`。strict JSON 拒绝 duplicate
  key 和 NaN/Inf token。
- 完整结果只通过新 staging、double seal、atomic rename 发布；post-publish 验证失败会隔离
  canonical。结果成功仍只授权后续新 mixed-type capture/simulator 合同，不授权 cycle、
  speedup、RTL、VCS/DC/PT、energy/PPA、system 或 DATE headline。

## 唯一单次重试命令

工作目录必须是 `/home/zhumd/work/sdformer_codex/SDformer`。执行前必须再次确认 launcher/
contract/tests/author-handoff outer-file、Python、checkpoint、`docs/359` 的 SHA，24/24 与
21/21 输入仍是 regular file，首次 M649 失败 staging 未漂移，M511/M649 canonical 均缺失，
且本机 GPU 空闲。唯一授权命令为：

```bash
/opt/anaconda3/envs/pytorch310/bin/python3.10 neuron_experiments/H9_bipolar_self_attention/entrypoints/audit_m649_h67_convtranspose_typed_numeric_inputs.py --contract hw_autoresearch_nts07/contracts/m649_h67_ep35_convtranspose_typed_numeric_audit_contract_r1_20260828.json --m511-contract hw_autoresearch_nts07/contracts/m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json --config neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml --checkpoint hw_autoresearch_nts07/system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth --output-dir hw_autoresearch_nts07/results/m649_h67_ep35_convtranspose_typed_numeric_audit_s10_r1_20260828 --samples 10 --num-workers 0 --channel-block 64
```

该命令只允许执行一次；不以 canonical 是否生成作为授权是否消费的判据。命令返回后必须
fresh post-result hammer；失败时若要再试，也必须新 contract/新身份/新 static hammer。

## P2 边界

1. frozen M511 的旧 `verify_inputs()` 仍是 resolve-first；本轮独立外层复验确认当前 21/21
   均无 symlink，所以不阻塞 exact cut，未来路径/身份变化必须重审。
2. launcher 不在代码内持久化本次 M656 review 的“授权已消费”标记；单次性由本双封 review
   的执行纪律约束。任何返回后的再次执行均未授权。
3. `typed_split_decision()` helper 本身不证明 10x4 lattice 唯一有序；生产 hook state machine
   与 processed/order 门已约束生成路径，fresh post-result verifier 仍必须逐项复核 40 格。

本 hammer 没有运行 GPU/model forward、M511、EDA 或远端任务，没有创建 M649 canonical 或
新的 staging/quarantine，也没有修改 `docs/359`。
