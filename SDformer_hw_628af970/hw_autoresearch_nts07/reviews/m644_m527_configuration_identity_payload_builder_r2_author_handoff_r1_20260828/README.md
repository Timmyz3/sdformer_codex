# M644｜M527 configuration-identity payload builder r2 作者交接

## 裁决

M644 以新文件 supersede M634 r1；没有修改 M634 作者 handoff、M638 fresh review、M527 r3 或 `docs/359`。本次只关闭 builder/verifier 的 fail-closed 缺口，**没有生成 production payload**。

当前允许的唯一正面陈述是：在真实上游 semantic receipt 和全部 live sources 到位后，M644 可生成并验证五档配置身份 payload。即使 PASS，也不得写成 M527 registry gate 已开启，更不得写 waterfall、system speedup、effective GOP/s 或 paper headline。

## M638 finding 对应修复

- P0：验证器从调用方再次提供的 authoritative live sources 重建完整 expected common/config/registry/receipt，四类文档必须完整结构相等；已知字段重封不能绕过。
- P1-01：每档 optimized/fallback partition、mechanism、resource/charge/fallback/claim，以及 configuration/simulator/trace/measurement/common 的每个 path/SHA 都回绑 live sources。
- P1-02：measurement v2 强制绑定 `m644_h67_decoder_complete_semantic_verification_receipt_v1`；现场核对 schema、PASS status、population、checkpoint、三个 manifest SHA、frame/density、完整 operator universe、六个 semantic proof 和 non-admission boundary。
- P1-03：staging 在 rename 前跑完整 live-source verifier；rename 后验证若失败，canonical output 被原子移入显式 quarantine，并写 `POST_PUBLISH_FAILURE.json`。
- P2：build/validate output 必须在 repo 内，逐层拒绝 symlink ancestor，并拒绝 dangling leaf symlink。

## 文件身份

- builder：`system_simulator/scripts/build_verify_m644_m527_configuration_payload_r2.py`
  - SHA256 `435baacb13da5da1c30ca649353b0947476e4bec7a4164d4421c3cdd615abea7`
- tests：`system_simulator/tests/test_m644_m527_configuration_payload_r2.py`
  - SHA256 `5be169c9b50c3b19a3e5240df3126d21d748fa82cc5d0846fa623e608edb0b23`
- contract：`contracts/m644_m527_configuration_identity_payload_builder_r2_contract_20260828.json`
  - SHA256 `82a4b62a3a3b256328a010189a2fa71fcd46225ac1c84aa373780358d6d621c5`
- frozen M634 base：SHA256 `b53429d9444e44f33cb9a240f696a3d847323da1af7929ed43e473e87fa564fa`
- frozen M527 r3：SHA256 `83ea25e43b53d12800ac64e971069a682e3077411ff10851a7861636ef77355b`
- `docs/359`：SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

## 定向验证

```bash
python3 -m py_compile \
  hw_autoresearch_nts07/system_simulator/scripts/build_verify_m644_m527_configuration_payload_r2.py \
  hw_autoresearch_nts07/system_simulator/tests/test_m644_m527_configuration_payload_r2.py

python3 -m unittest -v \
  hw_autoresearch_nts07/system_simulator/tests/test_m644_m527_configuration_payload_r2.py
```

结果：Python 3.6 compile PASS；12/12 tests PASS。A1--A9 全覆盖；另覆盖 repo 外 output 与 dangling leaf symlink。

## 尚缺的 production 输入（精确列表）

M644 当前没有、也没有伪造以下输入：

1. superseding M624 analyzer 产出的真实 decoder-complete semantic receipt，schema/status 必须精确等于 M644 contract；
2. 与该 receipt 一致的 decoder-complete trace manifest；
3. sequence-population manifest；
4. aggregation-weight manifest；
5. measurement-identity v2（含 checkpoint、population、frame/density bins、完整 operator universe 和上述 receipt path/SHA）；
6. safe unified simulator source；
7. common-resource source；
8. 五个 production configuration sources：B0、B1、B2、B3、Ours。

以上八类输入任一缺失、schema/status/identity 不一致或 SHA 漂移，build 必须 fail-close。production payload 只能在 fresh independent hammer 给出 P0=0/P1=0/GO 后生成。

## 边界

- production payload：未生成
- GPU/VCS/DC/PT/Formality/remote：未运行
- M511：未运行/未修改
- `docs/359`：未修改
- M527 gate/waterfall/system speedup/effective GOP/s/headline：全部 false

