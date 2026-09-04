# M1669 r2｜canonical review schema 纠错与 supersession

## 裁决

**FAIL-CLOSED / SUPERSEDED，100/100 纠错置信度**：`FAIL_CLOSED_M1669_CANONICAL_REVIEW_SCHEMA_MISMATCH__SUPERSEDED__NO_M1670_RELEASE`。

此前封存的 M1669 目录在文件 SHA 与递归 seal 层面完整，但其 `review.json` 不能被 M1668 的 `validate_future_authorities()` 消费。因此先前“授权 M1670 release authoring”的结论撤回；当前不得生成、封存或执行 M1670，也不得触发远端 launch、capture、GPU 或 attempt。

## P0：下游 exact schema 不闭合

对未修改的 M1668 validator 直接执行，canonical review 在 release 文件检查之前即返回：

`M1668Error: M1669 review mismatch`

具体差异如下：

- validator 读取 `review['score']`，canonical 仅写了 `score_out_of_100`，因此实际看到 0；
- identity 缺少 `m1647_source_sha256`、`m1648_review_sha256`、`m1649_release_sha256`、`profile_sha256`；
- identity 多出 `author_manifest_sha256`、`author_outer_seal_file_sha256`，违反 exact equality；
- authorization 缺少 `release_authoring`，却使用了 `m1670_release_authoring`；
- authorization 还多出 `attempt_write`、`remote_write`，同样违反 exact equality。

原 canonical 的 7 个 manifest 成员、manifest SHA `4168cfab...` 和 outer file SHA `f5425757...` 仍重算正确。这说明问题是“封得很完整的错误 schema”，不能用 seal 完整性抵消消费端契约失败。

## replacement candidate 只读验证

原封存目录内的 `cpython312_hammer.json` SHA 为 `8543f206...`，其 `score`、12-key identity 和 4-key authorization 与 M1668 exact clause 一致。纠错 hammer 在仓库内创建一次临时双封 review/release fixture，并调用未修改的 `validate_future_authorities()`，本地 schema 消费通过。

该 fixture 仅为 schema 测试：本机不存在远端专用 `/opt/conda/envs/sdformerflow/bin/python3.10`，所以临时绑定了当前解释器。`remote_interpreter_validated=false`，不得把此结果写成 remote launch preflight。

## 最小恢复路径

1. 不得原地编辑已经双封的 canonical M1669。
2. 由新的明确 authority 选择：发布不同路径的 source successor，或原子隔离旧 canonical 后发布 exact-shape replacement；编号和路径由主线统一分配，避免碰撞。
3. replacement 必须由不同作者再次 seal/review。
4. 真实 M1670 release 生成后，必须用真实 review/release 双封和远端解释器再次运行 `M1668.validate_future_authorities()`。
5. 在该门关闭前，M1670 authoring、release、remote launch、capture、GPU、attempt 和 retry 均为 false。

本纠错只执行本地只读 schema 审计与临时 fixture；没有连接远端，没有运行 capture/GPU/EDA，没有写 production attempt，也没有 commit/push。
