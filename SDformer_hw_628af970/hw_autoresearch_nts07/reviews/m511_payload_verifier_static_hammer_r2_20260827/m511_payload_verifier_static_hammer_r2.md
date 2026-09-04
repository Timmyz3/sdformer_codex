# M511 payload verifier 独立静态打铁 r2

结论：`NO_GO__SUPERSEDED_AFTER_WRONG_TOGETHER_REPAIR`，86/100，P0=2、P1=2。本轮只读 verifier、冻结 contract 和 producer；没有运行生产 capture、payload、checkpoint/model、GPU、VCS、DC 或 DSE。

审定 verifier SHA 为 `5a83c45f0cb84e7457c65d581d650d976032b689d205c6c23d108c7048fc9912`。该版已经正确关闭首轮四个问题：HW root 使用 `parents[2]`；sealed exact set 由 `manifest.json`、精确 completion marker 和 40 个精确 call path 构造；seal/outer/member 拒绝直接 symlink；21 inputs 做文件/路径/SHA start-end rehash；40 bitpack 逐文件 SHA、全量 popcount 和逐 timestep byte slicing 正确；发布后没有 fallible PASS print，普通 postpublish exception 的首个恢复操作为 unique quarantine rename。

但 r2 仍有两个阻断性 wrong-together 缺口。第一，命令行 contract 没有被要求位于 canonical contract path，也没 pin `e556743d...`，manifest 的 `identity.contract_path` 未校验；同时 inputs 只做 manifest-set 等于 contract-set，没有 exact 21-name set 和 repo-root containment。这样同 schema/status、同硬编码总量但自洽修改 samples/modules/inputs 的伪 contract 与重封 manifest 能一起通过。第二，`raw_validation_sources.samples[i]` 只校验 sample_id 和三个文件哈希，没有把 raw `sample_key`、文件 basename、event sequence directory 与 contract/records 绑定；不相关的原始样本账本可与 payload manifest 一起重封后通过。

P1-01：capture root 或祖先目录自身的 symlink 没有显式拒绝；成员和两个 seal 的直接 symlink 已拒绝。P1-02：runtime weight content SHA 只做格式检查，未从冻结 checkpoint 独立重建；在 payload-only claim 下可以保留，但不得借此声称 weight/output/cycle 已独立验证。

因此 5a83 版不得执行或作为 simulator admission。后续修订必须固定 canonical contract path+SHA、exact 21 names/containment、manifest contract path，并闭合 raw sample 到 contract 的身份关系。该版本由后续 acd9 版取代。

