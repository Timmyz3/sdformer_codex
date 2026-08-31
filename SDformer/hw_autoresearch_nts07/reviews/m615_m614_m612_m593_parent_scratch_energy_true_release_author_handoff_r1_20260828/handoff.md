# M615 author handoff｜M614/M612/M593 parent-scratch energy true release

状态：**TRUE RELEASE AUTHORED；STILL NOT EXECUTED；fresh M616 hammer required。**

M615 作者生成了冻结 M612 runner 唯一接受的 production admission 与 companion true release：

- admission `contracts/m614_m612_m593_parent_scratch_energy_true_launch_admission_r1_20260828.json`，SHA `0e194055d4a6ac396b091d6c3d0dba61b94d28d0936ecf89352c96e95a23f630`，outer-seal-file SHA `7cd77b75b5439fe46140b3e8d4889f2b57c1de720200f9b029500a62d4fa9e51`；
- release `contracts/m614_m612_m593_parent_scratch_energy_true_release_r1_20260828.json`，SHA `9f465b9a091ded283bdddb2a37dc596b2cbfed83e48b4f0567ba9297819e8fa2`，outer-seal-file SHA `a474c48cad9650d994de25f6fc9e016ed21df8764ab342f0f7593973511225ee`。

`m614` 仅为共享数字前缀。既有 PAFT artifact 的完整 ID/path 是
`m614_m579_paft_control_single_port_product_capture_r4_result_hammer_r1_20260828` / `reviews/...`；energy protocol 的完整 ID/path 是
`m614_m612_m593_parent_scratch_energy_true_launch_admission_r1_20260828` / `contracts/...json`。两者 exact ID 与 exact path 均不相等，PAFT M614 未修改或覆盖。

admission 使用 runner exact 10-key schema，绑定 M612 shell/Python/adapter、M613 PASS100 双封、M597 contract/analyzer 与唯一 result/attempt/consumed。只读 `verify_authorization` validate-only 通过，没有调用 analyzer 或 `--execute`。M613 已封存的 exact source preflight token 保持绑定。

作者三次 2 秒间隔资源快照全部过门：最小 commit headroom `84,209,324 KiB`、MemAvailable `414,671,592 KiB`、SwapFree `57,212,668 KiB`；session/user cgroup clean，UID-local collision=0。runner 不实施这些资源门，所以 M616 PASS 后 root 必须在唯一 invocation 紧前 fresh recheck。

release 明确 `max_attempts=1`、`still_not_executed=true`、GPU/EDA/remote=false。正式 result/attempt/consumed、runner staging、quarantine raw/staging/final 均未创建。`38.228307918921945%` 与 `1.2622562286593053 mJ/frozen sampled inference` 仍只是冻结诊断预期，不是 admitted result；只允许 component-only claim，正式 raw result 仍须 fresh result hammer。

`docs/359` 未修改，SHA `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
