# M511 payload verifier 独立静态打铁 r3

最终结论：`STATIC_GO__SAME_ISOLATED_REMOTE_ROOT__PAYLOAD_ONLY`，98/100，P0=0、P1=2。本轮只读 verifier、runner、冻结 contract 与既有静态 review；仅做 Python 静态 compile 和 runner `bash -n`，没有运行生产 capture/payload、checkpoint/model、GPU、VCS、DC 或 DSE。

最终审定身份：verifier `222d0402a57789671c975bac4a59a34a5188279b6b6a02319ddd26ad37c9ed1b`，reviewed runner `788d674e...`，contract `e556743d...`，runner r2 review outer seal file `c1621fc2...`。`docs/359` 仍为 `dedde7ce...`。旧 verifier `5a83...`、`acd9...`、`a3e7...`、`be560...` 均 superseded，不得执行。

本版已关闭全部阻断项。四个命令行路径先按 lexical absolute path 要求严格等于当前 isolated HW root 下固定 contract/capture/attempt/output，再拒绝 leaf symlink，最后才 resolve。contract canonical path 和 SHA 固定；exact 21 input name set 在同一 repo physical root 内做实际 start/end path+SHA rehash，checkpoint 再做 591,167,876 B 和 SHA 交叉检查。manifest contract path/SHA、load audit、eval、packing、CUDA fence、四模块结构/weight metadata 与 claim boundary 全部 fail closed。

capture seal 必须是 exact 42 members：`manifest.json`、精确 `RUN_COMPLETE.txt` 与按冻结 sample-major/module-minor 顺序构造的 40 个 bitpack。每个 bitpack 都做完整文件 SHA、size、全量 popcount；输入 shape 按 C-order `T,B,C,H,W`，逐 timestep slice 使用 `prod(B,C,H,W)/8`，四层均 byte-aligned。冻结总量闭合为 696,240,000 bit 和 87,030,000 B；record order、sample/module/path/shape、active aggregate 均逐项闭合。

runner attempt admission 也已成为必选门：固定 canonical attempt；全树 exact 8 files、仅 `initial` 目录、无 symlink；initial seal 传递覆盖 exact 3 members，top seal exact 提交 initial outer seal 与 `POSTCAPTURE_PASS.txt`。initial/final key set 精确；三次 preflight 以 6 行、sample 1/2/3 顺序与三 cgroup 字段解析；identity 必须是 runner/producer/contract/r4 review outer/docs359 五个 canonical resolved paths，并 pin runner SHA。final receipt 与 capture manifest/seal、cgroup start/end 和 capture-only claim 交叉绑定，整条 attempt 在解码前和结束时各验证一次。

发布事务满足当前 fail-closed 边界：输出 staging 先生成 seal，再完整 `verify_seal` 并要求 exact 2 members+精确 completion marker；只有已审定对象才 atomic rename 到 canonical。故 rename 后的 SIGKILL 只可能留下已经静态完整验证的对象；普通 postpublish 失败首先把 canonical 原子 quarantine，成功后无 fallible PASS print。

P1-01：weight content SHA 由 producer 记录，verifier 只检查格式、shape/dtype/bytes/byte-order/layout，未从 checkpoint 独立重建权重；这在 exact-binary-input payload-only claim 下可接受，但下游 weight extraction 必须另行绑定 checkpoint，不能把本 receipt 当 output/cycle 等价证明。P1-02：身份有意绑定 producer 所在的绝对 isolated physical root；remote checkpoint 必须是 repo 内实体文件或 hardlink，不能是逃逸 repo 的 symlink。verifier 应在 capture 原 root 执行；搬运 capture 后只能通过本 verifier receipt 中的 capture seal identity 再核传输，不能在新根静默重解释绝对路径。

授权只到一次远端后置 payload verification。调用方必须字面 pin verifier `222d...`、contract `e556...`、runner `788d...` 和本 r3 outer seal，固定 output 必须不存在。PASS 之后只准 exact envelope repair 与 A0/A1、PGPR/TDR 离线 fast-kill；仍不授权 RTL、cycle/speedup、energy/PPA、system headline，M512 已杀的 phase-balanced scheduler 不得恢复。

