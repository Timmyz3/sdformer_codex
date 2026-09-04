# M1640｜M1626 TSBG/S2 clean-child capture release 独立审核

日期：2026-09-01

状态：`PASS_M1640_M1626_CLEAN_CHILD_CAPTURE_RELEASE_HAMMER__REMOTE_ONE_SHOT_GO`

评分：99/100；P0=0，P1=0，P2=1。

## 结论

M1626 release 通过不同作者只读门，可放行严格一次的远端 parent call。release `identity` 与 M1624 source 在 `validate_future_authorities()` 中构造的 13 键 expected dictionary 完全相等：无缺键、无附加键、无宽松子集比较。M1625 review 的 identity、评分、P0/P1 和 release-only authorization 也与 source 的 exact expected dict 一致。

远端 child 解释器严格绑定为 `/opt/conda/envs/sdformerflow/bin/python3.10`，SHA256 为 `89520a3f2bc6e4f670921bd7a71a66eb0073775e685f6cbefda0dcda7bc42aa0`。source 不只检查字符串，还会在 child launch 前对该路径做 regular-file 与 SHA 校验。

## 一次性边界

release 数组严格为：1 个 parent call、1 个 clean child、1 次 GPU run、1 次 production capture，`automatic_retry=false`，`all_other_runs=0`。result、attempt、work 和 failure 四个 namespace 逐字匹配 M1624 source，彼此不同，审阅时全部 fresh。release 六个 claim（TSBG DSE、AEE、RTL、EDA、performance、paper result）全部为 false。

M1625 review 目录与 M1626 JSON 的内层/外层 seal 均通过严格拓扑和 SHA 检查。release 作者回执显示零 remote write、零 checkpoint load、零 GPU/capture/DSE/EDA run。

## 无远端执行的 runtime-path 验证

审核不连接远端。本地将 `subprocess.run` 替换为必失败哨兵后，调用 source 的 parent 前置路径：source 成功校验已出现的 M1625/M1626 authority，随后精确停在本机不存在的远端解释器依赖，而不是绕过 authority 进入 child/GPU。哨兵记录 subprocess=0，远端连接=0，GPU=0，capture=0，四个生产 namespace 写入=0。

CPython 3.6 和 3.10 各自通过 18 类静态/依赖检查，并各自拒绝 47/47 个变异，覆盖全部 identity 键、解释器、四项数量预算、retry、四 namespace、六 claim、seal 后门及 source 绕过。

## P2 与放行条件

本地只读审查不能证明远端解释器/GPU 实际可用或 capture 必然成功。放行仅限 M1626 指定的单次 parent/child/GPU/capture；远端 source 必须再次检查解释器、checkpoint/config、authority seals 和 fresh namespace。失败不得重试；成功产物仍需另一位作者的 result hammer，在此之前不得声称 TSBG/AEE/性能或论文结果。

`docs/359` 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`；本审核未修改 `ucli.key`。
