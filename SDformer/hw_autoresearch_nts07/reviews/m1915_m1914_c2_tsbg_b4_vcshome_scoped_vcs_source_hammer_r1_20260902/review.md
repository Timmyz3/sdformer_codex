# M1915：M1914 TSBG scoped-`VCS_HOME` VCS runner 静态打铁

## 裁决

**FAIL，82/100，P0/P1/P2 = 0/2/0。不得启动 M1914 的 license、attempt、VCS 或 simv。**

M1914 已把 M1907 指出的唯一工具环境差修对：`lmutil` 不接收 `VCS_HOME`，唯一 VCS compile 与唯一 simv 都接收固定的 `/opt/synopsys/vcs/V-2023.12-SP1`。冻结 RTL、adapter、filelist、SVA 与 TB 的 SHA 全部吻合；M1898 attempt/failure 与 M1907 failure review 均可双封复验；M1914 的 attempt/result/failure/lock namespace 在本次静态审阅时全新。

但 exact runner 存在两个必然阻断启动的 P1：

1. `RUNNER` 指向不存在的 `run_m1914_m1880_c2_tsbg_b4_cleanenv_vcshome_directed_vcs_one_shot.sh`，而被审文件是 `run_m1914_m1880_c2_tsbg_b4_vcshome_scoped_directed_vcs_one_shot.sh`。因此真实 runner 无法通过自身 `sha_exact`。
2. runner 要求 M1907 状态为 `...HAMMER_DO_NOT_RUN`，而精确冻结的 M1907 `review.json` 实际状态为 `...HAMMER__DO_NOT_AUTHORIZE_ATTEMPT`。SHA 绑定正确，但紧随其后的语义绑定必然失败。

两项都发生在 namespace、license 和 attempt 之前，所以当前结果是安全的 false negative，而不是越权执行；但也因此不能签发题目指定的 PASS 状态。

## 已通过的治理检查

- clean direct shebang、绝对 helper/EDA 路径、两个 64-hex inert 参数、无 `eval`/`source`/动态命令分派；
- same-UID EDA 与 common-shell 截断门、内存与 commit headroom 门；
- `mkdir LOCK` 成功后才设置 `LOCK_HELD=1`，signal trap 使用明确的 130/143/129 退出码；
- WORK_ACTIVE 在 WORK 创建前置位，attempt 在 license/EDA 前创建并双封，禁止自动 retry；
- 唯一一次 license、compile、simv，compile 带 `-assert svaext`；
- PASS token 必须恰好一次，assertion/error/fatal 负向检查存在；
- success/failure 均使用 no-replace publish 加目标存在、源消失与双封 postcondition；
- docs/359 SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 最小后继

新建 additive namespace，不覆盖或别名化 M1914。保持正确的 `VCS_HOME` scope、signal-safe trap、`LOCK_HELD`、冻结 source SHA 与 one-shot 治理不变，只修两项：`RUNNER` 必须命中后继自身真实文件名；M1907 status predicate 必须逐字匹配冻结值（含双下划线与 `DO_NOT_AUTHORIZE_ATTEMPT`）。随后重新做独立 source hammer，才可讨论一次性 launch release。

本审阅没有执行 license 查询、attempt、VCS、simv、DC 或 PT，也不授权任何性能、面积、能量或论文准入主张。
