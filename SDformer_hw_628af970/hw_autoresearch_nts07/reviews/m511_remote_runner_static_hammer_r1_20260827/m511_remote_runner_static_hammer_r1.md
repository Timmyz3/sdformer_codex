# M511 remote exact runner r1 独立静态打铁

日期：2026-08-27  
结论：`NO_GO__POST_PRODUCER_CANONICAL_ROLLBACK_IS_NOT_FAIL_CLOSED`  
评分：**78/100**  
P0：**1**  
P1：**2**  
runner / GPU / VCS / DC / DSE 实际执行：**否**

## 结论

runner 的身份、资源门、one-shot 消耗点和双阶段回执大体正确，但当前版本
不能授权远端 S10 capture。阻塞项位于 EXIT trap：producer 开始以后任何
非零退出都会尝试移动 canonical output，但 `mv` 的失败被 `|| true` 无条件
吞掉，随后仍以原错误码退出，且没有断言 canonical 已消失、quarantine 已
形成。因此在 producer 已发布 sealed canonical、runner 的 post-check 又失败
时，rename 竞争、权限/文件系统错误等会留下看似成功的 canonical 目录，却
没有最终 runner PASS seal。

当前 runner SHA 为：

```text
79c60cc0bcefec22bd75c6588b88de04dc4b671ddfa1947fac1815769e4ef4dc
```

该 SHA 只允许静态引用，**禁止执行**。修改后必须产生新 SHA 并重新静态审阅。

## P0｜post-producer 失败可能留下 canonical

问题位于 runner 第 24–29 行：

```bash
if [[ "${m511_capture_started}" -eq 1 && \
      "${m511_runner_success}" -ne 1 ]]; then
    mv -- "${m511_output}" "${m511_quarantine}" 2>/dev/null || true
fi
```

producer 自己会先把 staging 原子发布到 canonical；之后 runner 仍要执行输出
seal、全部输入身份、初始 identity、cgroup 终态和最终 attempt seal 检查。
这些步骤任一失败都会进入 trap。正常 rename 时 canonical 会移走，但 rename
失败被忽略，trap 也不检查 `! -e canonical`，所以“失败不能留下 canonical”
并未成立。

最低修复要求：仅在 canonical 存在时执行 quarantine rename；不得吞掉 rename
失败；rename 后必须断言 canonical 不存在且 quarantine 是目录。更稳妥的事务
方案是让 producer 发布到 runner 私有 staging，完成 post-check 和最终 attempt
seal 后才由 runner 原子发布 canonical。至少在后一方案落地前，所有消费者都
必须把最终 attempt outer seal 作为 admission 必需项，不能只凭 producer output
seal 接收结果。

## 已通过项

1. runner canonical path 和外部 caller trust anchor 正确：只有 caller 以被评审
   字面 SHA 设置 `M511_EXPECTED_RUNNER_SHA256`，且以字面 isolated root 设置
   `M511_EXPECTED_REPO_ROOT` 时才能继续。`bash -n` 通过。
2. producer `e16a454d...`、contract `e556743d...`、r4 review outer seal
   `1d2334c7...`、`docs/359` `dedde7ce...` 均在 attempt 前硬绑定并复核。
3. contract 恰有 21 个 inputs；本次独立只读复算全部 21 项均存在且 SHA 匹配。
4. 三次 10 秒训练/GPU idle 与 cgroup preflight，以及最终一次 idle gate，均位于
   atomic attempt `mkdir` 前；任何失败都不会消耗 one-shot。输出父目录的创建
   不等于 attempt 消耗。
5. output/attempt/quarantine 起始不存在门、父目录可写门和 2 GiB 空间门完整。
6. cgroup `failcnt`、`under_oom`、`oom_kill` 均有 start/end 账本；要求
   failcnt 保持 0、under_oom 保持 0、oom_kill 不增长。历史非零 oom_kill 被允许，
   但新增 OOM kill 会拒绝 PASS。
7. attempt 使用固定目录 atomic `mkdir` 竞选唯一 owner。initial 阶段封存 attempt
   身份、资源日志和 identity；postcapture 阶段封存 output hash、cgroup 终态和
   claim boundary。失败不会写 `POSTCAPTURE_PASS`。
8. EXIT trap 保留原始退出码，且会清理尚未移入 attempt 的 `mktemp` 文件；成功
   只在最终 seal 验证后置位并撤销 trap。

## P1

1. **不可捕获终止窗口。** producer 已发布 canonical 到 runner 最终 PASS 之间，
   SIGKILL、宿主断电或 shell 崩溃不会执行 EXIT trap。producer output 本身仍可能
   是 sealed PASS，但 runner attempt 没有最终 PASS。应采用 runner-owned staging
   再发布，或强制下游同时验证最终 attempt outer seal。
2. **最终阶段未重跑 initial nested seal。** 顶层 `SHA256SUMS` 哈希了
   `initial/SHA256SUMS.seal.sha256` 文件，但写最终 PASS 前没有再次执行 initial
   的 member seal 和 outer seal 检查。single-owner 下风险低；修订版应在
   `POSTCAPTURE_PASS` 前重跑两次 `sha256sum -c`。

## 裁决

P0 非零，因此当前 runner **不准执行**，也不准用人工观察 canonical 后继续。
应 supersede 此 SHA，修复 rollback 后重新静态打铁；通过后才可给出包含两个
字面环境变量的唯一远端启动命令。`docs/359` 未修改，仍为 `dedde7ce...`。
