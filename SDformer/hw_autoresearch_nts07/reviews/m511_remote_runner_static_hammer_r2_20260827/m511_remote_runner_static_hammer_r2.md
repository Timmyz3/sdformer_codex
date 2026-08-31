# M511 remote exact runner r2 独立静态复审

日期：2026-08-27  
结论：`STATIC_GO__EXACT_ENV_PINNED_ONE_SHOT_CAPTURE_ONLY`  
评分：**95/100**  
P0：**0**  
P1：**2**  
runner / producer / GPU / VCS / DC / DSE 实际执行：**否**

## 结论

r1 的 canonical rollback P0 已关闭。新 runner SHA 为：

```text
788d674eb3df23f3af6cd8525b3a6471fd26596459e298ef8c9df7aa6369b7fa
```

可以授权一次 exact-env-pinned 的远端 S10 capture。授权边界只到 capture；当前
盘上 verifier SHA `5a83c45f...` 尚无必选 `--attempt-dir`，因此在 verifier 修订、
静态审阅并同时通过 capture + final runner attempt admission 前，payload 仍不得
进入 envelope repair、cycle simulator 或任何性能口径。

## r1 P0 关闭

新 EXIT trap 只在 canonical 存在时回滚；`mv` 失败不再被吞掉，而是 exit 99；
rename 后还要求 canonical 不存在且 quarantine 为目录。对所有能够进入 Bash
EXIT trap 的 post-producer 失败，这已经实现 fail-closed rollback。

initial member seal 和 outer seal 也会在最终 top-level attempt seal 前重新运行
`sha256sum -c`。因此初始身份、资源 preflight 或 one-shot 记录漂移不能进入最终
runner PASS seal。

## SIGKILL 窗口：跨产物 admission 足够，但必须强制实现

producer 发布 canonical 后到 runner 完成之间，SIGKILL/宿主掉电不能执行 EXIT
trap。采用“payload verifier 必须同时验证 final attempt PASS seal”的跨产物
admission 可以正确关闭该窗口：

- 若终止发生在最终 attempt seal 之前，attempt 缺失或 seal 不完整，verifier
  必须拒绝 canonical；
- 若终止发生在最终 attempt seal 之后、`runner_success=1` 之前，output seal、
  输入身份、cgroup 终态、initial 复验和 final seal 已全部完成，接受是安全的；
- 因此无需把不可捕获信号误判成 runner P0，但 final attempt 不能只是可选回执。

修订后的 verifier 至少必须满足以下 admission 合同：

1. `--attempt-dir` 为 required，且解析到 isolated repo 下固定的 M511 attempt 路径；
   capture-dir 同样必须是 contract canonical 路径，二者均拒绝 symlink。
2. exact 检查 initial 与 top-level 文件集合，验证两层 member seal 和 outer seal；
   top-level seal 必须恰好提交 `initial/SHA256SUMS.seal.sha256` 与
   `POSTCAPTURE_PASS.txt`。
3. 从 `initial/identity.sha256` 解析并硬要求 canonical runner path 的 SHA 为
   `788d674e...`，同时重验 producer `e16a...`、contract `e556...`、r4 review
   seal `1d2334...` 和 `docs/359` `dedde7...`。
4. `ATTEMPT_CONSUMED.txt` 的 repo root、output、status 与 capture-dir 交叉一致；
   `POSTCAPTURE_PASS.txt` 的 status/claim boundary、manifest SHA、capture outer-seal
   file SHA 与实际 capture 一致。
5. cgroup end 账本必须与 start 交叉一致：failcnt 不增长且为 0、under_oom 为 0、
   oom_kill 不增长。只有这些检查全部通过后才能开始 payload 解码并发布 verifier
   PASS。

这套跨产物 admission 在逻辑上充分。当前 verifier 尚未实现，所以它是后续消费
的硬门，不是本 runner capture 的阻塞门。

## 其余通过项

1. runner canonical path、caller-supplied literal runner SHA 和 literal isolated
   repo root 都在 attempt 前检查；`bash -n` 通过。
2. producer `e16a454d...`、contract `e556743d...`、r4 review outer-seal file
   `1d2334c7...`、`docs/359` `dedde7ce...` 绑定正确；21 个 contract inputs 本轮
   再次独立复算，全部匹配。
3. 三次间隔 10 秒的 idle/cgroup gate 和最终 idle gate 都位于 atomic attempt
   `mkdir` 之前；资源失败不消耗 one-shot。
4. canonical/attempt/quarantine 不存在门、output/attempt parent 可写门和 2 GiB
   空间门闭合。当前 canonical 与 attempt 均不存在。
5. cgroup start/end 对 failcnt、under_oom、oom_kill 的处理正确：历史 oom_kill
   可非零，但新增 OOM kill 不能形成 PASS。
6. initial identity 在 attempt 前先生成并校验；fixed-directory `mkdir` 是 atomic
   single-owner election；任何 capture 后失败都不会生成有效 final attempt seal。
7. 成功路径只在 final attempt member/outer seal 验证后置 `runner_success=1` 并
   撤销 EXIT trap。

## P1

1. **跨产物 admission 尚未落到当前 verifier。** 当前 verifier `5a83c45f...`
   只有 contract/capture/output 参数，不读取 attempt，也不 pin runner SHA。capture
   可先执行，但 payload admission 必须等 verifier 新 SHA 静态通过。
2. **initial seal 复验晚于 PASS 文本写入。** runner 先写
   `POSTCAPTURE_PASS.txt`，随后才复验 initial seal。复验失败不会形成 top-level
   final seal，强制 admission 下不会被接受，因此不阻塞；为减少人工误读，后续
   可把 initial 复验移到 PASS 文本创建之前。

## 唯一授权方式

caller 必须把下列两个值都写成字面量，不能在命令行用 `sha256sum` 或 `pwd`
动态生成：

```text
M511_EXPECTED_RUNNER_SHA256=788d674eb3df23f3af6cd8525b3a6471fd26596459e298ef8c9df7aa6369b7fa
M511_EXPECTED_REPO_ROOT=<远端 isolated SDformer repo 的绝对字面路径>
```

启动前仍须确认 canonical 与固定 attempt 目录不存在。本文不授权直接运行
producer，不授权旧 verifier 接收结果，也不授权 RTL/cycle/speedup。`docs/359`
未修改。
