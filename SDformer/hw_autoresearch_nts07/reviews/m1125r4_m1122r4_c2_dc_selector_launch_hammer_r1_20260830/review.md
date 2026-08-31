# M1125r4 C2 final launch hammer

结论：`GO_ONE_ATTEMPT`。本轮只执行静态审计和隔离 mock；没有运行真实 launcher、engine、`pgrep`、`lmstat`、DC 或 VCS，也没有创建 M1122r4 的 attempt/result/work/failure/lock。

冻结身份包括 M1124r4 launcher、source contract、launch receipt、author receipt，以及 M1122r4 engine、engine contract、engine author receipt、M1121 和 M1123r4。789 项检查通过，24 个 mutation/attack 全部被拒。

已确认的启动语义：launcher 只接受零参数和精确 `env -i` 根环境；Python 固定为 3.10.18；子进程环境与许可证路由均由常量构造；唯一子命令为 pinned Python `-I` engine `--authorized-launch`；成功、非零返回和异常路径均清理精确的 mode-0700 私有 HOME，返回码/异常不被吞掉。

已确认的门控语义：任何旧 attempt/result/work/failure/lock 均在 child 前拒绝；同 UID EDA 精确进程名碰撞与 `pgrep` 诊断失败均拒绝；MemAvailable 与 commit headroom 各至少 8 GiB；许可证路由缺失或 `lmstat` 非零均拒绝。engine 在 M1125r4 自洽外封发现后最多消费一次新 attempt，失败只进入 quarantine，禁止自动重试和复用 namespace。

唯一授权命令仅供 root 在重新确认资源/冲突后执行一次：

```text
/usr/bin/env -i LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/usr/bin:/bin TMPDIR=/tmp PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/pytorch310/bin/python3.10 /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_m1122r4_c2_dc_selector_async_observation_authorized_launch_r1.py
```

本 hammer 不授权 agent 执行，不授权第二次调用、自动重试、参数或环境覆盖，也不产生任何论文性能/PPA 声明。
