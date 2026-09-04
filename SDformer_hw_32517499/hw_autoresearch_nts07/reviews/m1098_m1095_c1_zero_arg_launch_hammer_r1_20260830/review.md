# M1098 M1095 C1 zero-argument launch hammer

结论：**PASS 100/100，P0/P1/P2 = 0/0/0。** M1098 作为 external launch root，授权 root 使用下列 exact command 执行 M1095 **唯一一次**。本评审没有运行生产 preflight/full replay，也没有消费 attempt。

```bash
/opt/anaconda3/envs/pytorch310/bin/python3.10 -I \
  /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/system_simulator/scripts/run_m1095_m1094r2_c1_zero_work_full_replay_zero_arg.py
```

launcher 后不得附加参数，不得用环境变量注入 metric/authority，不得自动重试。

## 为什么可执行

- launcher SHA256：`74576584bcf3140a17d935f7f2bce2fb7fe6a373e8e4b2b0666f5e797e0a5f3b`。
- 只接受 `len(sys.argv)==1`，并要求 exact Python 3.10.18、`-I`、no-user-site 及解释器 SHA。
- 所有 predecessor authority、population 和 Python/docs 身份均在源码硬编码；没有 `os.environ/getenv` 或 caller metric/authority CLI。
- 顺序固定为 identity/resource/freshness → atomic lock → under-lock freshness → attempt mkdir/write/seal → exhaustive preflight → iterator once → recursive seal → `renameat2(RENAME_NOREPLACE)` publish。
- exhaustive preflight 必须为 `812160 tasks × 3 designs = 2436480` 个 work value；bool、nonfinite、population、caller-work 攻击全部拒绝。
- argv、非隔离 Python、污染环境、重复 attempt、stale lock、existing result/work/quarantine、claim-boundary 攻击均 fail-closed。
- caught post-attempt failure 进入 sealed quarantine；任何 attempt/stale evidence 都禁止 retry。

## External-root 规则

launcher 不读取 M1098 hash，这是刻意的：本目录的 review、manifest 与 outer seal 是 root 的外部信任根。root 在执行前必须重新验证本目录双封、review status 和 launcher SHA，同时确认 result/attempt/work/quarantine/lock 全部不存在。执行只能发生一次。

即使未来 M1095 成功，产物仍只是 `raw CPU-model`，`speedup_admitted=false`、`paper_citable=false`，必须经过 M1099 receipt-blind result hammer 后才能讨论 matched cycles/speedup。

本评审只进行了静态、read-only authority probe 和临时目录 synthetic 攻击；生产 canonical preflight、iterator、attempt 与 namespace 均未触碰。`docs/359` SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
