# M631｜M630/M511 本机 RTX3090 exact capture 启动链独立静态打铁 r1

最终结论：`NO_GO__TWO_IDENTITY_ADMISSION_P1`，90/100，P0=0、P1=2、P2=3。本轮严格没有运行生产 capture、payload verifier、checkpoint/model、CUDA、VCS、DC 或 DSE；canonical output 与 fixed attempt 在全部负控之后仍不存在。

## 阻断项

### P1-01｜caller SHA 不是独立信任根

runner 第 52--57 行只检查：当前 runner 的 SHA 等于 caller 提供的 `M511_EXPECTED_RUNNER_SHA256`。runner 内没有独立的 reviewed SHA `7856fe28...`。因此 caller 可在启动时动态计算当前文件 SHA；安全负控已经证明动态 SHA 能通过第一门并到达 repo-root 门。这会允许被改写的 runner 用自己的新 SHA 自授权，所谓 “literal reviewed runner SHA” 目前只是调用约定，不是 fail-closed 代码事实。

修复要求：增加独立、静态审阅的 launch trust root（例如不接受 caller SHA 覆盖、内部固定 runner `7856fe28...` 的小型 launcher），并由新一轮 hammer 只授权该 launcher 的字面命令。不能继续把运行时自算 SHA 作为合法调用。

### P1-02｜新 Python 只有启动前单次 SHA 门，没有进入最终身份证据

`/opt/anaconda3/envs/pytorch310/bin/python` 的 SHA `9f78cd42...` 只在 runner 第 63--68 行检查一次。它不在 `m511_verify_identity()` 中，不在 attempt 的 `identity.sha256` 中，capture 后也没有复哈希；当前 verifier 还硬要求 identity 只有五个文件。因此 preflight 到 producer exec 之间存在解释器 TOCTOU，最终 receipt 也无法证明实际运行时 Python 身份。

修复要求：把 Python 作为第六个 immutable identity，在 attempt 之前和 capture 之后都按 `9f78cd42...` 重哈希；写入 initial `identity.sha256`；verifier 改为 exact 六文件集合并 literal pin Python SHA。修改 runner/verifier 后必须产生新的 SHA 与 fresh static hammer。

任一 P1 都禁止一次性启动，所以本轮不提供 authorized literal command。

## 已通过的核心检查

- overlay、runner、verifier、producer、producer contract、Python、r4 outer seal 与 `docs/359` 均复算匹配请求身份；`docs/359` 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
- producer contract 的 21 个输入全部逐文件 SHA 通过，包括 591,167,876 B checkpoint；r4 producer review 与 r3 verifier review 的 member/outer seals 均通过。
- runner `bash -n` 通过；producer 与 verifier 均以 `compile()` 通过，未 import、未执行。
- canonical path、repo root、Python SHA、producer/contract/r4/docs 身份门均位于 attempt 前；producer 21-input rehash 在三次资源采样前及 attempt 前再执行。
- 三次有序 sample 1/2/3，每次记录 commit headroom、MemAvailable、SwapFree、GPU free、cgroup failcnt/under_oom/oom_kill 七字段；前两次间隔 10 秒；每次同时执行 idle gate，attempt 前还有最终 idle gate。
- 固定 attempt 由单次 `mkdir` 选主，位于全部资源/idle/身份 preflight 之后、producer 之前；initial receipt、preflight 和 identity 先 seal，再置 `capture_started=1` 调 producer。
- 普通 capture 后失败会将 canonical output 原子移至唯一 quarantine；成功路径重验 output seal、producer/contract/21 inputs、初始 identity 与 cgroup start/end 后才生成 final attempt seal。
- verifier literal pin 当前 runner `7856fe28...`，严格解析三次七字段 snapshot，要求 exact attempt tree/seals；保留 10 samples x 4 modules = 40 records、696,240,000 bits、87,030,000 B 的完整文件 SHA、全量 popcount、逐 timestep 统计、raw source 重哈希与 capture-only claim boundary。
- 缺 caller、错误 SHA、正确 SHA+错误 repo root、动态自算 SHA+错误 repo root 四个负控都在 output/attempt 创建前以 rc=3 退出；负控前后 canonical output 与 attempt 都不存在。

## P2

1. runner 对 canonical output/attempt 的不存在判断只使用 `-e`，没有同时拒绝 dangling symlink；verifier 最终会拒绝 symlink，故不会错误准入，但可能消耗 one-shot 并留下外部 staging/target。建议 runner preflight 与 trap 同时处理 `-L`。
2. Python 可执行文件 SHA 不覆盖 `torch/numpy/spikingjelly` 等 site-packages。即使修复 P1-02，环境包仍可独立漂移。建议至少把关键包版本、安装根与 RECORD/源 SHA 写入 immutable runtime manifest；本条在 exact-input payload-only 边界下暂列 P2。
3. overlay 的 “local RTX3090” 是 observation，不是 runner 的 GPU name/UUID 或 hostname gate；runner/verifier只固定 free-memory/idle/cgroup。建议 receipt 记录并验证 GPU name/UUID、driver/CUDA 与 hostname，避免把另一宿主的 capture 标成相同本机 coordinate。

## 授权边界

`authorized_literal_command = NONE`。修复两个 P1、重新封 runner/verifier/overlay 并通过 fresh static hammer 前，不得创建 attempt，不得启动 M511 producer。即使未来 capture 成功，也只能先运行独立 payload verifier；在 verifier PASS 之前不得进入 cycle simulator，之后也仍不自动授权 speedup、RTL、Synopsys、energy、PPA 或 DATE headline。

