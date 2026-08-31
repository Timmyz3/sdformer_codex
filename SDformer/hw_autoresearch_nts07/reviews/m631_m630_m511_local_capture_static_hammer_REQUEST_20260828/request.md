# M631｜M630/M511 本机 RTX3090 exact capture 启动链独立打铁请求

请独立审阅 `contracts/m630_m511_local_rtx3090_capture_launch_overlay_contract_r1_20260828.json`、M511 runner 与 M511 payload verifier 的当前 exact SHA。

本轮仅允许静态检查、`bash -n`、Python `compile()` 和不可能创建 canonical attempt/output 的负控。禁止生产采集、禁止加载 checkpoint/model、禁止触碰 CUDA、禁止运行后置 payload verifier。

重点寻找：解释器替换是否意外改变 producer/contract/sample/output 身份；caller 是否能用运行时自算 SHA 自通过；one-shot 是否在所有可失败 preflight 之后、producer 之前立即消费；GPU/host/cgroup 门是否三次有序采样且 verifier 完整解析；失败时 canonical output 是否 fail-closed；verifier 是否确实 pin 新 runner 而没有削弱原 40-record 全量 bitpack/popcount/原始文件重哈希语义。

输出必须给出 `GO/NO_GO`、0–100 分、P0/P1/P2，列出 authorized literal command。任何 P0 或 P1 均不得启动。

冻结红线：不得修改 `docs/359_DATE终局冻结_20260813.md`；预期 SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
