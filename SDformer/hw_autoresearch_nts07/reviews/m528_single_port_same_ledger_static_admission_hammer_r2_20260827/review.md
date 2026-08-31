# M528 r2 static admission 独立锤审

## 裁决

**100/100，P0=0、P1=0、P2=0；允许使用固定的三个环境变量调用 r2 runner 恰好一次。** 本审阅没有运行 runner/analyzer、CPU production、EDA、GPU 或 RTL，也没有修改 admission 与 `docs/359`。

新 admission 的 schema/status 精确为 `m528_single_port_same_ledger_static_admission_v2` / `AUTHORIZED_ONE_M528_R2_CPU_PRODUCTION_RUN`；授权值为 CPU=1、EDA=0、GPU=0、RTL=false。它完整通过 runner 实际使用的 jq 谓词和三个 64-hex identity-key 检查，严格 JSON 无重复键，member sidecar 与 outer seal 均通过。

## 身份与证据链

- Admission JSON：`5faab40ef7f727a7feab963a3601dd93d2cf74a514810a74c733c1c2e7c37170`
- Admission member sidecar：`bb6cdbf2053e33f210b6ab109249f0ab645c2bb3ce1201f81699f42e2d429080`
- Admission outer-seal file：`6860f106a502f894a3de71a08511156f7e2a0482503c8e3ca624ad5d3b6a2098`
- Runner：`36152576c07f8da496af99b2632a11ebfe04be2a00bc913e55b6f73ae866d386`
- Analyzer：`c611f8c98253e44ccf93743d47476da0adc9835b013b247bc4e2d821953afb8a`
- r2 execution contract：`fc0c3aee93d4055f0f1feda8268009d82d957c4b4d0adf5111ad8464122a95e2`
- Governing contract：`d0e3728f3a9991cf97c6af88181cd51996e457228b120f2b706a8986caf9ca51`
- Python：`9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115`
- Author outer-seal file：`7c9fbbc8d5b13a6d27c2a9b5ef072c0cbb3e313f144013558d09b30d35bf8f29`
- r2 static review JSON / outer-seal file：`259ee0e671113f67c01033f16b629be94b9b74f86c7ce11745c6406f7c2f16eb` / `2b296a74b0068a63d0585553988759155a884d4e7168cfed99dbb219ba080c77`
- Resource review JSON / outer-seal file：`789ce8099a711490c214cc8bf3efa897bcfa668b0e3021d98dacf52f8970824c` / `45c6615740093bb8162153181d24bfd5c693e3c4c03cdbead0b0fa45355df23a`
- `docs/359`：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

所有 live hash 与 admission/runner pin 一致，author、static review 和 resource review 的内外双封全部通过。static review 的 98/P0=0/P1=0/P2=3 与 admission 完全一致；资源依据的 6 GiB 静态上界、48 GiB 门、8x 裕量及保留门也一致。

封存 r1 admission 以同一个 r2 schema/status/runner/runtime 谓词检查返回 `rc=1`，不可复用。

## 未消费与动态门

审阅时 r2 canonical、attempt sentinel、work 与 quarantine 候选全部不存在。18:10:01 非授权快照为 commit headroom `60,376,672 KiB`、MemAvailable `415,284,660 KiB`、SwapFree `57,278,204 KiB`，OOM 三项全零且列出的 UID-local EDA/sim 进程均为零。

这张快照**不能替代 runner 门**。runner 启动时仍必须重新完成三次资源快照、OOM 检查和 UID-local EDA collision 检查；任何一项变化均应 fail closed。

## 唯一允许的 invocation 身份

```bash
M528_EXPECTED_STATIC_ADMISSION_PATH=contracts/m528_single_port_same_ledger_static_admission_r2_20260827.json \
M528_EXPECTED_STATIC_ADMISSION_SHA256=5faab40ef7f727a7feab963a3601dd93d2cf74a514810a74c733c1c2e7c37170 \
M528_EXPECTED_RUNNER_SHA256=36152576c07f8da496af99b2632a11ebfe04be2a00bc913e55b6f73ae866d386 \
system_simulator/scripts/run_m528_h67_single_port_same_ledger_recompute_r2_exact_sha.sh
```

上面仅授权执行恰好一次，禁止 output/worker override。即使 raw run 成功，它仍需新的独立 result hammer；不能直接成为 RTL、PPA、energy、full-network、system speedup 或 DATE headline。
