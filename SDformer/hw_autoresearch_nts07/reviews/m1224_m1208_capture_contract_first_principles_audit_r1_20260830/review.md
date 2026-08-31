# M1224 M1208 capture contract 第一性原理审计

## 裁决

`NO_GO_PROMOTE_M1208__GO_PARTIAL_SALVAGE_AFTER_HAMMER__AUTHOR_MINIMAL_LIVE_DEAD_SUCCESSOR`。

M1208 不是 GPU、checkpoint、decoder cohort 或模块复用故障。失败来自一个精确的合同错误：R2 把 **105 个静态存在的 ATLIF** 全部要求为每模块 40 次 runtime call；但 H60 的 12 个 `sn_v` 是静态安装、运行时合法不执行的死路径。

## 第一性原理调用账

模型为 `MS_SpikingformerFlowNet_en4`，Swin 深度为 `2+2+6+2=12`。每个样本都执行完整 encoder、两个 residual block 和四级 decoder。cohort 只选择 10 个 C1 输入与三组各 10 个 decoder 输入，capture loop 中没有按 cohort 裁剪 forward。因此四个 C1 Conv 与四个 ConvTranspose 都应为 40 次；staging 的 320 个 payload stem 也逐样本证明这一点。

配置的 `all_non_qk` 会静态安装每个 attention 的 `sn_v` ATLIF；H60 patched forward 只执行 Q/K，随后 `attn = k_orig * gate`，不执行 `linear_v/bn_v/sn_v`。绑定的 ep35 authority 已给出直接先验：`atlif_activity.csv` 只有 93 个 live ATLIF，每个 `calls=10`；12 个 `...attn.sn_v.spiking_neuron` 全部不在 runtime 表。

| 类别 | 每样本 live module |
|---|---:|
| C1 Conv3x3 | 4 |
| Decoder ConvTranspose | 4 |
| ATLIF | 93 |
| FC1 / FC2 | 12 / 12 |
| Patch embed | 8 |
| BatchNorm | 78 |
| Q/K projection | 24 |
| Attention parent | 12 |
| **合计** | **247** |

静态 inventory 为 259；差值恰为 12 个 dead `sn_v`。远端 payload 文件名中的全局 order 对所有 40 个样本严格满足 `s*247 + [220,223,226,229,232,236,240,244]`，与上述 live 账完全吻合。R2 的 `259×40` 不是实际 forward 合同；正确 ordered 总数应为 `247×40=9880`。

## staging 中真实存在的证据

- Attention：480 个 NPZ，严格 `40 samples × 12 blocks`；逐文件 SHA、Q/K/gate 非空、sample key 与冻结 cohort 均通过。manifest SHA 为 `edbe96ceb23cfc9a104eb00000becf8fc31c5bee73f0aa2e5d9a20a8643a40e0`。
- C1/decoder payload：640 文件，即 `40 × 8 targets × {fp32.zlib, support/sign bitpack}`；总计 1,082,771,863 bytes，完整内容人口 digest 为 `aa8ed1399661f842a47883fa39ba17519f6cc5f3207cb9feef02cd6ae33de774`。
- 身份：ep29 checkpoint `2144df...286a`、config `c7b5b9...d955`、load missing/unexpected 与 overlay missing/unexpected 均为 0。

这些文件是失败 staging 中的局部 payload，尚无 staging 递归 seal，不能冒充 canonical M1208。

## 当前不可恢复项

`writer.close()` 在 R1 写文件之前抛错：ordered 文件在源码第 487 行以后，runtime/ATLIF 在第 490–496 行。因此以下数据只存在过于已经退出的进程内存，磁盘没有：

- `unified_ordered_records.jsonl` 的 9880 条完整行及 tensor statistics；
- `execution_trace.json`；
- `operator_runtime.json`；
- `atlif_activity.json`；
- 根 manifest、`RUN_COMPLETE` 与递归双 seal。

八个 payload 的 order 只能定位八个 retained hook；attention 又是独立 collector。用它们补出其余 239 个 hook、算子统计或 ATLIF 活动属于伪造，明确禁止。

## 可执行决策门

1. **当前 M1208：FAIL CLOSED。** 不得重命名 staging、不得重试已消费 attempt、不得声称 ordered/runtime/cycle/speedup/energy/PPA。
2. **局部 salvage：可先写 source。** 新的不同作者 hammer 必须逐 SHA 绑定 480+640 文件、FAILED/attempt/source/config/checkpoint/cohort；输出只能标为 partial component payload evidence。
3. **最小 successor：可写 source，不碰 GPU。** 静态 topology 仍要求 ATLIF=105；runtime contract 改为 93 live×40 + 精确 12 个 `sn_v`×0，并按 `(sample,module)` 验证 247 个 live 名每样本恰一次。失败诊断应先落盘，canonical publication 仍在全部 gate 后。
4. **新 capture 准入。** fresh namespace、40 个冻结样本、9880 ordered、480 attention、640 payload、93 ATLIF runtime rows 各 calls=40、显式 12 dead 名、完整 execution/operator/ATLIF/manifest/RUN_COMPLETE/双 seal，且需 fresh result hammer。

本审计只读远端，不授权远程写、GPU、capture 或 EDA；docs/359 SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

