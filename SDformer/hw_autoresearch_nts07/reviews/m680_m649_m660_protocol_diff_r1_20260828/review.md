# M680｜M649 与 M660-r4 只读协议差分

## 裁决

`PASS_PROTOCOL_DIFF__NO_GO_NEXT_CAPTURE_UNTIL_CUDNN_TF32_POLICY_REPAIRED`。
严重度为 **P0=0 / P1=1 / P2=2**。当前 M660-r4 已正确 fail closed，未发布
canonical payload；但下一次 GPU capture 在协议统一、重新立合同并独立评审前
不得启动。

## 根因

最可能且已有强因果旁证的根因是 **cuDNN TF32 执行政策漂移**，不是 packing、
计数、样本顺序、BN、reset 或 hook 位置。

M649 和更早的 M511 都没有设置或记录 deterministic/TF32。两者使用相同
config/checkpoint/dataset/preprocess/build_model/no-running-BN/reset/hook 路径；M511
保留的 `s00_d0` bitpack 有 839,586 个 one，而且它的 1536 个逐通道 one count
逐项全部等于 M649。M660-r4 则在模型构建前调用
`configure_deterministic_execution()`，强制 deterministic、cuDNN benchmark=false、
CUDA matmul TF32=false、**cuDNN TF32=false**，runner 又用 `env -i` 和
`CUBLAS_WORKSPACE_CONFIG=:4096:8` 启动。

冻结 YAML 的 `runtime.allow_tf32: true`、`cudnn_benchmark: true` 并不会修正这个
差异：`profile_nts11_hardware_p0.load_config()` 只解析/改 loader 和 input size，
不消费 runtime 数值政策；仓库中训练入口会消费这些字段，但本次 profile
build path 不会。因而 M660 producer 的显式 TF32=false 是实际生效值。

“少 1182 个 one”容易误导。M660 有 838,404 个 one，净差确实是 -1,182；但
两份 4,608,000-bit 数据实际有 **264,066 位不同**：132,624 个 1→0，131,442
个 0→1。差异遍布 10/10 timestep 和 1536/1536 channel；M660 对 M649 有
1499/1536 个逐通道 count 不同。这是阈值型 SNN 对上游浮点卷积执行模式变化
的典型放大结果，不是少写 1182 位或 bit-order 错误。

工作区在本评审期间出现了已双封但尚未独立准入的 M681 三档 S00/D0 诊断。
本评审只将其作为旁证：legacy 和 deterministic+cuDNN-TF32-on 均逐 bit 得到
M511 SHA `ad2251...`；deterministic+cuDNN-TF32-off 得到 M660 SHA `10981f...`。
三档报告中 deterministic、CUBLAS、matmul-TF32 和 benchmark 条件可比时，唯一
随两种 SHA 改变的报告字段就是 `cudnn_allow_tf32`。M681 尚缺独立 result hammer
和完整 launch/CuPy provenance，不能单独授权 payload 或 GPU 重跑。

## 逐段结果

| 协议段 | M649 | M660-r4 | 结论 |
|---|---|---|---|
| config/checkpoint/M511 | 相同冻结 SHA | 相同冻结 SHA | 排除身份漂移 |
| dataset/sample/preprocess | 相同 DSEC valid S10、顺序和函数 | 相同 | 排除样本/预处理 |
| device/dtype/shape/stride | CUDA FP32；`[10,1,1536,15,20]` | prior gate 已通过 | 排除布局 |
| BN | no-running，78 modules | 同一 helper，78 modules | 排除 BN |
| reset | build 后 reset；每样本 forward 前 reset | 相同，额外只读 theta check | 排除 reset |
| hook | exact ConvTranspose2d post-hook 的 `inputs[0]` | 相同；D1 hooks 尚未触发即在 D0 失败 | 排除 hook |
| CuPy | config 指向 cupy，但实际 backend/version 无 receipt | `env -i` 使 config cupy 生效意图更强，仍未记录 actual backend/version | P2 |
| deterministic/TF32 | 未设置、未记录 | deterministic + TF32-off | **直接分叉** |
| runtime receipt | 无 | 双封完整 receipt | M649 P2 延续 |

P1 是把 M649 的跨运行精确 count 当 miter，同时又在 M660 中改变算术政策。严格
fail closed 本身正确；错误在于合同同时要求“字节级复现旧运行”和“采用不同
浮点执行模式”，这两个约束不相容。

## 可证据化修复与下一合同门

下一合同应采用最小协议修复：保留 deterministic=true、warn-only=false、
cuDNN deterministic=true、benchmark=false、CUDA matmul TF32=false 和
`CUBLAS_WORKSPACE_CONFIG=:4096:8`，仅将 **cuDNN TF32 显式设为 true**。这是
M681 旁证中恢复历史 SHA 的最小单轴变化，也与 YAML `allow_tf32:true` 一致。

必须同时满足以下门，才可请求新的 fresh hammer 和一次新 one-shot：

1. 新 producer/runner/contract 全部新 SHA；旧 consumed attempt 和失败 staging
   只读封存，不恢复、不覆盖。
2. M681 先做独立静态/result hammer；确认三个结果双封、producer SHA、三档
   runtime fields、M511/M660 reference SHA，无 GPU 重跑。
3. capture 启动前 receipt 明确断言 cuDNN TF32=true，其余五个 deterministic
   字段保持上述值；合同注明 capture overlay 覆盖 YAML 中未被 profile 消费的
   `cudnn_benchmark:true`。
4. receipt 新增实际 SNN backend=`cupy`、CuPy 版本、每类目标 neuron backend、
   `torch.get_float32_matmul_precision()`，禁止只根据 config 推断 backend。
5. 同一次 forward、同一个 D0 hook tensor完成 exact-binary audit、bitpack、立即
   unpack elementwise miter和 raw/packed SHA；跨运行 M649 count 只作 reference，
   不替代 same-tensor miter。
6. S00/D0 在继续 S10 前必须满足：shape/stride/dtype exact，one=839586、
   zero=3768414、packed SHA=`ad2251...`、对 M511 Hamming=0。失败则停止且不写
   其他 activation candidate。
7. 增加 preprocessed `x` SHA 和至少 patch-embed、encoder bottleneck、D0-input
   三个只读 sentinel SHA；若再次漂移，必须报告第一处分叉而非只报最终 count。
8. D0/D2/D3 仍逐 tensor 证明 `{0,1}`；D1 theta/folded miter政策不放宽。

本评审未运行 GPU/EDA，没有修改任何作者、result、contract、runner 或
docs/359 文件，也没有产生性能数字。

