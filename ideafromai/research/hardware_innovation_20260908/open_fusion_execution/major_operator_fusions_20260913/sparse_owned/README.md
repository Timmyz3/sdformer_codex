# K4 nonzero-fill／请求感知 exception：执行完成，当前不升格为性能方案

**结论。** 实做了 `w·g = c·(popcount(g)−g_s)+v·g_s` 的系数压缩、源 gate collector、读响应解码、整字请求和完整整数后继窗口。它允许共同值与 exception 在同一源上先代数抵消；不把两项强制算成两次 MAC。请求感知选 exception 有独立的局部增量，但目前真实 r0、r1 的普通 2:4 均更快且 diverse10 AEE 更好。保留表示与请求选择研究接口，停止把这个固定格式作为胜出硬件方案。

**真实 r0 大头与强对照。** 使用 root 当前原生 r0conv2，W 为 96×864，全部非零，实际输入为 {0,1}、θ=1；该模块每次 profile 为 63.701G dense MAC。原始 Conv 无 bias（capture 脚本只在 bias 存在时保存它），NPZ 的零 bias 是 adapter bridge。首帧 64 个真实位置中偶序 32 个校准、奇序 32 个测试，各 T10；这是同帧局部测试，不能称独立帧泛化。所有普通／shift 臂共享 natural K4、全部合法支持、相同 Gram ridge refit、FP32 系数及恢复权限。普通 2:4／3:4 同样获得一次请求优化与每个 H8-K4 总校准 SSE 不超过最小值 1.10 倍的额度；magnitude 固定权重误差支持，Gram 固定校准误差支持。没有训练、参数扫选或 downstream gold 控制执行。

| r0 臂 | 局部相对平方误差 | value CR32 | metadata CR32 | SIMD8 MAC | C | diverse10 AEE |
|---|---:|---:|---:|---:|---:|---:|
| ordinary 2:4 | .058119 | 43,718 | 4,014 | 107,637 | 231,001 | 1.159183 |
| ordinary 3:4 | .012037 | 57,108 | 2,688 | 132,238 | 279,730 | 1.186072 |
| shift 1:4 magnitude | .027161 | 54,499 | 2,688 | 152,278 | 294,552 | 1.177745 |
| shift 1:4 Gram | .028702 | 54,249 | 2,688 | 151,381 | 293,155 | 1.194979 |
| shift 1:4 request | .029085 | 52,882 | 2,688 | 145,137 | 284,177 | 1.176247 |

C = 2×(value+metadata CR32)+MAC+27,900 次共同 8-lane 支持解码，是明确列项的服务计数，**不是周期、GPU 加速或完整层延迟**。Gram→request 单独隔离请求选择：value CR −2.52%、MAC −4.12%、C −3.06%，局部误差 +1.33%，同一 diverse10 AEE 从 1.194979 降至 1.176247。此前 magnitude→request 的 −3.89% 混入了支持选取目标改变，不能全部归因于请求优化。request 相对 ordinary24：value CR +20.96%、MAC +34.84%、C +23.02%；metadata CR −33.03% 无法抵消。

`r0_transfer/transfer.py` 真正序列化 FP32 H8 整字并从 bytes 读回；普通24 使用 3-bit 六选二码，普通34 使用 2-bit omitted index，shift 使用 2-bit exception。跨 32B 元数据边界付实际请求，只有一个 metadata response latch；系数请求只看源操作数，不能预知未读系数为零。五臂字节解码结果与有效 W 的 float64 实际激活点积 maxabs 均为 0。有效载荷含元数据分别为 173,664／254,016／171,072 B（dense 为 331,776 B），不使用 NPZ archive 文件大小冒充硬件压缩率。共同源包为 P2×T10 的 4-bit masks，真实 pack/decode 校验；H8 外循环的 20 个 accumulator vectors 需要重复读源 12 次，共 41,472 个 16B 包，含零包。源收集／im2col、包生成、仲裁、源延迟、目标写回和物理 mux 时序尚未计入 C。

**r1 执行边界已补齐。** `response_execution.py` 在原 Machine 上增加 64B 读 staging、32B collector、32B value 和 metadata latch、16B support；8-lane bit／3-bit count 直接送既有 16×24 乘法宽度，未增加 RF 读端口，实际 decode/select/CR/SR/SW/issue 均入账。四真实帧、两窗口完整 K864→U16 中 shift14 比 dense 省 10.4%，但 `fill_chain.py` 接上同 Machine 的 F／raw merge／gate／U24／V96／连续 PED 和写出后，固定完整 interior 窗口 dense 612,457、ordinary24 604,847、shift14 606,702、允许抵消的 fill 606,688 服务槽：fill 仅比 dense 省 **0.94%**，仍比 ordinary24 多 0.30%。压力序列为 689,882／683,386／684,570／684,570。U、77,760 个 updated 值、77,760 个 gate 值、15,360 个 PED 值均与各自独立有效 W 金标完全一致。上游 PSN/preview、原生 projection convolution 与 global BN 不在这个 continuation 内。r1 保留旧 2*n 元数据编码；进一步紧凑普通24只会增强已胜出的对照。

root 的 [r1 AEE](../root_owned/aee/summary.json) 为 parent 1.159737、ordinary24 1.164651、shift14 1.173541；[r0 AEE](../root_owned/r0_sparse_aee/summary.json) 与追加的 [Gram 控制](../root_owned/control_aee/r0_shift14_gram_summary.json) 如表。全部低于同协议 NB0 diverse10 1.45460286107，但局部 L2 更低并未转成任务指标更好，10 帧也不能宣称统计显著。valid825 NB0 是 **1.44535253468097**，本轮未跑 valid825；A800 dense adapter 只检验质量，不测稀疏执行速度。

**借入 A 与剩余 X。** 非零 fill／只存例外已是 [Finch tensor formats](https://finch-tensor.org/Finch.jl/stable/docs/tensor_formats/) 的表示能力；重复非零权重共享输入归约已有 [UCNN](https://research.nvidia.com/publication/2018-06_ucnn-exploiting-computational-reuse-deep-neural-networks-weight-repetition)，仿射偏移项亦见 [LUT-GEMM](https://arxiv.org/html/2206.09557v4)。因此均值减法、`c+residual` 和共享 popcount 不是 X。[TASD](https://arxiv.org/html/2403.07953v3) 的结构分量／共享激活驻留、[Bishop](https://arxiv.org/html/2505.12281v1) 的时序打包与 W/输入联合对齐、[VENOM](https://github.com/UDC-GAC/venom) 的 N:M 执行也须给足先验；[QP-SNN](https://arxiv.org/html/2502.05905v2) 的重缩放量化／时空奇异值剪枝并非本次非零 fill 分解。

本次可继续检验的 X 是：**在相同重拟合误差约束下，以真实 `pop−selected` 操作数造成的 H8 CR32 词并集和 SIMD 活跃时间为目标，联合选择 nonzero-fill exception，并把该选择落到源 gate 读响应 collector。** Gram 消融证明这个选择确有局部作用，尚不能证明首创或净性能胜出。下一步只有在同预算 ordinary24／普通34 也获得相同联合选择、更紧凑支持编码及缓存权限后，完整大 conv 的收集与后继服务仍改善，且任务质量维持，才值得升级为硬件研究主线；若继续被普通24支配，就停止该固定格式，保留编解码器与控制作为负结果。

复现：在本目录运行 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 r0_transfer/transfer.py`；r1 原型分别运行 `probe.py`、`response_execution.py`、`fill_chain.py`（压力加 `--stress`）。结果为对应 JSON／log；r0 五个 NPZ 直接交给 `../decomposition_owned/adapter.py` 的 `install_conv`，目标路径写在 `r0_transfer/results.json`。r1 `package.py` 仅替换有效 U 常量，保留原指数。全部脚本／结果局限于本 owned 目录，无训练、EDA 或生产文件改动。
