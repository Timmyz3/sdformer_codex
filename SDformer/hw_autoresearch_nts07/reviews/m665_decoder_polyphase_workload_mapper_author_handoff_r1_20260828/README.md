# M665｜decoder polyphase workload mapper 作者交接

## 裁决

`STATIC_AUTHOR_HANDOFF_ONLY__FRESH_HAMMER_REQUIRED`。

本轮新增的是 M514 K3/S2/P1/OP1 `ConvTranspose2d` 的 exact CPU workload 输入层。它把单条 little-bit、C-order `[T,1,C,H,W]` activation bitpack 映成四个 destination-parity `[T,M,K]` 矩阵，并提供分块流式迭代；没有读取或绑定尚未完成的 M660 canonical 结果，没有运行 GPU、VCS、DC 或任何 EDA，也没有生成周期或加速倍率。

## 映射合同

- M514 phase 顺序固定为 bank `3,2,1,0`，即 destination parity `11,10,01,00`。
- phase 内 tap 顺序固定为：
  - `11`: `(0,0),(0,2),(2,0),(2,2)`；
  - `10`: `(0,1),(2,1)`；
  - `01`: `(1,0),(1,2)`；
  - `00`: `(1,1)`。
- K 顺序固定为 `phase -> tap -> source channel`；mapper 明确拒绝 channel-major/K-order 漂移。
- 每个 phase 的 `M=H*W`；`K=phase_tap_count*C`。目的坐标在该 parity 内 row-major。
- 逆坐标严格用 `source=(destination+1-kernel_index)/2`；越界源坐标成为结构零，不物化 zero-expanded image。
- 权重布局固定为 PyTorch `ConvTranspose2d` 的 `[Cin,Cout,Ky,Kx]`；bias 不属于 workload mapper。

## 新文件身份

- mapper：`hw_autoresearch_nts07/system_simulator/scripts/map_m665_decoder_convtranspose_polyphase_workload.py`
  - SHA256 `07dd6474764993add120091514334deb02c5a71caa0c9955b85d8f577634abd4`
- tests：`hw_autoresearch_nts07/system_simulator/tests/test_m665_decoder_convtranspose_polyphase_workload.py`
  - SHA256 `736056eade13dec69039122813e78afe682ee132ab9c9fdaee0c0f103c0cc280`
- contract：`hw_autoresearch_nts07/contracts/m665_decoder_convtranspose_polyphase_workload_mapper_contract_r1_20260828.json`
  - SHA256 `52eb24ec95b63fd21ed4b9af7eb8e5e584af039e4fb4987d19812102e3962299`

冻结 M514 RTL、VCS contract、VCS receipt 分别为 `90c44fc...`、`60e4fe59...`、`aa6fb4d6...`；`docs/359` 保持 `dedde7ce...`。

## 作者验证

固定 PyTorch Python 的 CPU-only suite 为 `17 passed`。三组随机二值 mask 与随机整数权重均与 `torch.nn.functional.conv_transpose2d` 全输出逐元素相等。另有独立守恒测试闭合 source popcount、valid tap、active tap 和 output-channel product 数；流式 M 分块拼接与一次性 materialize 相同。

负测覆盖 big-bit、非 C-order、shape/batch、byte length、tail padding、K-order、kernel/stride/padding/output-padding/dilation/groups、weight layout、symlink 与 M660 schema/packing/route 漂移。

## M660 集成边界

mapper 只声明预期 schema `m660_h67_ep35_layer_static_decoder_payload_v1`，接受 `EXACT_BINARY_BITPACK` 与 `EXACT_SCALED_BINARY_BITPACK`；`COMMON_FP32_DENSE_FALLBACK` 不进入二值 mapper。没有预写或绑定任何未完成 M660 manifest/payload SHA。

只有 canonical M660 结果和 fresh result hammer 都存在后，后续集成合同才能绑定其 double seal，再生成官方 Prosperity replay 和公平 B0/B1/K1x8/K8/Ours 周期输入。

## Claim boundary

本轮只准入 exact CPU input mapping 源码与作者测试。`production_workload_mapped=false`、`cycles=false`、`speedup=false`、`rtl=false`、`eda=false`、`date_headline=false`。fresh independent hammer 必须 P0=0、P1=0 才可进入 payload 集成。
