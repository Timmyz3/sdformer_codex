# M667｜M665 decoder polyphase mapper 独立打铁评审

## 裁决

**78/100，P0=0、P1=3、P2=1，`NO_GO_M660_PAYLOAD_INTEGRATION__M665_REPAIR_REQUIRED`。**

映射算术核心是可靠的：本评审没有把作者测试当 oracle，而是从 M514 RTL 重新导出九个 tap 的槽位顺序，以三组新 seed 和非方形尺寸重建嵌套循环 NumPy `ConvTranspose2d` oracle；在可用的 PyTorch 3.10 环境中又逐元素对上 `torch.nn.functional.conv_transpose2d`。四相位、所有 destination/tap/channel K 槽、上下左右边界、tile 尾部及 popcount/product 守恒全部通过。

但 M666 request 的授权条件是 P0=0 且 P1=0。独立攻击确认三个 payload 集成级 P1，因此当前不得把 M665 接入 canonical M660，不得据此运行周期模型或形成性能数字。

## 三个 P1

1. **shape 乘积发生 int64 溢出。** `[2^32,1,2^32,1,1]` 的各维均通过正整数检查，但 `np.int64` 乘积绕回 0，空文件遂通过 byte-length 与 tail 检查。修复应在 NumPy cast 前用 Python arbitrary-precision 整数累乘，拒绝 bool，并设置逐维与总元素显式上限。
2. **父目录 symlink 可逃出包根。** 当前只拒绝 leaf symlink；词法安全的 `alias/x` 若 `alias` 是软链接，仍可读取 sealed package 外部文件。修复需对 root 和 candidate 做 resolve/relative-to 检查，并拒绝路径链任一 symlink。
3. **route/list/module 未形成闭合关系。** `d0_d2_d3_binary_records` 中放入 module 3 的 `EXACT_SCALED_BINARY_BITPACK` 行，会被接受并消费 `theta_binary_candidate`。应强制 D0/D2/D3 list 只能是 `{0,2,3}+EXACT_BINARY+row.input`，D1 list 只能是 `{1}+EXACT_SCALED+theta_binary_candidate`。

另有一个 P2：Python 的 `bool` 是 `int` 子类，导致 `tile_m=True`、`phases=(True,)`、`output_channels=True` 被接受。它不改变已确认的正常映射算术，但应统一改成显式 non-bool integral validator。

## 独立验证覆盖

- 两套 Python 均完成 hammer 11/11：Python 3.10.18 + NumPy 2.1.2 + Torch 2.7.1（CPU），以及 Python 3.6.8 + NumPy 1.19.5（无 Torch，运行独立 NumPy oracle）。
- 新 seed：667003、667021、667089；输入覆盖 `[T,C,H,W]` 的 `1x1x1x5`、`2x2x2x5`、`3x3x4x2` 与 2/3/4 个输出通道。
- 从 RTL 独立导出的 slot 顺序为 `(0,0),(0,2),(2,0),(2,2),(0,1),(2,1),(1,0),(1,2),(1,1)`；phase 顺序为 `3,2,1,0`。
- 边界 fixture 闭合 43 个 active-tap event、6 个 source popcount 和 `43*7=301` 个 product；tile_m 覆盖 1、2、M-1、M、M+1、>M，无 gap、overlap 或 OOB。
- 攻击覆盖 little/big bit、C/K order、tail/short payload、batch/shape、kernel/stride/padding/output-padding/dilation/groups、weight layout、重复 phase、schema/packing/route、路径遍历、绝对路径、leaf/parent symlink、重复 JSON key、重复 record 和 shape overflow。
- 作者固定测试另行重跑为 `17 passed`，只作回归佐证，不作为独立 oracle。

## Claim boundary 与后续

本评审只准入“正常输入下映射算术得到强正证据”；不准入 production payload、cycle、speedup、energy、PPA 或 DATE headline。没有运行 GPU、VCS、DC、PTPX 或其他 EDA，没有修改作者文件或 `docs/359`。

唯一允许的下一步是：修复三个 P1，更新 mapper/tests/contract 身份并双封作者交接，再发起新的 fresh hammer。修复前不得消费 M660 payload 集成授权。
