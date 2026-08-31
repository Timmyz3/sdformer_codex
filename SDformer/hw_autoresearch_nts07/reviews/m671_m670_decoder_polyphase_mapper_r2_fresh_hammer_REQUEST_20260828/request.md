# M671 fresh hammer 请求｜M670/M665-r2

请由未参与 r2 作者实现的独立 reviewer 审查冻结 mapper `875b31ed...`、pytest `39c4fd84...`、dual smoke `8c77912a...` 与 contract `b4e0c7ab...`。

必须独立复现 M667 的三个 P1 与一个 P2 已被关闭，同时重新构建 M514 phase/tap/K 与 PyTorch ConvTranspose oracle，不得将作者测试作为 oracle。重点攻击 Python 大整数边界、trusted package root 的每级 symlink/escape、完整 S10×4 container/module/name/route/identity lattice，以及所有 bool/np.bool 整数入口。

只有 P0=0、P1=0 可返回 `GO_M660_PAYLOAD_INTEGRATION_ONLY`。本请求不授权 GPU、EDA、production mapping 或任何性能数字；不得修改 r1、M667、`docs/359` 或 M670 作者文件。
