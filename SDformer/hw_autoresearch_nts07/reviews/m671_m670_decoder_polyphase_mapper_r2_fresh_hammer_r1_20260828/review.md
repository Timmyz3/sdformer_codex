# M671｜M670 decoder polyphase mapper r2 fresh hammer

结论：**NO-GO，P0=0 / P1=2 / P2=0，84/100。** 不能进入 M660 payload integration。

数值核心通过。本评审从 M514 RTL 的 `destination=2*source-1+kernel`、九 tap phase-major 顺序重新建立 oracle，而非把作者测试当 oracle。三组新 seed、三种非方形尺寸在两套 Python/NumPy 下逐元素对上独立嵌套循环；Python 3.10 下又对上三组 PyTorch CPU `ConvTranspose2d`。四相位、所有 destination/tap/channel K 槽、tile 尾部、边界、popcount 与 product 守恒均闭合。作者 32 个 pytest 与双 Python smoke 也全部通过。

但还剩两个会让“已验证的文件”与“实际消费的文件”分离的 P1：

1. `validate_bitpack` 返回可信路径后，`iter_polyphase_tiles` 的 `np.memmap` 和 `_unpack_input` 的 `np.fromfile` 仍使用调用者原始路径。攻击中 package 下全零文件通过验证，而 CWD 同名全一文件被实际消费，得到 mapped sum 46、source popcount 8。验证后同尺寸替换也能改变实际消费 SHA。
2. `_trusted_root` 只 `lstat` 最终 root，root 祖先目录若是 symlink 仍被接受，未满足作者合同和请求中的 every-parent-symlink 约束。

最小修复是所有消费路径统一使用 validated absolute path，最好把 no-follow 打开的 fd、inode、SHA 和实际读取绑定为一次身份；trusted root 则从 filesystem anchor 到 leaf 逐组件 `lstat`，明确系统父目录是否允许 symlink，并将 resolve 后 root 身份固定给所有候选。

其余攻击均通过：Python 大整数总量门、零/负数、所有公共整数接口的 `bool`/`np.bool`、leaf/候选 parent symlink、traversal/absolute escape、directory/missing leaf、非直属 manifest、duplicate JSON key，以及 D0/D2/D3↔D1 的 container/module/name/route/identity-field 完整 S10×4 交叉。

本评审未运行 GPU、EDA、production mapping，未生成 cycle、speedup 或任何性能数字，未修改 r1、M667、作者 r2 artifacts 或 docs/359。
