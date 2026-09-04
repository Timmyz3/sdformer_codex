# M677｜M672 decoder polyphase mapper r3 fresh hammer

结论：**GO，P0=0 / P1=0 / P2=0，98/100。** 准入范围严格限定为
`GO_M660_PAYLOAD_INTEGRATION_ONLY`。

M671 的两处阻断已关闭。第一，r3 先由 r2 验证 bitpack，再把返回的
package 内绝对路径传给每个数据消费入口。独立攻击在 package 放全零
`same.bitpack`、在 CWD 放同名全一文件；`iter_polyphase_tiles`、
`materialize_polyphase`、`reconstruct_convtranspose` 和
`workload_accounting` 分别得到零 tile、零矩阵、零输出与
`source_popcount=0`。运行时边界探针确认四个 r2 入口收到的均为同一个
已验证绝对路径。M660 另以“CWD 恶意空 manifest + 同名全一 payload”攻击，
package manifest 仍返回完整 40 条记录，所有 payload 路径均是 package 下的
绝对路径且 SHA 与实际文件一致。

第二，r3 从文件系统 anchor 到 trusted root leaf 对每个 lexical component
执行 `lstat`。以祖先目录 symlink 构造的 root 在 validate、iterator、
materialize、reconstruct、accounting、direct trusted-file 和 M660 manifest
共七个公开入口全部拒绝；探针确认拒绝发生在任何 r2 数据调用之前。

数值回归没有把作者测试作为 oracle。三组新 seed、三种非方形输入重新按
`destination=2*source-1+kernel` 建立嵌套循环 ConvTranspose oracle，逐元素
一致，active-tap/product 守恒。冻结 r2 的 32 个 pytest、r3 的 9 个 pytest
均通过；r2 dual smoke 与本评审独立 hammer 在 Python 3.10 / NumPy 2.1.2
和 Python 3.6 / NumPy 1.19.5 两套环境均通过。

边界必须保留：r3 明确假设一次 evaluation 期间 trusted package 不可变，
并不声称抵抗 hostile concurrent replacement。若后续 integration 允许并发
写入或不可信 producer，应升级为 no-follow 打开的 FD/inode/SHA 与实际读取
绑定；这不影响当前冻结、只读 package 范围内的准入。

本评审未运行 GPU、EDA、production M660 mapping，未产生 cycle、speedup、
energy、PPA 或 paper-headline 数字；未修改 M672/M670 作者文件或 docs/359。

