# M219 RTL independent hammer review

结论：**88/100，P0=0，GO matched service-only DC with strict scope**。缺 M218↔M219
cross-module miter、全幅 INT8/Acc24 动态边界、L4/有限存储 recurrence 均为
P1：不阻止独立 M219 服务逻辑综合，但它们严格阻止 `4.952×`、fair
area/energy、complete FC2/FFN、physical/system/headline admission。

我独立校验了 sealed VCS 的 61 个 manifest 条目和 contract/RTL/SVA/TB exact
SHA，`docs/359_DATE终局冻结_20260813.md` 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
再用 Synopsys VCS V-2023.12-SP1、seed 219997 独立编译/仿真，RC=0/0，
无 compile warning/error marker、无 assertion failure；重现 144 group / 864 request /
864 response / 864 context write / 864 active-bank read / 102 result / 5 done。
FIFO4、O8、same-edge slot/context reuse、OOO、stall、flush quarantine、identity/
duplicate/timeout 均有命中。不过 TB 无随机激励，换 seed 不会改变时序。

最重要的攻击结果：TB 里所谓的 K8 snapshot 并没有实例化 M218。
`send_group(mask=8'hff)` 会把 8 个 bit 各自发成一个 M219 K1 group，因此
capture mode 1/2 是两次 K1 重放，只证明不同 block interleave 的加法结果
一致，不是 M218 K8 对 M219 K1 的 cross-module 证明。

M219 在结构上确实保留 O8/FIFO4、18,432-bit context、epoch16/gen32/
flush1024，并裁到单个 128-bit response 和单源 signed-INT8 累加。3-bit
bank ID 可表示 8 个外部物理 bank，但 bank 容量、macro、8-to-1 选择路径和
能耗仍在模块之外。因此下一步只允许与 M218 同条件的 3 ns、ideal-clock、
ZeroWireload、zero-macro service-only DC，用来量化内部逻辑/寄存器敏感性。

完整评分、P1、DC gate 和 claim boundary 见
`m219_rtl_independent_hammer_review_r1.json`；重算数据见
`m219_independent_recompute.json`；独立 VCS/SHA 脚本为
`run_independent_vcs_and_audit.sh`。
