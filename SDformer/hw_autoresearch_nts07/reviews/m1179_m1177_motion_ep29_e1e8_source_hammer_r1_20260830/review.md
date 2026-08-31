# M1179｜M1177 ep29 E1/E8 source fresh hammer（r1）

## 裁决

**72/100，FAIL_CLOSED；禁止 release、remote、GPU 或 production。**

M1177 的基本方向是对的：ep29 checkpoint/config 常量正确，E1 只有固定的 dyadic 与 hardware-order 两行且没有搜索，E8 只有一个 `build_model` 调用点，Conv/ConvTranspose/Linear、BN、ATLIF、Acc19/24 与 240 KiB 编码逻辑都已搭起；作者测试 11/11，Python 3.10 编译通过，docs/359 未变。

但 fresh hammer 实际构造了三个被接受的攻击，所以 r1 不能进入 release：

1. 任意 `NOT_M1175_ADMISSION` 文件只要 launch 自报 SHA，就被当作 M1175 admission；
2. 任意 profiler/evaluator 文件只要 launch 自报当前 SHA，也会被接受；
3. E1 launch 混入 E8-only cohort 不被拒绝。

此外还有五个 fail-closed 缺口：canonical lease 未强制；40 样本由 launcher 自选而非绑定 sealed cohort；weight/range/BN 只要求非空，不要求 exact layer census 与每层×40 覆盖；BN 四张量可缺省；未来 source-hammer receipt 也没有语义解析。

## 必须修复后再 hammer

- 固定并解析 M1175 exact path/SHA/PASS/ep29 admission；
- 固定 profiler `04f692c...` 与 evaluator `ba40b42c...`，不能让 launch 自签当前字节；
- 强制 canonical `gpu_profile_lease.lock`；
- E1/E8 exact key set，拒 mode-mix 和 unknown keys；
- 绑定 sealed 40-source cohort manifest，逐行验证 path/size/SHA/order；
- 对 Conv/ConvTranspose/Linear/BN/ATLIF 建 exact unique census，并要求 dynamic 每层每样本覆盖；
- BN 必须完整导出 gamma/beta/running_mean/running_var/epsilon；
- 解析并固定 fresh source-hammer 的 PASS、source/contract/test SHA 与 outer seal。

详细机器证据见 `hammer_output.json` 和 `review.json`。r1 保留为失败证据，不得覆盖。
