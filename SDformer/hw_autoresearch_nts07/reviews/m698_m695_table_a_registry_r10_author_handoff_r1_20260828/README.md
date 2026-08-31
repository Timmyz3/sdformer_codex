# M698｜M695 Table-A production registry r10 author handoff

## 结论

M698 以 additive r10 修复 M695 对 M691 r9 的五个 P1。作者测试 12/12 通过，
并明确覆盖 M695 未被旧测试触达的 `b0_dense96_fixed_t10` 强制行。

r10 把两个状态严格分开：

1. `structural_evidence_pass` 只说明全部原始证据能独立交叉；
2. `production authority` 必须由后续 additive builder 用 fresh hammer 的精确 SHA
   钉死。

r10 内的 production authority allowlist 故意为空。因此 synthetic fixture 即便通过
完整 grammar，也不能成为 production row；任意自写 `P0=P1=0` review JSON 同样不能
获得信任。canonical 输出保持 `production_runs=0 authority=0 bundles=0 eligible=0`，
headline 与 analytical 均为 false。

## 五个 P1 的闭合

- P1-01：工具证据绑定 installed path、realpath、device、inode、build ID、snapshot
  SHA、`/proc/exe` SHA 与 raw version SHA；五类工具 build/hash 必须互异。DB 另需
  DC native-read 记录、真实 library/corner 身份、cell/opcond/voltage/time-unit census。
- P1-02：八个 step 的 executable snapshot path 必须与 `argv[0]` 完全相等，SHA
  必须回到对应 logical tool；`vcs_run` 必须回到本次 exact simv。
- P1-03：十算子各自必须有正的 sequential/combinational/leaf census，并逐层与 top
  求和；Formality 必须绑定 exact RTL/netlist SHA、正 compare points 与零 unmatched。
  native 模式拒绝 RTL==netlist、`wire alive` stub，以及没有 mapped stdcell token 的模块。
- P1-04：annotation 分子来自显式 PT map rows，net source 必须存在于真实 SAIF TC；
  分母来自独立 mapped-design census，native SAIF 至少 100 个 distinct TC net，net/pin
  覆盖率都不得低于 95%。
- P1-05：netlist hierarchy、DC area split 与 PTPX per-instance 三份报告独立交叉同一
  8 weight + 8 state + 1 parent SRAM；DC 强制 `total=logic+macro`，PTPX 强制逐实例
  `total=internal+switching+leakage` 并与 SRAM 总功耗相等。

## 作者验证

- Python compile：PASS。
- unittest：12/12 PASS。
- canonical builder：
  `M698_REGISTRY_PASS production_runs=0 authority=0 bundles=0 eligible=0 headline=false analytical=false`。
- M695 攻击复现：`/bin/true` 多工具冒充、executable/argv0 解耦、ELF bytes 冒充 DB、
  十个 wire stub、one-TC 自报高覆盖、缺宏/错误面积等式、任意 self-authored authority
  均被拒绝。
- EDA/GPU/remote/training：均未运行。
- `docs/359` SHA 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## Fresh hammer 决策门

本交接包只是 author handoff。只有 fresh receipt-blind r10 hammer 达到 `P0=0,P1=0`
后，后续真实 native run 才可以另建 additive authority-pinning revision。不得原地修改
r10 allowlist，也不得用本交接包声称任何 Table-A PPA、功耗、系统倍速或 headline。
