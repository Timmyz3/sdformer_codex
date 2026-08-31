# M773：M768 decoder address-timed source fresh hammer

## 裁决

**FAIL；只通过身份与基础 scheduler 单元语义，不授权一次 production replay。**

M686/M692 与 M699/M705 的 member/outer seal、40/120 record lattice、contract/analyzer/tests SHA、contract 双层 sidecar、M672 mapper 身份和 `docs/359` 均通过。作者测试在冻结 Python 下为 `16 passed`；独立攻击也确认 same-bank/不同 bank、1RW/1R1W、outstanding=1 同拍回收、未解析 dependency 和容量 manifest cliff 的基础行为正确。

但 fresh hammer 找到 3 个 P0。当前执行器不能生成可引用的 decoder component 周期：

1. `iter_record_transactions` 对 A1-OSG、equal-service K1x8、typed signed K8 没有不同的 work/descriptor 分支。synthetic exact-mapper 小例的非标签事务投影完全相同，三者均为 29 cycle。此时所谓 K8 对 K1x8 只能是同一路径换标签。
2. 245,760 B 只在资源 manifest 中守恒，地址 scheduler 不检查 residency。超出 221,184 B psum 分区的地址仍被接受；D3 dense psum 地址跨度为 29,491,200 B，即分区的 133.33×，代码没有 stripe、evict、restore 或 external backing transaction。
3. frozen weight layout 要求 `bank=source_channel mod 8`、bank-local address 按 `tap → source_channel_div_8 → slice`。对 flattened K 24/25，代码生成 `[384,400]`，合同布局应为 `[48,48]`。

## 次级缺口

- exact-binary route 不收费 source/descriptor external read，也没有 13,824 B weight buffer 的 refill；因此还不能声称 memory timing/DRAM bytes closed。
- M712/M722R2 只验 SHA，未成为 contributor multiset、stripe/storage traffic 的 executable oracle；`source_flat_index` 在生成器中赋值后未使用。
- D1 虽被正确锁为 diagnostic/common fallback，但 compute 永远只收一个请求，与层几何无关，不能作为可信 dense-FP32 周期。
- `cycle_classes` 的和确实等于 total cycles，但任一资源发请求就记作 active service、任一 inflight 就可能记 dependency，尚不是可信的逐端口瓶颈拆分。

## 最小修复门

只修一套 M768，不另开 decoder matcher：

1. 明确实现 A1、K1×8、K8 三条不同的 descriptor/work 构造，K8 只对 equal-service K1×8；三者保留相同 96 lanes、245,760 B、Acc24、3 ns、192 B/cycle、commit hash 和 fallback。
2. 修正 flattened-K bank-local row；加入 contributor/address/hash 与 M672/M712 oracle 的守恒检查。
3. 实现真实 bounded residency、stripe/refill/spill/restore，并收费 binary source/descriptor 与 weight refill。
4. D1 要么实现共同收费的真实 fallback，要么不输出 D1 cycle，继续保持 `decoder_complete=false`。
5. additive source repair 再经 fresh hammer 通过后，才可申请唯一一次 production replay。

本评审没有运行 M686/M699 production population，没有生成 cycle/speedup，没有运行 RTL/VCS/DC/GPU/remote，也没有修改 M768 source 或 `docs/359`。
