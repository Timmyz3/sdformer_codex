# M781：M777 decoder address-timed repair fresh hammer

## 裁决

**FAIL，78/100；身份、三架构服务差异、边界/K-golden、外部流量存在性和 D1 共同 fallback 通过，但物理 residency 仍不闭合，不授权 production replay。**

M777 已经实质修掉 M773 的一半问题：去掉所有 config/transaction 标识后，A1、equal-service K1x8、typed K8 的执行投影仍两两不同；三者保持同一 96 lanes、8 banks、Acc24、245,760 B、3 ns、192 B/cycle 和 dense commit hash，且代码只允许 `K8 / equal-service K1x8`。221,184 B 边界、D3 的 100×768-vector stripe、K24/K25 `[bank 0,row 48] / [bank 1,row 48]` 均正确。逐类删除 source/descriptor/weight-refill 会因依赖缺失而 fail；M712/M722 contributor/group mismatch 也会 fail。D1 三配置均为 16,632,000 次 full-shape compute，保持 diagnostic/non-headline。

但 fresh attack 找到两个新的 P0。

1. **dirty psum 槽存在 overwrite-before-evict hazard。** 容量为一个 vector 的反例中，victim 的 external eviction 最晚到 cycle 15，新 key 对同一物理槽的 `psum_read` 却可在 cycle 11 issue；replacement 没有依赖 evict completion。transaction 列表顺序不能修复该问题，因为 scheduler 允许不同资源的后续请求在更早的逻辑周期 issue。
2. **weight LRU 没有物理 slot。** `WeightResidency` 只保存 `(stripe, output_block, tap, source_tile)` 逻辑 key，不返回九个物理 tile slot。两个不同 output block 的权重都可被报告为 resident、无 eviction，却都从 `bank 0 / local row 0` 读取；外部 refill 也没有对应的 weight-bank write。当前地址 timing 因此没有表示实际的 13,824 B weight SRAM。

另有两个 P1：M722 `a1_storage_plan` 的 storage 字段没有被拿来做守恒，注入 `stripe_count=999` 仍被接受；实际 D3 中 M722 是 2 个 line-buffer stripe、offchip spill 0，而 M777 是 100 个 global-vector stripe并产生 dirty backing。并且 psum evict/restore 只收 external port，没有显式六 bank local read/write；weight refill 同样只收 external port。

## 最小 additive 修复门

1. residency event 必须携带物理 slot；evict 建模为 six-bank psum read → external write，restore 建模为 external read → six-bank psum write，replacement access 必须依赖前者完成。
2. weight LRU 分配 0..8 的物理 slot，所有 bank-local row 经 slot 重映射；refill 必须生成 1R1W local bank writes，并证明不同 output block 不别名、不会 overwrite-before-use。
3. M722 storage 要么与实际 stripe/backing traffic 逐项守恒，要么明确只保留 contributor/group oracle，另加独立、固定身份的 M777 storage oracle。
4. 把上述三个反例加入测试；再做一次 fresh source hammer。此前不得申请 production replay。

本评审只运行冻结 Python 的 14 项 author test、source identity validation 和 synthetic receipt-blind attacks；没有读取 M686/M699 production population、没有生成 decoder cycle/speedup/result、没有运行 RTL/VCS/DC/GPU/remote，也没有修改 source 或 `docs/359`。
