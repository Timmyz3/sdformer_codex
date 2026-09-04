# M790：M785 decoder physical-residency source fresh hammer

## 裁决

**PASS，100/100，P0/P1/P2=0；M781 的 physical-residency 修复在 source-only synthetic 范围内闭合。M785 可以申请一份新的 additive production release，但本评审本身不授权 replay，也不产生或准入 cycle/speedup。**

冻结身份为 analyzer `7fbd72d2...`、tests `1ec8730c...`、独立 storage oracle `422da36a...`、contract `612a2ba3...`、request manifest `3d2abc6a...`。14 项作者回归、synthetic self-test 和 sealed source validation 均通过；`docs/359` 仍为 `dedde7ce...`。

## M781 P0/P1 复验

1. **Dirty psum 物理槽复用已闭合。** 容量一、victim 延迟到 cycle 37 的反例中，victim local write return=38，六 bank evict local-read 为 38→40，external evict 为 40→43，replacement read 到 cycle 44 才 issue。进一步的 evict+restore 攻击严格得到 `psum_read → external_write → external_read → psum_write`，每一级首拍都不早于上一级末拍 return；1RW psum 本地端口和外部端口延迟均收费。
2. **Weight 物理 slot/别名已闭合。** 九个 key 与 slot 0..8 保持双向一一映射，第十 key 只复用已退休的 slot 0；同 bank 的不同 slot 地址相差 192 B，不发生并存内容别名。一次 refill 为 8×192 B external beat 和 12×8×16 B local write，读严格等待第 12 次 local write return。
3. **逐 key/逐 bank last-use 已在集成生成器上闭合。** 10 个 output block 的真实 synthetic transaction construction 强制产生 181 次 slot overwrite refill；每次 local overwrite 都携带旧 key 各 bank 的最后 read token，且复用槽的新 read 等待 local terminal token。该结果不是仅调用 helper 得到。
4. **两个不等价 storage 对象已正确分开。** M722 只作 contributor/group oracle，但其 line-buffer plan 仍按完整字段严格复算；对 stripe count、stripe list、total bytes、offchip spill 的注入全部失败。M785 独立 global-vector oracle 对 stripe count/list、psum partition、offchip span 的注入也全部失败。

D3 独立复算为 768 vector/full stripe、100 stripe、221,184 B psum partition、22,118,400 B direct-key backing address span；明确不等价于 M722 的 line buffer。

## 公平性与边界

删除 config/population/transaction/request 身份后，A1、equal-service K1×8、typed K8 三条 service projection 仍两两不同，dense commit hash 相同。三者共用 96 lanes、245,760 B macro-rounded SRAM、Acc24、3 ns、192 B/cycle 和 resource SHA `a7400bdd...`。唯一合法倍率仍是 typed K8 / equal-service K1×8；K8/A1 与 K8/single-K1 均禁止。

D1 三配置均为共同的 16,632,000-count full-shape diagnostic fallback，headline=false。source-only pass 不表示 decoder complete、full-network complete、Table-A 或 system speedup。

## 授权边界

本 hammer 没有读取 M686/M699 production population，没有生成 decoder cycle/speedup/result，没有运行 RTL/VCS/DC/GPU/remote，也没有修改 M785 source/production 或 `docs/359`。下一步必须另封 exact-identity additive release；该 release 才可授权唯一一次 production replay。
