2026-09-14，缺失强控制迁移与一次条件适配。

1. 忠实迁移：从 rr_modular 保留两context普通RR、完整raw/J/I24、416bit z权限与已付proof；从phase_borrow迁入同一consumer64 ALU复用及consumer固定优先。新增mode3四P13，原mode1/2可逐计数复现。复制宽ALU、多给z口、预先给gold、改变RNE均不允许。
2. 先16个既有fixture、双context压力、159/19197起3tile，以及带断言冷暖跨mode；专门构造相邻tile共享source列的异质输入，使consumer与producer借宽链真实同时请求。正确性和资源账目通过后才进128起64tile。
3. 忠实迁移在64无冲突流有效；异质pair出现条件失败：固定consumer优先让borrow相对modular慢112拍（无BP）/305拍（BP）。以此为依据只做一次适配：在相同宽链上增加consumer与producer组RR，保留两context原RR及mode3。目的为减少关键producer等待，同时完整记入consumer等待。未做参数扫描。
4. 独立adapt_rr保留全部四mode，先小测与双向拒绝holding断言，再64，并在同一个适配模块重跑原三mode控制。残留warm/BP小负结果保留，不能推成家族无效或普遍提速。
5. 仅Verilator4.028 --cc --exe + make；只写本目录。复用已存在全量native数组，不生成第二份大fixture；不做EDA、训练、生产修改或commit。
