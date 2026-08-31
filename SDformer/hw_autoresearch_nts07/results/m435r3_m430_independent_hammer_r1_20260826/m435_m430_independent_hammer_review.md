# M435 M430 独立打铁评审

**评分：93/100；结论：带资源限定通过。**

- GO：生成 full-population RTL stimulus，并进入保留 persistent old_psum 的合法 dual co-read 功能 RTL。
- NO-GO：把 1.435375x 写成同资源或系统加速；144/160 B/cycle 新端口必须做 DC/PT/存储与互联归一。
- 独立重算：51.84M held-out rows、17,280 phases、逐 phase timestamp 全部 0 mismatch。
- 精确周期：M430 517,041,352；strong-zero 742,148,386；M423+dual 527,837,132；M401 serial 641,790,704。
- 语义红线：不做 seed-first-correction fusion；`old_psum += PWP + correction`。

P1 是资源不匹配；P2 是 one-shot 历史只能由封存/marker/时间序辅证，以及废弃候选的独立生成只抽查 64/1728 partition。
