# M454 对 M450 fixed160 exact co-pack 的独立打铁结论

评分 **96/100**，P0=0、P1=0、P2=2。M450 的负向筛选成立：在冻结的 fixed160 接口下，PWP descriptor 的 16 B wide slack 或 64 B narrow slack 都装不下一条完整 exact correction vector，因此 atomic co-pack 候选为 0，周期严格维持 **517,041,352，1.000000x**。该方向应当关停，不进入 RTL/DC/Formality/PT。

## 独立重算

独立审计器没有读取 raw M40 payload，逐一枚举 442,368 条冻结权重/correction vector，重现 signed-width histogram：6 bit 为 6 条、7 bit 为 70,724 条、8 bit 为 371,638 条。完整 vector 的 payload 范围为 72–96 B；即使最小 72 B 也大于 narrow slack 64 B，故不可能存在完整原子同拍打包。

独立程序只遍历一次冻结 M430 phase CSV，17,280 个 phase 与全部聚合量均为 0 mismatch。M430 和 fixed160 均为 517,041,352 cycle。

| 口径 | cycle | 相对 M430 | 是否可执行 |
|---|---:|---:|---|
| fixed160 atomic co-pack | 517,041,352 | 1.000000x | 是，但无收益 |
| 96 B global fragment pool | 486,694,570 | 1.062353x | 否 |
| 所有 vector 强制为最小 72 B | 476,578,975 | 1.084902x | 否，且是故意不可能的上界 |

后两点把不同 descriptor/phase 的空闲字节全局汇总，没有可执行 catalog、buffer、端口协议或 scheduler；72 B 点还把 442,368 条 vector 全部假定为观测到的最小宽度。二者只能用于 falsification：即使如此激进，仍达不到预冻结的 1.10x 继续阈值。不得把 1.062353x 或 1.084902x 写成 M450 已实现性能。

## 证据边界

M450 contract 的 raw M40 输入列表为空，analyzer AST 没有 raw packed/value-payload 路径字面量，本次独立运行的 raw M40 read 为 0；这些足以支持可复跑路径未重读 raw M40。历史已结束进程的真实 I/O 不能仅凭封存产物反演，因此不作超出证据的法证式断言。

M430 catalog 审计前后均为 `3ff522ff2296a021b005ca5733d846cc169560c125c8713c814b22a14d372f78`；`docs/359` 审计前后均为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。M430、M442、M449、M450 的 seal 自验证均从各自结果目录 cwd 执行并得到 0 mismatch。

## 去留

- M450 的分析方法和 NO-GO 结论：GO。
- fixed160 atomic exact co-pack RTL：NO-GO。
- 96 B / 72 B fragment ceiling：只保留为非执行上限，禁止进入执行性能表。
- 不改 catalog、不扩端口、不新增 scheduler；这些属于新架构合同，不是 M450 的“免费 slack”。
- 保留 M430 K1 separate 与 M433 admitted point。

