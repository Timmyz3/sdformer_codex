# M1638 decoder compact L2 session-configuration-bound successor｜作者收据

状态：`PASS_AUTHOR_M1629_CONFIGURATION_RELABEL_P1_SOURCE_REPAIR__M1639_DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_EXECUTION`

M1638 不覆盖 M1628，只继承其 request/state/cache/ledger 实现，并重建 session 构造、finish authentication 与 bundle coverage policy。修改范围仅针对 M1629 的 configuration relabel P1。

隐藏 `issued` registry 现在保存 `(exact owner, immutable initial_configuration)`。`accept_request_pair`、`accept_destination_pair`、`_finish_payload`、`finish` 与 bundle receipt inspection 都必须重新核对该初始绑定。finish HMAC payload 的 configuration 必须与 registry 初值相同。

bundle 的 coverage policy 由“只要求 dense 行为第一行成立”收紧为精确三行：`DENSE_TYPED_K8=(true,true)`，`BIT_EQUAL_SERVICE_K1X8=(false,false)`，`BIT_TYPED_K8=(false,false)`。三个实际均完整执行 dense 配置、在 finish 前把后两个 owner 重标为 bit 配置的攻击，现在在 finish 层稳定拒绝。

新测试继承 M1628 的 18 类检查，因此原 8 类 survivor、clone/tag/duplicate/order/shared-commit/replay 仍全部拒绝；另增加 relabel 回归与 request/state/payload/finish 四层绑定回归，总计 CPython 3.6/3.10 各 20/20 PASS。两版本 `--describe` 输出 byte-identical。

本包没有命名或打开 ep34 payload，没有 actual L2/L3、attempt、release、GPU 或 EDA。M1639 不同作者评审前，禁止 actual runner source 与 M1640 release，也不产生周期、流量、加速、能量、Table-A 或论文结果。
